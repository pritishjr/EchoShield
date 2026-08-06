
"""
app/workers/transcribe.py

Runs INSIDE a worker process — never inside the async event loop.
Owns two things:

1. The process-local model slot (set once by init_worker.py, read on
   every task after that).
2. transcribe_and_redact() — the function actually submitted to the
   ProcessPoolExecutor per audio chunk via
   `loop.run_in_executor(pool, transcribe_and_redact, audio_bytes)`.

Because this module is only ever imported inside worker processes,
each worker owns its own private copy of `_model`. There is no shared
state and no locking required between sibling workers.

All of this happens per chunk inside a worker process concurrently
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.redaction.patterns import redact_text

logger = logging.getLogger("worker.transcribe")

# Set once by init_worker.init_worker(), read many times after that.
_model = None


def set_model(model) -> None:
    """Called exactly once by init_worker.py, right after the model
    finishes loading for this process."""
    global _model
    _model = model


def get_model():
    """Exposed mainly for tests — lets a test inject a stub model
    directly, without going through the full executor + initializer
    machinery."""
    return _model


@dataclass
class TranscriptionResult:
    """
    Return type of transcribe_and_redact(). This crosses the process
    boundary back to the event loop via pickling, so keep it a plain,
    simple, picklable data container — no live objects (models,
    sockets, file handles) belong on this class.

    `redacted_text` is the only field safe to forward to the client.
    `raw_text` is included for server-side audit/QA logging only —
    it still contains unredacted PII, and it is the responsibility of
    services/pipeline.py (not this module) to make sure raw_text never
    reaches the WebSocket output.
    """
    redacted_text: str
    raw_text: str
    duration_ms: float
    error: Optional[str] = None


# --- audio decoding assumption ---------------------------------------
# The client (scripts/simulate_client.py) streams raw, headerless
# PCM16LE mono samples at SAMPLE_RATE per chunk — not a wav-wrapped
# chunk every time. If your client instead sends a fully wav-encoded
# blob per chunk, decode with soundfile.read() on a BytesIO buffer
# here instead of the raw frombuffer call below.
#
# TODO: currently duplicated as a bare constant. Once config.py exists,
# this should be sourced from Settings and threaded through initargs
# the same way model_name/device/compute_type are in init_worker.py.
SAMPLE_RATE = 16_000


def _decode_pcm16(audio_bytes: bytes) -> np.ndarray:
    """Raw PCM16LE bytes -> float32 array in [-1.0, 1.0], the format
    faster-whisper expects when handed a numpy array directly (as
    opposed to a file path)."""
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm16.astype(np.float32) / 32768.0


def transcribe_and_redact(audio_bytes: bytes) -> TranscriptionResult: 
    """
    The function the ProcessPoolExecutor actually runs per chunk.

    Steps, entirely inside this worker process:
      1. Decode raw PCM bytes into the float32 array Whisper expects.
      2. Transcribe using the model this worker loaded at startup.
      3. Redact PII from the resulting text.
      4. Return a picklable result.

    Per-chunk failures are CAUGHT and returned via the `error` field,
    never raised. A single malformed or silent chunk must not kill
    this worker process — that would shrink pool capacity and force
    a respawn mid-stream. Fatal failure is reserved for init_worker's
    model-load step only; once a worker is alive, it stays alive.
    """
    start = time.perf_counter() #for run-time calculations

    if _model is None:
        # Should be unreachable if init_worker succeeded, but guard
        # explicitly rather than segfaulting into an AttributeError
        # deep inside faster_whisper's internals.
        return TranscriptionResult(
            redacted_text="",
            raw_text="",
            duration_ms=0.0,
            error="worker has no model loaded",
        )

    try:
        #converting the raw 16-bit PCM bytes into 32 bit floating-point audio numpy array: (defined earlier)
        audio_array = _decode_pcm16(audio_bytes)

        """FAST INFERENCE: (faster-whisper)"""
        # beam_size=1 (greedy decode) trades a little accuracy for
        # latency — appropriate for a ~1-second streaming chunk, where
        # the next chunk will supply more context anyway. Revisit if
        # per-chunk transcript quality proves too noisy in practice.
        segments, _info = _model.transcribe(
            audio_array,
            language="en",
            beam_size=1,
            vad_filter=False,  # chunks are already pre-segmented by the caller
        )
        raw_text = " ".join(segment.text.strip() for segment in segments).strip() #extract raw text (stays in memory?)
        redacted_text = redact_text(raw_text) #redact raw text

        #high accuracy latency (run-time) computation using time (ms)
        duration_ms = (time.perf_counter() - start) * 1000
        
        return TranscriptionResult(
            redacted_text=redacted_text,
            raw_text=raw_text,
            duration_ms=duration_ms,
        )

    except Exception as exc:  # noqa: BLE001 — intentionally broad; see docstring
        logger.exception("transcribe_and_redact failed on a chunk")
        duration_ms = (time.perf_counter() - start) * 1000
        return TranscriptionResult(
            redacted_text="",
            raw_text="",
            duration_ms=duration_ms,
            error=str(exc),
        )
        

#so <transcribe_and_redact> returns a TranscriptionResult data object that stores not only the redacted text but also the raw text. (see notes)
#this is for server-side quality assurance and auditing purposes. this data must complly with certain data security compliances.