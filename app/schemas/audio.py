
    
"""
app/schemas/audio.py

The wire contract for this WebSocket endpoint: the exact, deliberate
shape of every message the server sends to a client. Nothing in this
file talks to audio, models, or caches — its only job is to define
what's allowed to cross the network boundary, and how.

Why this exists as a layer separate from workers/transcribe.py's
TranscriptionResult:

TranscriptionResult is an INTERNAL type. Its shape is driven by what a
worker process happens to compute, and by what needs to survive a
pickle round-trip across a process boundary (see its own docstring).
Those are implementation concerns — not the same question as "what
are we willing to promise an external client this field will always
mean." Serializing TranscriptionResult directly onto the wire (e.g.
via dataclasses.asdict()) silently welds the two together: any field
added to TranscriptionResult later for an internal/debugging reason
would automatically start appearing in client-facing JSON, whether or
not anyone actually decided that was safe or desirable to expose.
TranscriptResponse.from_result() below is the one deliberate seam
where that translation happens — adding a field to TranscriptionResult
does NOT put it on the wire until someone explicitly adds it here too.

Why a discriminated union, not "just send whatever dict shape fits":

This endpoint emits more than one kind of message — a successful
transcript, or an error for a chunk that failed. A client parsing
these needs a reliable way to tell them apart before it knows which
shape to expect; that's what the `type` literal field is for.
Pydantic's discriminated unions use that field to pick the right model
automatically. That matters most on a READ path (validating an
incoming heterogeneous message), which this protocol doesn't have yet
— today the client only ever sends raw binary, no JSON. But it's worth
establishing the pattern now: if this protocol ever needs to validate
something the client sends (e.g. a handshake message before the
binary stream starts, declaring sample rate or format), this file is
exactly where that model gets added, using the same shape already set
up here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from app.workers.transcribe import TranscriptionResult


class TranscriptResponse(BaseModel):
    """
    Sent for every chunk that was transcribed successfully — whether
    the result came from a fresh worker computation, a Tier 1 hit, or
    a Tier 2 hit. The client cannot tell which path produced it, and
    doesn't need to.

    Field selection here is a deliberate SUBSET of TranscriptionResult,
    not an automatic mirror of it:
      - redacted_text, language, is_silent: what a UI needs to render
        a live transcript.
      - redaction_count: not needed to render text, but cheap to
        expose and genuinely useful downstream — a compliance
        dashboard can sum this across a call to show "N items
        redacted" without the backend needing a new field or endpoint
        later.
      - duration_ms: kept because it's a legitimate client-side
        latency signal, AND because on a cache hit it will be exactly
        0.0 by construction (see TranscriptionResult's docstring) —
        genuinely informative, not internal noise.

    If TranscriptionResult ever grows a field that's purely an
    internal/debugging concern (which cache tier served this, a
    worker's PID, a raw pre-redaction confidence score), the right
    move is to NOT add it here. from_result() below is exactly where
    that inclusion/exclusion decision gets made explicitly, once —
    rather than by omission inside a generic serializer that has no
    opinion either way.
    """

    type: Literal["transcript"] = "transcript"
    redacted_text: str
    redaction_count: int
    language: str | None
    duration_ms: float
    is_silent: bool

    @classmethod
    def from_result(cls, result: TranscriptionResult) -> "TranscriptResponse":
        """
        The one place TranscriptionResult -> wire-format translation
        happens. Deliberately explicit, field by field — not
        cls(**dataclasses.asdict(result)). The entire point of this
        class existing is that the internal and external shapes are
        ALLOWED to diverge; an automatic unpack would quietly undo
        that protection the moment the two shapes actually do drift.
        """
        return cls(
            redacted_text=result.redacted_text,
            redaction_count=result.redaction_count,
            language=result.language,
            duration_ms=result.duration_ms,
            is_silent=result.is_silent,
        )


class ErrorResponse(BaseModel):
    """
    Sent for a chunk that failed processing (see PipelineError in
    services/pipeline.py). The connection stays open after this — it's
    a per-chunk error, not a connection-level one.

    `message` is TRUSTED to already be client-safe by the time it
    reaches this model — this class does not sanitize it. That
    responsibility lives in services/pipeline.py, which constructs
    PipelineError with descriptive-but-safe text (e.g. "could not
    decode audio chunk: ...") rather than forwarding a raw internal
    exception string. That's an assumption this file leans on, not one
    it enforces: if pipeline.py's error branches are ever changed to
    include raw internals "for debuggability," this model would
    faithfully — and wrongly — ship that straight to the client.
    """

    type: Literal["error"] = "error"
    message: str  

    pass

# Every message this endpoint can send, tagged for automatic
# discrimination on the `type` field. Used today mainly as a single
# source of truth for "what can this endpoint emit" — useful for
# tests, and for generating client-side types later. It becomes
# directly load-bearing (i.e. actually used to validate/parse,
# not just document) the moment this file gains an incoming-message
# model that needs the same discrimination on a read path.
AudioStreamMessage = Annotated[
    Union[TranscriptResponse, ErrorResponse],
    Field(discriminator="type"),
]
