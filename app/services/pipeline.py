#THE BRAIN OF HOW EVERYTHING CONNECTS:
#check Tier 1 → check Tier 2 → submit to pool → write back to both tiers

"""
app/services/pipeline.py

The orchestrator. This is the ONLY module in the codebase allowed to
know about every layer at once: hashing, both cache tiers, and the
worker pool. Every other module deliberately knows about at most one
of these. That constraint is what keeps this file readable top to
bottom as "the whole story" of what happens to a single audio chunk.

Flow for one chunk:
    1. Hash the audio bytes (cache/hashing.py).
    2. Compose the ACTUAL cache key: hash + model name + a manually
       bumped ruleset version (CACHE_KEY_VERSION below) — never the
       raw hash alone. This is what makes a redaction-pattern change
       or a model swap auto-invalidate old entries, instead of
       silently continuing to serve output computed under old rules.
    3. Check Tier 1 (local, in-process, synchronous).
    4. On a Tier 1 miss, check Tier 2 (Redis, async).
       - On a Tier 2 hit: promote the value into Tier 1 so the next
         lookup on THIS process doesn't need Redis again.
    5. On a miss in both tiers: submit to the global ProcessPoolExecutor,
       await the result.
    6. Write the fresh result into Tier 1 (sync) and Tier 2
       (fire-and-forget, see _fire_and_forget) before returning it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.cache import redis_cache
from app.cache.hashing import audio_hash
from app.cache.local_cache import LocalCache
from app.core.config import settings
from app.workers import pool
from app.workers.transcribe import AudioDecodeError, TranscriptionResult, transcribe_and_redact

logger = logging.getLogger("services.pipeline")


CACHE_KEY_VERSION = "v1" #unchanged unredacted audio chunk.
#v2 becomes the redacted audio chunk final transcription which is picked by the processpoolexecutor.
#structure of a cache key: {audio_hash}:{model_name}:{v1}
#no need to flush the redis cache deploy manually when there is a independent change in the redaction functions or the transcription model itself.

#facade exception which collects all the low-level ERRORS which must be handled this banner. (else we would have to handle all of them independently.)
class PipelineError(Exception):
    """
    Raised when a chunk cannot be processed — either the audio itself
    was unusable, or the worker pool failed after both caches missed.
    Wraps the underlying cause so the WebSocket layer has exactly one
    exception type to catch, without needing to know whether the
    failure was a bad decode, an empty chunk, or something unexpected
    inside a worker.
    """
    pass
#-----

#instantiating the local cache object which does not dependent on any external modules outside the eventloop
_local_cache = LocalCache(
    max_size=settings.LOCAL_CACHE_MAX_SIZE,
    ttl_seconds=settings.LOCAL_CACHE_TTL_SECONDS,
)

#stores all the bg tasks garbage (asynchronous asyncio.Task references) in a set.
_background_tasks: set[asyncio.Task] = set()

#Asynchronously process the input audio chunk
async def process_chunk(audio_bytes: bytes) -> TranscriptionResult:
    """
    The single entry point the WebSocket layer calls per chunk.
    Returns a TranscriptionResult whether it came from Tier 1, Tier 2,
    or a fresh worker computation — callers don't need to care which.
    """
    try:
        hashed_audio = audio_hash(audio_bytes)
    except ValueError as exc:
        # Empty/degenerate chunk. Fail this one chunk only.
        # WebSocket connection is still healthy.
        raise PipelineError(f"cannot process chunk: {exc}") from exc

    key = _cache_key(hashed_audio)

    tier1_hit = _local_cache.get(key)
    if tier1_hit is not None:
        logger.debug("cache hit (tier1/local): %s", key)
        return _from_cache_entry(tier1_hit)

    tier2_hit = await redis_cache.get(key)
    if tier2_hit is not None:
        logger.debug("cache hit (tier2/redis): %s", key)
        _local_cache.set(key, tier2_hit)  # promote into tier1 for next time
        return _from_cache_entry(tier2_hit)

    logger.debug("cache miss (both tiers) — submitting to worker pool: %s", key)
    result = await _run_in_pool(audio_bytes)

    entry = _to_cache_entry(result)
    _local_cache.set(key, entry)
    _fire_and_forget(redis_cache.set(key, entry))

    return result #returns a TranscriptionResult object


def _cache_key(hashed_audio: str) -> str:
    """
    Composes the real cache key from the raw audio hash plus a version
    component. Both cache tiers only ever see THIS string — never the
    raw hash alone — precisely so a model or redaction-rule change
    invalidates old entries automatically rather than needing a manual
    cache flush on deploy.
    """
    return f"{hashed_audio}:{settings.MODEL_NAME}:{CACHE_KEY_VERSION}"


async def _run_in_pool(audio_bytes: bytes) -> TranscriptionResult:
    """
    Submits one chunk to the global worker pool and awaits the result.
    Translates worker-side exceptions into PipelineError so callers of
    this module only ever need to handle one exception type.
    """
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            pool.get_pool(), transcribe_and_redact, audio_bytes
        )
    except AudioDecodeError as exc:
        logger.warning("audio decode failed for chunk: %s", exc)
        raise PipelineError(f"could not decode audio chunk: {exc}") from exc
    except Exception as exc:
        logger.exception("unexpected worker failure")
        raise PipelineError("transcription failed unexpectedly") from exc


def _to_cache_entry(result: TranscriptionResult) -> dict[str, Any]:
    """
    Converts a TranscriptionResult into the plain dict shape the cache
    layer stores. duration_ms is deliberately dropped here — it
    describes how long THIS worker invocation took, which is
    meaningless (and misleading) to replay on a future cache hit.
    """
    return {
        "redacted_text": result.redacted_text,
        "redaction_count": result.redaction_count,
        "language": result.language,
        "is_silent": result.is_silent,
    }


def _from_cache_entry(entry: dict[str, Any]) -> TranscriptionResult:
    """
    Reconstructs a TranscriptionResult from a cached dict. duration_ms
    is set to 0.0 — an explicit, self-documenting signal that no
    actual model inference happened for this response. That also
    makes cache-driven wins visible during a Phase 5-style profiling
    pass: a near-zero duration_ms on a busy pipeline is the cache
    working as intended, not a suspiciously fast model.
    """
    return TranscriptionResult(
        redacted_text=entry["redacted_text"],
        redaction_count=entry["redaction_count"],
        language=entry["language"],
        duration_ms=0.0,
        is_silent=entry["is_silent"],
    )


def _fire_and_forget(coro) -> None:
    """
    Schedules a coroutine (the Tier 2 write) without awaiting it, so a
    chunk's response never waits on a Redis round trip. The task
    reference is held in _background_tasks until it completes, then
    discarded — see the module-level comment on why that matters.
    Any exception is logged here rather than left to surface as an
    unhandled "Task exception was never retrieved" warning;
    redis_cache.set() already catches RedisError itself, so reaching
    this handler at all would indicate something unexpected.
    """
    task = asyncio.create_task(_log_background_failure(coro))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _log_background_failure(coro) -> None:
    try:
        await coro
    except Exception:
        logger.exception("background cache write failed unexpectedly")