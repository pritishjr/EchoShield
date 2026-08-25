
#TIER 2 OF CACHING

"""
app/cache/redis_cache.py

Tier 2 of the two-tier cache: a shared, external cache so a hit in
one server process/replica is visible to every other one. Where
LocalCache (Tier 1) only helps a single process avoid re-transcribing
audio it personally has seen, this tier saves work across the whole
fleet — e.g. the same hold-music clip arriving on two connections
handled by two different uvicorn worker processes.

Scope and boundaries, same spirit as local_cache.py:
  - Knows nothing about audio, models, or transcription. Stores and
    retrieves plain JSON-serializable dicts against a string key. Key
    composition (audio hash + a redaction/model version tag) happens
    upstream — see the version-tag note in local_cache.py, which
    applies here too and matters MORE here, since Redis entries can
    outlive a single deploy.
  - A cache is an optimization, never a hard dependency. If Redis is
    slow, unreachable, or down, every call here degrades to a miss
    (get) or a no-op (set) rather than raising — a Redis outage must
    not be able to take the transcription pipeline down with it.
    Failures are logged, never silently ignored.
  - This module does NOT implement its own eviction logic. Redis
    entries carry a TTL (see set()), and — operationally, on the
    Redis server itself — a maxmemory-policy such as allkeys-lru
    handles eviction under memory pressure. Duplicating that in
    application code would just be a worse version of what the
    server already does.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

#redis imports
import redis.asyncio as redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger("cache.redis")

#lazy initializating the redis client:
_client: Optional["redis.Redis"] = None

#initialization is synchronous:
def create_redis_client() -> "redis.Redis":
    """
    builds the one-and-only single, wide across all processes- Redis Client that boots up alongside the worker-processes pool.
    The main objective of this is to program the guardrails in case of a server shotage using get() and set() functions.
    """
    global _client
    #make sure only one redis client is active
    if _client is not None:
        raise RuntimeError(
            "create_redis_client() called but a client already exists — "
            "it must be created exactly once, at startup."
        )

    redis_url = str(settings.REDIS_URL)
    logger.info("creating redis client: %s", redis_url)
    _client = redis.from_url(
        redis_url, #database port
        decode_responses=True,       # get back str, not bytes, from redis-py
        socket_connect_timeout=2.0,  # fail fast per-call rather than hang the event loop
        socket_timeout=2.0,
    )
    return _client

#to check whether client is up and active (synchronous)
def get_redis_client() -> "redis.Redis":
    """Accessor used by get()/set() below. Raises if called before startup."""
    if _client is None:
        raise RuntimeError(
            "get_redis_client() called before create_redis_client() — "
            "has app startup finished running?"
        )
    return _client

#BEHAVOIURS ARE ALWAYS ASYNCHRONOUS

#to close all the active connections with the pool due to server shutdown.
async def close_redis_client() -> None:
    """Called once, from lifespan teardown, on server shutdown."""
    global _client
    if _client is None:
        logger.warning("close_redis_client() called but no client exists — no-op")
        return

    #why async?
    await _client.aclose()
    _client = None

#to fetch the value for a key:
#it is asynchronous since the signal is a network i/0 (outbound)
async def get(key: str) -> Optional[dict[str, Any]]:
    """
    Returns the cached value or None or a "miss"
    """
    #initialize client
    client = get_redis_client()
    try:
        raw = await client.get(key) #fetches data
    except RedisError:
        logger.warning(
            "redis GET failed for key=%s — treating as cache miss", key, exc_info=True
        )
        return None

    if raw is None:
        return None

    #treating cache miss
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # A corrupt/unexpected value under this key should not crash the
        # request — treat it as a miss and let a fresh transcription
        # overwrite it on the next set().
        logger.warning("redis value for key=%s was not valid JSON — treating as cache miss", key)
        return None

#to write a value to a cache key.
async def set(key: str, value: dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
    """
    stores the value to the address/key.
    checks whether the ttl seconds is mentioned else the default is the pre-deteremined config file global constant.
    """
    client = get_redis_client()
    ttl = ttl_seconds if ttl_seconds is not None else settings.REDIS_TTL_SECONDS

    try:
        await client.set(key, json.dumps(value), ex=ttl)
    except RedisError:
        logger.warning("redis SET failed for key=%s — entry not cached", key, exc_info=True)