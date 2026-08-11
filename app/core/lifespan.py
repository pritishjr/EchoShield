"""
app/core/lifespan.py

FastAPI lifespan context manager. Owns application-wide startup and
shutdown for the two long-lived, expensive-to-create resources this
service depends on: the Redis client (Tier 2 cache) and the
ProcessPoolExecutor (transcription workers).

Wired into the app via:

    app = FastAPI(lifespan=lifespan)

Everything before `yield` runs once, before the app accepts its first
connection — uvicorn holds off signaling "startup complete" until this
generator reaches that point, which is exactly the behavior we want:
a readiness probe shouldn't pass while workers are still loading
models. Everything after `yield` runs once, after the app stops
accepting new connections.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.cache.redis_cache import close_redis_client, create_redis_client
from app.workers.pool import create_pool, shutdown_pool

logger = logging.getLogger("core.lifespan")

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """
    Startup order: Redis client first, then the worker pool.

    Redis client creation is deliberately cheap here — it doesn't ping
    the server, it just builds a client object (see
    redis_cache.create_redis_client's own docstring), so there's
    barely any startup cost either way. The worker pool is the
    expensive step: it spawns every worker process and blocks until
    each one has finished loading its model. Doing the cheap step
    first means that if the Redis URL itself is genuinely malformed
    (a real misconfiguration, not a transient network issue — those
    are handled per-call, not at startup), we fail before paying the
    cost of spinning up every worker process only to tear them down a
    moment later.

    If pool creation fails after the Redis client already exists, we
    explicitly close that client before re-raising. An app that fails
    to start must not leak the resources it DID manage to acquire —
    without this, every failed startup attempt would leave an
    orphaned Redis connection behind.

    Note on blocking: create_pool() is a synchronous, blocking call —
    it is not offloaded to a thread. That's intentional, not an
    oversight: during this phase there is no other concurrent work for
    the event loop to be doing (no connections are being accepted
    yet), and the app genuinely cannot do anything useful until the
    pool exists anyway. Backgrounding it would add complexity for zero
    concurrency benefit.
    """
    logger.info("application startup: initializing shared resources")

    create_redis_client()
    logger.info("redis client initialized")

    try:
        create_pool()
    except Exception:
        logger.exception("worker pool failed to initialize — rolling back redis client")
        await close_redis_client()
        raise

    logger.info("worker pool initialized — application ready to accept connections")

    try:
        yield
    finally:
        # Runs on shutdown regardless of how the app's lifetime ended
        # (a clean shutdown signal, or an exception raised somewhere
        # during the yield). Both resources are torn down every time,
        # in the REVERSE of creation order (pool, then redis) — LIFO —
        # so a partial teardown can never leave one resource released
        # while the other is still silently holding a connection open.
        logger.info("application shutdown: releasing shared resources")
        shutdown_pool()
        await close_redis_client()
        logger.info("shutdown complete")