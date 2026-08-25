
"""
app/workers/pool.py

Owns the single, process-wide ProcessPoolExecutor. Created exactly
once, at application startup (from app/core/lifespan.py — not yet
written), and shared across every WebSocket connection for the life
of the server.

This module holds a module-level singleton rather than attaching the
pool to `app.state` or similar, so services/pipeline.py can reach it
via a plain function call (get_pool()) without threading a request/
websocket object through every call site. The tradeoff is module-level
global state, which is acceptable here because there is exactly one
pool for the life of the process — if that assumption ever changes,
this file is the seam to change first.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from app.core.config import settings
from app.workers.init_worker import init_worker

logger = logging.getLogger("worker.pool")

#lazy initializing the worker pool.
_pool: Optional[ProcessPoolExecutor] = None

def create_pool() -> ProcessPoolExecutor:
    """
    Builds the global ProcessPoolExecutor. Must be called exactly once,
    during app startup — never per-request, never per-connection.

    Raises if a pool already exists rather than silently returning it
    or silently creating a second pool alongside it. A double-create
    here is always a lifecycle bug (e.g. a startup hook firing twice)
    and should fail loudly, not be papered over with idempotency.
    """
    global _pool #giving global context to interpreter about _pool wrt this fxn.
    if _pool is not None:
        raise RuntimeError(
            "create_pool() called but a pool already exists — "
            "the executor must be created exactly once, at startup."
        )

    worker_count = _resolve_worker_count() #reserves one for the event loop. #7

    # Explicit "spawn" context rather than the platform default (which
    # is "fork" on Linux). Two reasons this matters here specifically:
    #   1. If settings.DEVICE is ever "cuda", CUDA contexts are not
    #      fork-safe — spawn is required, not optional, for GPU workers.
    #   2. By the time this runs, the parent process has already started
    #      uvicorn's event loop and its own threads. Forking a
    #      multi-threaded process is a known source of subtle deadlocks
    #      (a forked child can inherit a lock held by a thread that no
    #      longer exists in the child). Spawn avoids this by starting
    #      each worker as a fresh interpreter. clones memory.
    # Cost: slightly slower worker startup (each one re-imports its own
    # dependencies) — irrelevant here since workers are long-lived.
    mp_context = multiprocessing.get_context("spawn") #launches a fresh interpreter across the worker pool during runtime. [its different from the set_start_method("spawn") which does it globally for the entire python runtime]

    logger.info(
        "creating process pool: workers=%d model=%s device=%s compute_type=%s",
        worker_count, settings.MODEL_NAME, settings.DEVICE, settings.COMPUTE_TYPE,
    )

    _pool = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=init_worker,
        initargs=(settings.MODEL_NAME, settings.DEVICE, settings.COMPUTE_TYPE),
    )

    # ProcessPoolExecutor spawns workers LAZILY by default — a worker
    # process (and its model load) doesn't actually start until the
    # first task is submitted. For an ML pipeline that's a trap: the
    # first real audio chunk would pay the full model-load latency on
    # top of its own processing time, and any init_worker() failure
    # would surface as a confusing first-request error instead of a
    # startup failure. So we force every worker to spawn and finish
    # initializing right now, before the app accepts connections.
    _warm_up_workers(_pool, worker_count)

    logger.info("process pool ready: %d workers initialized", worker_count)
    return _pool

def get_pool() -> ProcessPoolExecutor:
    """
    Accessor used by services/pipeline.py on every chunk. Raises rather
    than lazily creating a pool on first access — a call reaching here
    before startup has finished is a lifecycle bug worth surfacing
    immediately, not a case to silently work around.
    """
    if _pool is None:
        raise RuntimeError(
            "get_pool() called before create_pool() — "
            "has app startup finished running?"
        )
    return _pool

#############################

def shutdown_pool() -> None:
    """
    Called once, from lifespan teardown, on server shutdown.

    wait=True: block until in-flight tasks finish rather than killing
    workers mid-transcription — a half-processed audio chunk is worse
    than a slightly slower shutdown.
    cancel_futures=True: anything still QUEUED (not yet started) is
    cancelled outright — no reason to keep draining a backlog while
    the server is already on its way down.
    """
    global _pool
    if _pool is None:
        logger.warning("shutdown_pool() called but no pool exists — no-op")
        return

    logger.info("shutting down process pool")
    _pool.shutdown(wait=True, cancel_futures=True)
    _pool = None


def _resolve_worker_count() -> int:
    """
    Defaults to CPU count minus one, reserving a core for the event
    loop / OS, with a floor of 1 so this doesn't break on a single-core
    dev container. settings.POOL_WORKERS, if set, always overrides the
    computed default — useful for capping worker count in memory-
    limited deployments, since each worker holds its own full copy of
    the model in RAM.
    """
    if settings.POOL_WORKERS is not None:
        return max(1, settings.POOL_WORKERS)

    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count - 1) #=7

#this function eagerly initializes all worker processes instead of waiting for user requests on startup in real time.
def _warm_up_workers(pool: ProcessPoolExecutor, worker_count: int) -> None:
    """
    Forces `worker_count` worker processes to SPAWN and run their
    initializer immediately, instead of lazily on first task. Submits
    one trivial task per worker and blocks until all complete.

    This isn't a perfectly guaranteed one-task-per-process mapping —
    the executor's internal scheduler owns that decision — but in
    practice, submitting >= worker_count no-op tasks with no other
    work queued reliably spins every worker up, since there's nothing
    else for the pool to schedule instead. If any worker's initializer
    raises (e.g. the model file is missing), that exception propagates
    out of .result() here, at startup — not on some future request.
    """
    #submit dummy tasks (noop is no-op (dummy) tasks)
    futures = [pool.submit(_noop) for _ in range(worker_count)]
    for future in futures: #this blocks the main server thread
        future.result()  # re-raises any init_worker() exception immediately
        
    #why is this important? check notes.

def _noop() -> None:
    """Trivial task used only to force worker process startup."""
    return None