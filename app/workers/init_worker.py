"""
app/workers/init_worker.py

Runs exactly ONCE per worker process — at the moment the shared,
global ProcessPoolExecutor spins that process up. It is never called
per-task.

This is the only place in the codebase permitted to load the speech
model. If model-loading code ever shows up inside transcribe.py's
per-chunk function, that's a regression: it means every single audio
chunk is reloading the model from disk, which defeats the entire
point of a persistent worker pool.

Design notes
------------
1. Each worker process gets its OWN private copy of every Python
   module-level global — that's a property of multiprocessing, not
   something we have to build. So setting a global here is safe: it
   cannot leak into the parent event-loop process or into sibling
   workers.

2. We cap BLAS/OpenMP thread count BEFORE numpy/torch get imported
   anywhere in this process, because those libraries size their
   internal thread pools at import time. With N worker processes each
   silently spawning their own multi-threaded math backend, you get
   CPU oversubscription — more threads fighting over cores than you
   have cores — which can make the pool slower than running on a
   single process. This is why the os.environ calls sit at the very
   top of the file, before any other import.

3. Failure here must be loud and fatal. A worker with no model cannot
   do its job. We'd rather crash the process at pool startup (visible
   immediately, pool creation fails fast) than have it silently error
   out on every task it's handed later.
"""

import os

# Must execute before faster_whisper (and the numpy/torch it pulls in)
# is imported anywhere in this process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
#separate logger for initializer.
logger = logging.getLogger("worker.init")


def init_worker(model_name: str, device: str, compute_type: str) -> None:
    """
    Initializer callback handed to ProcessPoolExecutor at pool-creation
    time, e.g.:

        ProcessPoolExecutor(
            max_workers=settings.POOL_WORKERS,
            initializer=init_worker,
            initargs=(settings.MODEL_NAME, settings.DEVICE, settings.COMPUTE_TYPE),
        )

    The executor calls this once for every worker process it spawns,
    before that process accepts its first task. Parameters are passed
    explicitly via initargs rather than imported from config directly,
    so this function has no hidden dependency on how settings are
    loaded — it just needs three strings.
    """
    pid = os.getpid()
    logger.info(
        "worker %s: loading model '%s' (device=%s, compute_type=%s)",
        pid, model_name, device, compute_type,
    )

    try:
        # Imported inside the function, not at module top-level, so
        # the parent process — which coordinates the pool but never
        # transcribes anything itself — never pays this import cost.
        from faster_whisper import WhisperModel
        from app.workers import transcribe

        model = WhisperModel(model_name, device=device, compute_type=compute_type)

        # Hand the loaded model to transcribe.py's module-global slot.
        # Because this worker process owns a private copy of that
        # module, this cannot race with any other worker process.
        transcribe.set_model(model)

    except Exception:
        logger.exception("worker %s: model load failed — exiting", pid)
        raise  # let the process die; a silent half-initialized worker is worse

    logger.info("worker %s: model ready, standing by for tasks", pid)