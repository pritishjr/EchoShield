# call-audit-pipeline

Scaffold for a FastAPI websocket pipeline that separates websocket handling, worker execution, caching, redaction, and test layers.

## Layout

- `app/main.py` creates the FastAPI app and wires startup/shutdown lifecycle hooks.
- `app/api/websocket.py` keeps the websocket route thin and receive-only.
- `app/core/` holds config, logging, and lifespan setup.
- `app/workers/` contains process-pool and transcription worker stubs.
- `app/cache/` contains hashing and cache layers.
- `app/services/pipeline.py` is the orchestration boundary.

## Commands

- `make run`
- `make test`
- `make profile`
