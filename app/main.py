"""
app/main.py

Application entry point. The only file that imports EVERY other
top-level piece (config, logging, lifespan, routers) purely to wire
them together — it contains no business logic of its own, matching
the same "thin" discipline applied to api/websocket.py.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from app.core.config import settings  
from app.core.logging import setup_logging 

setup_logging(level=settings.LOG_LEVEL)

from fastapi import FastAPI  # noqa: E402

#importing the websoct router
from app.api.websocket import router as websocket_router  # noqa: E402
from app.core.lifespan import lifespan  # noqa: E402

#good protocol to set it up inside a function.
#helps during running tests
def create_app() -> FastAPI:
    """
    Builds a fresh FastAPI instance. 
    """
    app = FastAPI(
        title="Call Audit Pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(websocket_router)
    return app

# What `uvicorn app.main:app` actually imports and serves.
app = create_app()