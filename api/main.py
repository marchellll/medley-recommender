"""FastAPI application main file."""

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router, set_searcher
from src.database.db import close_db, init_db
from src.index.searcher import IndexSearcher
from src.utils.config import settings

# #region agent log
import sys

def _log_debug(location: str, message: str, data: dict = None, session_id: str = "debug-session", run_id: str = "run1", hypothesis_id: str = None):
    """Write NDJSON log entry to stdout for docker logs."""
    try:
        log_entry = {
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "sessionId": session_id,
            "runId": run_id,
        }
        if hypothesis_id:
            log_entry["hypothesisId"] = hypothesis_id
        print(json.dumps(log_entry), file=sys.stdout, flush=True)
    except Exception:
        pass  # Silently fail if logging fails
# #endregion


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    await init_db()

    # Load index searcher
    try:
        searcher = IndexSearcher()
        searcher.load()
        set_searcher(searcher)
    except Exception as e:
        print(f"Warning: Could not load index: {e}")
        print("Index will be loaded on first search request")

    yield

    # Shutdown
    await close_db()


app = FastAPI(
    title="Medley Recommender API",
    description=f"API for worship/praise song recommendation (running on {settings.api_host}:{settings.api_port})",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api", tags=["songs"])


@app.get("/")
async def root():
    """Root endpoint."""
    # #region agent log
    _log_debug("api/main.py:44", "root endpoint called", {})
    # #endregion
    return {
        "message": "Medley Recommender API",
        "version": "0.1.0",
        "host": settings.api_host,
        "port": settings.api_port,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    # #region agent log
    _log_debug("api/main.py:55", "health endpoint called", {})
    # #endregion
    return {"status": "healthy"}


