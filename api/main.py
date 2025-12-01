"""FastAPI application main file."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router, set_searcher
from src.database.db import close_db, init_db
from src.index.searcher import IndexSearcher
from src.utils.config import settings


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
    return {
        "message": "Medley Recommender API",
        "version": "0.1.0",
        "host": settings.api_host,
        "port": settings.api_port,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


