"""API routes."""

import json
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import AddSongRequest, AddSongResponse, SearchRequest, SearchResponse, SongResult
from src.database.db import (
    AsyncSessionLocal,
    create_or_update_song,
    get_all_songs,
    get_song,
)
from src.database.models import Song
from src.embeddings.encoder import generate_embedding
from src.index.searcher import IndexSearcher

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

router = APIRouter()

# Global searcher instance (loaded at startup)
searcher: Optional[IndexSearcher] = None


async def get_session() -> AsyncSession:
    """Dependency for database session."""
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/search", response_model=SearchResponse)
async def search_songs(
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    """
    Search for songs by semantic similarity with native filtering by key and BPM.

    Chroma handles filtering natively during search, making it efficient.

    Args:
        request: Search request with query and filters
        session: Database session

    Returns:
        Search results with similarity scores
    """
    # #region agent log
    _log_debug("api/routes.py:33", "search_songs endpoint called", {
        "query": request.query,
        "limit": request.limit,
        "keys": request.keys,
        "bpm_min": request.bpm_min,
        "bpm_max": request.bpm_max,
    })
    # #endregion

    if searcher is None:
        # #region agent log
        _log_debug("api/routes.py:49", "searcher is None, raising 503", {})
        # #endregion
        raise HTTPException(status_code=503, detail="Index not loaded")

    # Generate query embedding
    # #region agent log
    _log_debug("api/routes.py:53", "generating query embedding", {"query": request.query})
    # #endregion
    query_embedding = generate_embedding(request.query)
    # #region agent log
    _log_debug("api/routes.py:53", "query embedding generated", {"embedding_shape": query_embedding.shape if hasattr(query_embedding, 'shape') else "unknown"})
    # #endregion

    # Search with native Chroma filtering
    # #region agent log
    _log_debug("api/routes.py:56", "calling searcher.search", {
        "k": request.limit,
        "keys": request.keys,
        "bpm_min": request.bpm_min,
        "bpm_max": request.bpm_max,
    })
    # #endregion
    search_results = searcher.search(
        query_embedding,
        k=request.limit,
        keys=request.keys,
        bpm_min=request.bpm_min,
        bpm_max=request.bpm_max,
    )
    # #region agent log
    _log_debug("api/routes.py:62", "searcher.search completed", {"result_count": len(search_results)})
    # #endregion

    if not search_results:
        # #region agent log
        _log_debug("api/routes.py:64", "no search results, returning empty", {})
        # #endregion
        return SearchResponse(results=[], total=0)

    # Fetch song metadata from database for the filtered results
    song_ids = [song_id for song_id, _ in search_results]
    # #region agent log
    _log_debug("api/routes.py:68", "fetching song metadata from database", {"song_ids": song_ids})
    # #endregion
    stmt = select(Song).where(Song.song_id.in_(song_ids))
    result = await session.execute(stmt)
    songs_dict = {song.song_id: song for song in result.scalars().all()}
    # #region agent log
    _log_debug("api/routes.py:71", "song metadata fetched", {"songs_found": len(songs_dict)})
    # #endregion

    # Build results (filters already applied, so just convert to response format)
    results = []
    for song_id, distance in search_results:
        if song_id not in songs_dict:
            continue

        song = songs_dict[song_id]

        # Convert distance to similarity score (1 - distance for cosine similarity)
        similarity_score = 1.0 - distance

        results.append(
            SongResult(
                song_id=song.song_id,
                title=song.title,
                bpm=song.bpm,
                key=song.key,
                similarity_score=similarity_score,
                youtube_url=song.youtube_url,
            )
        )

    # #region agent log
    _log_debug("api/routes.py:95", "search_songs completed", {"total_results": len(results)})
    # #endregion
    return SearchResponse(results=results, total=len(results))


@router.post("/add_song", response_model=AddSongResponse)
async def add_song(
    request: AddSongRequest,
    session: AsyncSession = Depends(get_session),
) -> AddSongResponse:
    """
    Add a new song to the system.

    Args:
        request: Song data
        session: Database session

    Returns:
        Success response with song ID
    """
    # #region agent log
    _log_debug("api/routes.py:98", "add_song endpoint called", {
        "title": request.title,
        "youtube_url": str(request.youtube_url),
        "has_lyrics": bool(request.lyrics),
    })
    # #endregion

    import hashlib
    import re
    from datetime import datetime
    from pathlib import Path

    # Extract video ID from YouTube URL
    youtube_url_str = str(request.youtube_url)
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    video_id = None
    for pattern in patterns:
        match = re.search(pattern, youtube_url_str)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        # #region agent log
        _log_debug("api/routes.py:131", "invalid YouTube URL", {"youtube_url": youtube_url_str})
        # #endregion
        return AddSongResponse(
            success=False,
            message="Invalid YouTube URL",
        )

    # Generate song_id from video ID
    song_id = f"youtube/{video_id}"
    # #region agent log
    _log_debug("api/routes.py:138", "video ID extracted", {"video_id": video_id, "song_id": song_id})
    # #endregion

    try:
        # Save to new_songs directory
        new_songs_dir = Path("data/new_songs")
        new_songs_dir.mkdir(parents=True, exist_ok=True)

        song_data = {
            "song_id": song_id,
            "title": request.title,
            "youtube_url": youtube_url_str,
            "lyrics": request.lyrics,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        song_file = new_songs_dir / f"{timestamp}_{song_id}.json"

        import json

        with open(song_file, "w") as f:
            json.dump(song_data, f, indent=2)

        # Also add to database (without metadata yet)
        # #region agent log
        _log_debug("api/routes.py:161", "adding song to database", {"song_id": song_id})
        # #endregion
        await create_or_update_song(
            session,
            song_id,
            request.title,
            youtube_url_str,
            request.lyrics,
        )
        # #region agent log
        _log_debug("api/routes.py:167", "song added to database successfully", {"song_id": song_id})
        # #endregion

        # #region agent log
        _log_debug("api/routes.py:169", "add_song completed successfully", {"song_id": song_id})
        # #endregion
        return AddSongResponse(
            success=True,
            song_id=song_id,
            message=f"Song added successfully. Run pipeline to process: {song_id}",
        )
    except Exception as e:
        # #region agent log
        _log_debug("api/routes.py:174", "add_song failed with exception", {"error": str(e)})
        # #endregion
        return AddSongResponse(
            success=False,
            message=f"Failed to add song: {str(e)}",
        )


def set_searcher(new_searcher: IndexSearcher) -> None:
    """Set the global searcher instance."""
    global searcher
    searcher = new_searcher

