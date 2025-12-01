"""API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import AddSongRequest, AddSongResponse, SearchRequest, SearchResponse, SongResult
from src.database.db import (
    AsyncSessionLocal,
    create_or_update_song,
    get_all_songs,
    get_candidate_song_ids,
    get_song,
)
from src.database.models import Song
from src.embeddings.encoder import generate_embedding
from src.index.searcher import IndexSearcher

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
    Search for songs by semantic similarity with optimized filtering by key and BPM.

    This implementation pre-filters candidates in the database before ANN search,
    making it much more efficient when filters are applied.

    Args:
        request: Search request with query and filters
        session: Database session

    Returns:
        Search results with similarity scores
    """
    if searcher is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    # Generate query embedding
    query_embedding = generate_embedding(request.query)

    # Pre-filter candidates by key/BPM if filters are provided
    # This optimizes the search by only searching within matching songs
    candidate_song_ids = None
    has_filters = (
        (request.keys is not None and len(request.keys) > 0)
        or request.bpm_min is not None
        or request.bpm_max is not None
    )

    if has_filters:
        candidate_song_ids = await get_candidate_song_ids(
            session,
            keys=request.keys,
            bpm_min=request.bpm_min,
            bpm_max=request.bpm_max,
        )

        # If no candidates match the filters, return empty results
        if not candidate_song_ids:
            return SearchResponse(results=[], total=0)

    # Search in index with candidate filtering (if applicable)
    # Use adaptive k to ensure we get enough results after filtering
    max_k = 500  # Reasonable upper bound for adaptive search
    if candidate_song_ids:
        # Use optimized search with candidate filtering
        search_results = searcher.search_with_candidates(
            query_embedding,
            candidate_song_ids=candidate_song_ids,
            k=request.limit,
            max_k=max_k,
        )
    else:
        # No filters, use regular search
        search_results = searcher.search(query_embedding, k=request.limit)

    if not search_results:
        return SearchResponse(results=[], total=0)

    # Fetch song metadata from database for the filtered results
    song_ids = [song_id for song_id, _ in search_results]
    stmt = select(Song).where(Song.song_id.in_(song_ids))
    result = await session.execute(stmt)
    songs_dict = {song.song_id: song for song in result.scalars().all()}

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
                artist=song.artist,
                bpm=song.bpm,
                key=song.key,
                similarity_score=similarity_score,
                youtube_url=song.youtube_url,
            )
        )

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
    import hashlib
    from datetime import datetime
    from pathlib import Path

    # Generate song_id from title and artist
    song_id_base = f"{request.title}_{request.artist}".lower().replace(" ", "_")
    song_id = hashlib.md5(song_id_base.encode()).hexdigest()[:16]

    try:
        # Save to new_songs directory
        new_songs_dir = Path("data/new_songs")
        new_songs_dir.mkdir(parents=True, exist_ok=True)

        song_data = {
            "song_id": song_id,
            "title": request.title,
            "artist": request.artist,
            "youtube_url": str(request.youtube_url),
            "lyrics": request.lyrics,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        song_file = new_songs_dir / f"{timestamp}_{song_id}.json"

        import json

        with open(song_file, "w") as f:
            json.dump(song_data, f, indent=2)

        # Also add to database (without metadata yet)
        await create_or_update_song(
            session,
            song_id,
            request.title,
            request.artist,
            str(request.youtube_url),
            request.lyrics,
        )

        return AddSongResponse(
            success=True,
            song_id=song_id,
            message=f"Song added successfully. Run pipeline to process: {song_id}",
        )
    except Exception as e:
        return AddSongResponse(
            success=False,
            message=f"Failed to add song: {str(e)}",
        )


def set_searcher(new_searcher: IndexSearcher) -> None:
    """Set the global searcher instance."""
    global searcher
    searcher = new_searcher

