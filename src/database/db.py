"""Database connection and utilities."""

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.database.models import Base, Song
from src.utils.config import settings
from src.utils.lyrics import clean_lyrics


# Create async engine
engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.database_path_resolved}",
    echo=False,
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_song(session: AsyncSession, song_id: str) -> Optional[Song]:
    """Get a song by ID."""
    from sqlalchemy import select

    result = await session.execute(select(Song).where(Song.song_id == song_id))
    return result.scalar_one_or_none()


async def create_or_update_song(
    session: AsyncSession,
    song_id: str,
    title: str,
    youtube_url: str,
    lyrics: str,
    **kwargs,
) -> Song:
    """Create or update a song.

    Lyrics are automatically cleaned (removes hidden characters, normalizes typographic characters)
    before being saved to the database.

    IMPORTANT: This function is idempotent - existing lyrics are preserved if new lyrics are empty.
    This prevents the pipeline from overwriting manually inputted lyrics.
    """
    # Clean lyrics before saving to ensure consistent, clean data in database
    cleaned_lyrics = clean_lyrics(lyrics)

    song = await get_song(session, song_id)
    if song:
        # Update existing song
        song.title = title
        song.youtube_url = youtube_url
        # Preserve existing lyrics if new lyrics are empty (idempotent behavior)
        # This prevents the pipeline from overwriting manually inputted lyrics
        if cleaned_lyrics.strip():
            song.lyrics = cleaned_lyrics
        # If cleaned_lyrics is empty, keep existing lyrics unchanged
        for key, value in kwargs.items():
            if hasattr(song, key):
                setattr(song, key, value)
    else:
        # Create new song
        song = Song(
            song_id=song_id,
            title=title,
            youtube_url=youtube_url,
            lyrics=cleaned_lyrics,
            **kwargs,
        )
        session.add(song)

    await session.commit()
    await session.refresh(song)
    return song


async def get_all_songs(session: AsyncSession) -> list[Song]:
    """Get all songs."""
    from sqlalchemy import select

    result = await session.execute(select(Song))
    return list(result.scalars().all())


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


