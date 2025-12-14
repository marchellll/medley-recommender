"""Embedding generation from lyrics."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.db import get_song, create_or_update_song
from src.embeddings.model_loader import get_embedding_model
from src.utils.config import settings
from src.utils.lyrics import clean_lyrics


def generate_embedding(
    lyrics: str,
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """
    Generate embedding vector from lyrics.

    Args:
        lyrics: Song lyrics text
        model: Optional model instance (uses cached model if not provided)

    Returns:
        Embedding vector as numpy array
    """
    if model is None:
        model = get_embedding_model()

    # Clean lyrics
    cleaned_lyrics = clean_lyrics(lyrics)

    # Generate embedding
    # Use deterministic seed for reproducibility
    embedding = model.encode(
        cleaned_lyrics,
        normalize_embeddings=True,  # Normalize for cosine similarity
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embedding


async def generate_and_save_embedding(
    session: AsyncSession,
    song_id: str,
    lyrics: str,
    model: Optional[SentenceTransformer] = None,
    force: bool = False,
) -> Path:
    """
    Generate embedding for lyrics and save to file.

    Args:
        session: Database session
        song_id: Song identifier
        lyrics: Song lyrics
        model: Optional model instance
        force: If True, regenerate even if embedding already exists

    Returns:
        Path to saved embedding file
    """
    embeddings_dir = settings.embeddings_dir
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embedding_file = embeddings_dir / f"{song_id}.json"

    # Ensure parent directory exists (in case song_id contains path separators like "youtube/video_id")
    embedding_file.parent.mkdir(parents=True, exist_ok=True)

    # Skip expensive generation if embedding already exists (unless forced)
    if not force and embedding_file.exists():
        song = await get_song(session, song_id)
        if song and song.embedding_file_path:
            return embedding_file

    # Generate embedding (expensive operation)
    embedding = generate_embedding(lyrics, model)

    # Convert to list for JSON serialization
    embedding_list = embedding.tolist()

    # Save as JSON
    with open(embedding_file, "w") as f:
        json.dump(embedding_list, f)

    # Update database
    song = await get_song(session, song_id)
    if song:
        await create_or_update_song(
            session,
            song.song_id,
            song.title,
            song.youtube_url,
            song.lyrics,
            embedding_file_path=str(embedding_file.resolve()),
        )

    return embedding_file


async def generate_embedding_for_song(
    session: AsyncSession,
    song_id: str,
    model: Optional[SentenceTransformer] = None,
    force: bool = False,
) -> Path:
    """
    Generate embedding for a song from its lyrics.

    Args:
        session: Database session
        song_id: Song identifier
        model: Optional model instance
        force: If True, regenerate even if embedding already exists

    Returns:
        Path to saved embedding file

    Raises:
        ValueError: If song not found or lyrics are missing/empty
    """
    song = await get_song(session, song_id)
    if not song:
        raise ValueError(f"Song not found: {song_id}")

    if not song.lyrics or not song.lyrics.strip():
        raise ValueError(f"Song {song_id} ({song.title or 'unknown'}) is missing lyrics. Please add lyrics manually before generating embeddings.")

    # Clean lyrics before generating embedding
    from src.utils.lyrics import clean_lyrics
    cleaned_lyrics = clean_lyrics(song.lyrics)

    if not cleaned_lyrics.strip():
        raise ValueError(f"Song {song_id} ({song.title or 'unknown'}) has empty lyrics after cleaning. Please add valid lyrics.")

    return await generate_and_save_embedding(session, song_id, cleaned_lyrics, model, force=force)


def load_embedding(embedding_file: Path) -> np.ndarray:
    """
    Load embedding from JSON file.

    Args:
        embedding_file: Path to embedding JSON file

    Returns:
        Embedding vector as numpy array
    """
    with open(embedding_file, "r") as f:
        embedding_list = json.load(f)

    return np.array(embedding_list, dtype=np.float32)


