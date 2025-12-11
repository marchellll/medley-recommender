"""Audio metadata extraction using librosa."""

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.db import get_song, create_or_update_song


async def extract_metadata(
    session: AsyncSession,
    song_id: str,
    audio_path: Path,
) -> dict:
    """
    Extract metadata from audio file.

    Args:
        session: Database session
        song_id: Song identifier
        audio_path: Path to audio file

    Returns:
        Dictionary with extracted metadata
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio file directly (should be MP3 from downloader post-processor)
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {str(e)}") from e

    # Extract duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Extract BPM (tempo)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if len(tempo) > 0 else None

    # Extract key (chroma-based estimation)
    # This is a simplified key detection - for production, consider using keyfinder
    key = None
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        # Map chroma to key (simplified - 12 keys)
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_idx = np.argmax(chroma_mean)
        key = key_names[key_idx]
    except Exception:
        # If key detection fails, leave as None
        pass

    metadata = {
        "bpm": bpm,
        "key": key,
        "duration": float(duration),
    }

    # Update database
    song = await get_song(session, song_id)
    if song:
        await create_or_update_song(
            session,
            song.song_id,
            song.title,
            song.youtube_url,
            song.lyrics,
            **metadata,
        )

    return metadata


async def extract_metadata_for_song(
    session: AsyncSession,
    song_id: str,
    force: bool = False,
) -> dict:
    """
    Extract metadata for a song from its audio file.

    Args:
        session: Database session
        song_id: Song identifier
        force: If True, re-extract even if metadata already exists

    Returns:
        Dictionary with extracted metadata

    Raises:
        ValueError: If song not found or audio file path missing
    """
    song = await get_song(session, song_id)
    if not song:
        raise ValueError(f"Song not found: {song_id}")

    if not song.audio_file_path:
        raise ValueError(f"Audio file path not set for song: {song_id}")

    # Skip expensive extraction if metadata already exists (unless forced)
    if not force and song.bpm is not None:
        audio_path = Path(song.audio_file_path)
        if audio_path.exists():
            # Return existing metadata
            return {
                "bpm": song.bpm,
                "key": song.key,
                "duration": song.duration,
            }

    audio_path = Path(song.audio_file_path)
    return await extract_metadata(session, song_id, audio_path)


