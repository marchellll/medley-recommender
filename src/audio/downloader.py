"""YouTube audio download functionality."""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import yt_dlp
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.db import create_or_update_song
from src.utils.config import settings

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for filesystem compatibility.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def validate_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, False otherwise
    """
    patterns = [
        r"^https?://(www\.)?(youtube\.com|youtu\.be)/",
        r"youtube\.com/watch\?v=[\w-]+",
        r"youtu\.be/[\w-]+",
    ]
    return any(re.match(pattern, url) for pattern in patterns)


async def download_audio(
    song_id: str,
    youtube_url: str,
    output_path: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
) -> Path:
    """
    Download audio from YouTube URL.

    Args:
        song_id: Unique song identifier
        youtube_url: YouTube video URL
        output_path: Optional output path (defaults to data/audio/{song_id}.wav)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to downloaded audio file

    Raises:
        ValueError: If URL is invalid
        RuntimeError: If download fails
    """
    if not validate_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

    if output_path is None:
        # Convert to WAV format (librosa/PySoundFile can read this natively)
        output_path = settings.audio_dir / f"{song_id}.wav"

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if output_path.exists():
        return output_path

    # Configure yt-dlp options - download and convert to WAV using post-processor
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",  # Not used for WAV but kept for compatibility
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    # Add progress hook if callback provided
    if progress_callback:

        def progress_hook(d: dict) -> None:
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                downloaded = d.get("downloaded_bytes", 0)
                if total > 0:
                    percent = (downloaded / total) * 100
                    progress_callback(percent, downloaded, total)

        ydl_opts["progress_hooks"] = [progress_hook]

    # Run download in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def _download() -> None:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first for logging
                try:
                    info = ydl.extract_info(youtube_url, download=False)
                    logger.debug(f"Video info extracted for {youtube_url}: {json.dumps({
                        'title': info.get('title'),
                        'duration': info.get('duration'),
                        'formats_count': len(info.get('formats', [])),
                    }, indent=2)}")
                except Exception as info_error:
                    logger.warning(f"Could not extract video info: {info_error}")

                # Now download
                ydl.download([youtube_url])
        except yt_dlp.utils.DownloadError as e:
            # Log detailed download error
            error_details = {
                "url": youtube_url,
                "song_id": song_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "yt_dlp_opts": ydl_opts,
            }
            logger.error(f"Download error for {song_id}: {json.dumps(error_details, indent=2, default=str)}")
            raise
        except Exception as e:
            # Log other errors
            error_details = {
                "url": youtube_url,
                "song_id": song_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            logger.error(f"Unexpected error downloading {song_id}: {json.dumps(error_details, indent=2, default=str)}")
            raise

    try:
        await loop.run_in_executor(None, _download)
    except yt_dlp.utils.DownloadError as e:
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        error_msg = f"Failed to download audio from {youtube_url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        # Clean up partial file if it exists
        if output_path.exists():
            output_path.unlink()
        error_msg = f"Failed to download audio: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

    # FFmpegExtractAudio creates .wav file, find it and rename if needed
    wav_file = output_path
    if not wav_file.exists():
        # Try to find wav file (post-processor changes extension)
        possible_files = list(output_path.parent.glob(f"{song_id}*.wav"))
        if possible_files:
            wav_file = possible_files[0]
            # Rename to expected name if different
            if wav_file != output_path:
                wav_file.rename(output_path)
        else:
            raise RuntimeError(f"Download completed but .wav file not found at {output_path}")

    # Verify file was created
    if not output_path.exists():
        raise RuntimeError("Download completed but file not found")

    return output_path


async def download_audio_for_song(
    session: AsyncSession,
    song_id: str,
    title: str,
    artist: str,
    youtube_url: str,
    lyrics: str,
    progress_callback: Optional[callable] = None,
) -> Path:
    """
    Download audio for a song and update database.

    Args:
        session: Database session
        song_id: Unique song identifier
        title: Song title
        artist: Song artist
        youtube_url: YouTube video URL
        lyrics: Song lyrics
        progress_callback: Optional callback for progress updates

    Returns:
        Path to downloaded audio file
    """
    # Check if audio file already exists in database
    from src.database.db import get_song

    song = await get_song(session, song_id)
    if song and song.audio_file_path and Path(song.audio_file_path).exists():
        return Path(song.audio_file_path)

    # Download audio
    audio_path = await download_audio(song_id, youtube_url, progress_callback=progress_callback)

    # Update database
    await create_or_update_song(
        session,
        song_id,
        title,
        artist,
        youtube_url,
        lyrics,
        audio_file_path=str(audio_path.resolve()),
    )

    return audio_path


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


