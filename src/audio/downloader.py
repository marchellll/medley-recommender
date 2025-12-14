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


async def extract_title_from_youtube(youtube_url: str) -> str:
    """
    Extract title from YouTube video metadata.

    Args:
        youtube_url: YouTube video URL

    Returns:
        Video title

    Raises:
        RuntimeError: If extraction fails
    """
    loop = asyncio.get_event_loop()

    def _extract_title() -> str:
        """Extract title synchronously."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Unknown Title")

    try:
        title = await loop.run_in_executor(None, _extract_title)
        return title
    except Exception as e:
        logger.warning(f"Failed to extract title from YouTube: {e}")
        return "Unknown Title"


async def download_audio(
    song_id: str,
    youtube_url: str,
    output_path: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
    title: Optional[str] = None,
    verbose: bool = False,
) -> tuple[Path, str]:
    """
    Download audio from YouTube URL.

    Args:
        song_id: Unique song identifier
        youtube_url: YouTube video URL
        output_path: Optional output path (defaults to data/audio/{song_id}.mp3)
        progress_callback: Optional callback for progress updates
        title: Optional title to avoid YouTube API call if file already exists

    Returns:
        Tuple of (path to downloaded audio file, video title)

    Raises:
        ValueError: If URL is invalid
        RuntimeError: If download fails
    """
    if not validate_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

    if output_path is None:
        # Convert to MP3 format (librosa can read this natively)
        output_path = settings.audio_dir / f"{song_id}.mp3"

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        logger.info(f"    Output path: {output_path}")
        logger.info(f"    File exists: {output_path.exists()}")

    # Check if file already exists (skip expensive download)
    if output_path.exists():
        if verbose:
            logger.info(f"    File already exists at {output_path}")
        # Use provided title if available, otherwise use default to avoid network call
        # Note: This function is typically called from download_audio_for_song which
        # already checks and returns early, so this path is rarely taken
        if title:
            if verbose:
                logger.info(f"    Using provided title: '{title}'")
            return output_path, title
        # File exists but no title provided - use default instead of making network call
        # The title should have been provided by the caller if available
        if verbose:
            logger.warning(f"    File {output_path} exists but no title provided, using default title")
        else:
            logger.warning(f"File {output_path} exists but no title provided, using default title")
        return output_path, "Unknown Title"

    # Configure yt-dlp options - download and convert to MP3 using post-processor
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",  # MP3 bitrate: 192kbps
            }
        ],
        "quiet": not verbose,  # Show yt-dlp output if verbose
        "no_warnings": not verbose,
    }
    if verbose:
        logger.info(f"    Starting download from YouTube: {youtube_url}")
        logger.info(f"    Output will be saved to: {output_path}")

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

    title = "Unknown Title"
    # Capture verbose in closure
    verbose_flag = verbose

    def _download() -> str:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get title
                try:
                    if verbose_flag:
                        logger.info(f"    Extracting video info from YouTube...")
                    info = ydl.extract_info(youtube_url, download=False)
                    title = info.get("title", "Unknown Title")
                    if verbose_flag:
                        logger.info(f"    Video title: '{title}'")
                        logger.info(f"    Duration: {info.get('duration', 'unknown')} seconds")
                    logger.debug(f"Video info extracted for {youtube_url}: {json.dumps({
                        'title': title,
                        'duration': info.get('duration'),
                        'formats_count': len(info.get('formats', [])),
                    }, indent=2)}")
                except Exception as info_error:
                    logger.warning(f"Could not extract video info: {info_error}")

                # Now download
                if verbose_flag:
                    logger.info(f"    Downloading audio...")
                ydl.download([youtube_url])
                if verbose_flag:
                    logger.info(f"    Download completed")
                return title
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
        title = await loop.run_in_executor(None, _download)
        if verbose:
            logger.info(f"    Successfully downloaded: {title}")
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

    # FFmpegExtractAudio creates .mp3 file, find it and rename if needed
    mp3_file = output_path
    if not mp3_file.exists():
        # Try to find mp3 file (post-processor changes extension)
        possible_files = list(output_path.parent.glob(f"{song_id}*.mp3"))
        if possible_files:
            mp3_file = possible_files[0]
            # Rename to expected name if different
            if mp3_file != output_path:
                mp3_file.rename(output_path)
        else:
            raise RuntimeError(f"Download completed but .mp3 file not found at {output_path}")

    # Verify file was created
    if not output_path.exists():
        raise RuntimeError("Download completed but file not found")

    return output_path, title


async def download_audio_for_song(
    session: AsyncSession,
    song_id: str,
    youtube_url: str,
    lyrics: str = "",
    progress_callback: Optional[callable] = None,
    force: bool = False,
    verbose: bool = False,
) -> tuple[Path, str]:
    """
    Download audio for a song and update database.

    Optimizations:
    - Skips download if audio file already exists (unless forced)
    - Checks both database record AND file system existence
    - Avoids expensive YouTube download and API calls when file exists

    Args:
        session: Database session
        song_id: Unique song identifier
        youtube_url: YouTube video URL
        lyrics: Song lyrics (optional, can be empty initially)
        progress_callback: Optional callback for progress updates
        force: If True, re-download even if file already exists
        verbose: If True, log detailed information about the download process

    Returns:
        Tuple of (path to downloaded audio file, video title)
    """
    from src.database.db import get_song

    # Get song from database (if exists)
    song = await get_song(session, song_id)
    if verbose:
        if song:
            logger.info(f"  Found existing DB record for {song_id}: title='{song.title}', audio_file_path='{song.audio_file_path}'")
        else:
            logger.info(f"  No existing DB record for {song_id}")

    # Determine expected file path
    expected_path = settings.audio_dir / f"{song_id}.mp3"
    if verbose:
        logger.info(f"  Expected file path: {expected_path}")

    # Check if audio file already exists (skip expensive download unless forced)
    if not force:
        # First check: file exists at expected path
        if expected_path.exists():
            if verbose:
                logger.info(f"  ✓ File exists at expected path: {expected_path}")
            # Update DB with file path if missing
            if song and not song.audio_file_path:
                if verbose:
                    logger.info(f"  Updating DB record with file path: {expected_path}")
                song.audio_file_path = str(expected_path.resolve())
                await session.commit()
            # Return with title from DB if available, otherwise use default
            title = song.title if song else "Unknown Title"
            if verbose:
                logger.info(f"  Using title from DB: '{title}'")
            # Create DB record if it doesn't exist
            if not song:
                if verbose:
                    logger.info(f"  Creating new DB record for {song_id}")
                await create_or_update_song(
                    session,
                    song_id,
                    title,
                    youtube_url,
                    lyrics,
                    audio_file_path=str(expected_path.resolve()),
                )
            if verbose:
                logger.info(f"  ⏸ Skipping download for {song_id} (file already exists)")
            return expected_path, title

        # Second check: file exists at path stored in DB (defense in depth)
        if song and song.audio_file_path and Path(song.audio_file_path).exists():
            if verbose:
                logger.info(f"  ✓ File exists at DB path: {song.audio_file_path}")
                logger.info(f"  ⏸ Skipping download for {song_id} (file already exists at DB path)")
            return Path(song.audio_file_path), song.title

    # File doesn't exist - download audio (this also extracts title from YouTube)
    if verbose:
        logger.info(f"  File doesn't exist, proceeding with download...")
    # Pass title from DB if available to avoid network call if file somehow exists
    title_from_db = song.title if song else None
    if verbose and title_from_db:
        logger.info(f"  Using title from DB to avoid network call: '{title_from_db}'")
    audio_path, title = await download_audio(
        song_id,
        youtube_url,
        progress_callback=progress_callback,
        title=title_from_db,
        verbose=verbose,
    )

    # Update database
    if verbose:
        logger.info(f"  Updating database record for {song_id}")
    await create_or_update_song(
        session,
        song_id,
        title,
        youtube_url,
        lyrics,
        audio_file_path=str(audio_path.resolve()),
    )
    if verbose:
        logger.info(f"  ✓ Download complete for {song_id}: '{title}' -> {audio_path}")

    return audio_path, title


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


