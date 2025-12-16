"""Master pipeline script that orchestrates the entire data processing pipeline."""

import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from sqlalchemy.ext.asyncio import AsyncSession

from src.audio.downloader import download_audio_for_song, compute_file_hash
from src.audio.extractor import extract_metadata_for_song
from src.database.db import AsyncSessionLocal, create_or_update_song, get_song, init_db
from src.embeddings.encoder import generate_embedding_for_song
from src.index.builder import build_index
from src.utils.config import settings

app = typer.Typer(help="Process worship songs through the complete pipeline")
console = Console()


def setup_logging() -> logging.Logger:
    """Set up file logging for errors."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline_errors.log"

    # Create logger
    logger = logging.getLogger("medley_pipeline")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Detailed formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logging()


def extract_video_id(youtube_url: str) -> str:
    """Extract video ID from YouTube URL."""
    import re
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


def compute_song_hash(song_data: dict) -> str:
    """Compute deterministic hash for a song."""
    # Create a stable representation
    stable_repr = json.dumps(
        {
            "song_id": song_data["song_id"],
            "title": song_data.get("title", ""),
            "youtube_url": song_data["youtube_url"],
            "lyrics": song_data.get("lyrics", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha256(stable_repr.encode()).hexdigest()


def create_header() -> Panel:
    """Create header panel."""
    header_text = Text("🎵 Medley Recommender Pipeline", style="bold cyan")
    header_text.append("\n", style="default")
    header_text.append("Processing worship songs through the complete pipeline", style="dim")
    return Panel(header_text, border_style="cyan")


def create_status_panel(current_step: str, current_song: Optional[str] = None) -> Panel:
    """Create status panel."""
    status_lines = [
        f"[cyan]Step:[/cyan] {current_step}",
    ]
    if current_song:
        status_lines.append(f"[cyan]Song:[/cyan] {current_song}")
    return Panel("\n".join(status_lines), title="Status", border_style="blue")


async def process_download_step(
    session: AsyncSession,
    songs: list[dict],
    progress: Progress,
    force: bool,
    skip: bool,
    results: dict,
    console: Console,
    quiet: bool,
    verbose: bool = False,
) -> None:
    """
    Process download step.

    Optimizations:
    - Skips download if audio file already exists (unless --force)
    - Checks both database entry and file system existence
    - Avoids expensive YouTube download and API calls when file exists
    - Double-checked in download_audio_for_song for defense-in-depth
    """
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping download step[/blue]\n")
        return

    if not songs:
        if not quiet:
            console.print("[blue]⏸ No songs to download (links.json is empty)[/blue]\n")
        return

    if not quiet:
        console.print("[cyan]📥 Step 1: Downloading audio from YouTube...[/cyan]")
    task = progress.add_task("[cyan]📥 Downloading audio...", total=len(songs))

    for song_data in songs:
        song_id = song_data["song_id"]
        youtube_url = song_data["youtube_url"]
        progress.update(task, description=f"[cyan]📥 Downloading: {song_id}")

        # Check if already downloaded - check both DB record AND file system
        song = await get_song(session, song_id)
        expected_path = settings.audio_dir / f"{song_id}.mp3"

        # Skip if file exists (either from DB record or on disk)
        if not force:
            # Check DB record first
            if song and song.audio_file_path:
                audio_path = Path(song.audio_file_path)
                if audio_path.exists():
                    if verbose:
                        msg = f"⏸ Skipping {song_id}: File exists at {audio_path} (from DB record)"
                        logger.info(msg)
                        console.print(f"[dim]{msg}[/dim]")
                    results[song_id]["download"] = "⏸ skipped"
                    progress.advance(task)
                    continue
            # Also check expected file path even if no DB record
            elif expected_path.exists():
                if verbose:
                    msg = f"⏸ Skipping {song_id}: File exists at {expected_path} (no DB record)"
                    logger.info(msg)
                    console.print(f"[dim]{msg}[/dim]")
                results[song_id]["download"] = "⏸ skipped"
                progress.advance(task)
                continue

        # Download - fail immediately on error
        if verbose:
            msg = f"📥 Downloading {song_id} from {youtube_url}"
            logger.info(msg)
            console.print(f"[cyan]{msg}[/cyan]")
        try:
            audio_path, title = await download_audio_for_song(
                session,
                song_id,
                youtube_url,
                lyrics="",  # Lyrics should be added manually to database
                force=force,
                verbose=verbose,
            )
            # Update song_data with extracted title
            song_data["title"] = title
            results[song_id]["download"] = "✓ success"
        except Exception as e:
            error_msg = str(e)
            # Log to file with full details
            logger.error(
                f"Download failed for song {song_id}: {error_msg}",
                exc_info=True,
                extra={
                    "song_id": song_id,
                    "youtube_url": youtube_url,
                }
            )
            # Fail immediately
            raise RuntimeError(f"Download failed for {song_id}: {error_msg}") from e

        progress.advance(task)


async def process_metadata_step(
    session: AsyncSession,
    songs: list[dict],
    progress: Progress,
    force: bool,
    skip: bool,
    results: dict,
    console: Console,
    quiet: bool,
) -> None:
    """
    Process metadata extraction step.

    Optimizations:
    - Skips extraction if BPM already exists in database (unless --force)
    - Verifies audio file exists before skipping
    - Avoids expensive librosa processing when metadata already exists
    """
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping metadata extraction step[/blue]\n")
        return

    if not songs:
        if not quiet:
            console.print("[blue]⏸ No songs to process (links.json is empty)[/blue]\n")
        return

    if not quiet:
        console.print("\n[yellow]🔍 Step 2: Extracting metadata (BPM, key, duration)...[/yellow]")
    task = progress.add_task("[yellow]🔍 Extracting metadata...", total=len(songs))

    for song_data in songs:
        song_id = song_data["song_id"]
        progress.update(task, description=f"[yellow]🔍 Extracting: {song_data['title']}")

        # Check if already extracted (verify both database entry and audio file exist)
        song = await get_song(session, song_id)
        if not force and song and song.bpm is not None:
            # Also verify audio file exists before skipping
            if song.audio_file_path and Path(song.audio_file_path).exists():
                results[song_id]["metadata"] = "⏸ skipped"
            progress.advance(task)
            continue

        # Extract metadata - fail immediately on error
        try:
            await extract_metadata_for_song(session, song_id, force=force)
            results[song_id]["metadata"] = "✓ success"
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Metadata extraction failed for song {song_id}: {error_msg}",
                exc_info=True,
                extra={"song_id": song_id, "title": song_data.get("title", "unknown")}
            )
            # Fail immediately
            raise RuntimeError(f"Metadata extraction failed for {song_data.get('title', song_id)}: {error_msg}") from e

        progress.advance(task)


async def process_embeddings_step(
    session: AsyncSession,
    songs: list[dict],
    progress: Progress,
    force: bool,
    skip: bool,
    results: dict,
    console: Console,
    quiet: bool,
    all_songs: bool = False,
) -> None:
    """
    Process embeddings generation step.

    Optimizations:
    - Pre-checks which songs need processing BEFORE loading expensive model
    - Skips model loading entirely if all embeddings already exist
    - Skips generation if embedding file already exists (unless --force)
    - Avoids expensive model loading and embedding generation when not needed
    """
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping embeddings generation step[/blue]\n")
        return

    if not songs:
        if not quiet:
            if all_songs:
                console.print("[blue]⏸ No songs found in database[/blue]\n")
            else:
                console.print("[blue]⏸ No songs to process (links.json is empty)[/blue]\n")
        return

    if not quiet:
        console.print("\n[green]🧠 Step 3: Generating embeddings from lyrics...[/green]")

    # CRITICAL: Validate all songs in database have lyrics before processing embeddings
    # This prevents wasting time loading the model if songs are missing lyrics
    # Check ALL songs in database, not just ones from links.json
    from src.database.db import get_all_songs

    all_songs_in_db = await get_all_songs(session)
    songs_missing_lyrics = []

    for song in all_songs_in_db:
        if not song.lyrics or not song.lyrics.strip():
            songs_missing_lyrics.append((song.song_id, song.title or song.song_id))

    if songs_missing_lyrics:
        error_msg = "\n" + "="*80 + "\n"
        error_msg += "❌ ERROR: Songs missing lyrics cannot be processed for embeddings!\n"
        error_msg += "="*80 + "\n\n"
        error_msg += f"Found {len(songs_missing_lyrics)} song(s) in database missing lyrics:\n\n"

        # Show first 20 songs to avoid overwhelming output
        display_count = min(20, len(songs_missing_lyrics))
        for song_id, title in songs_missing_lyrics[:display_count]:
            error_msg += f"  • {song_id}"
            if title != song_id:
                error_msg += f" ({title})"
            error_msg += "\n"

        if len(songs_missing_lyrics) > display_count:
            error_msg += f"  ... and {len(songs_missing_lyrics) - display_count} more song(s)\n"

        error_msg += "\n"
        error_msg += "Please add lyrics manually to ALL songs before running embeddings.\n"
        error_msg += "You can update lyrics in the database or use the API to add them.\n"
        error_msg += "="*80 + "\n"

        console.print(f"[red]{error_msg}[/red]")
        logger.error(f"Embeddings step failed: {len(songs_missing_lyrics)} songs missing lyrics")
        raise RuntimeError(f"Cannot generate embeddings: {len(songs_missing_lyrics)} song(s) in database are missing lyrics. Please add lyrics manually to all songs before running embeddings.")

    if not quiet:
        console.print("[green]✓ All songs have lyrics, proceeding with embeddings...[/green]\n")

    # Pre-check which songs need embedding generation (before loading expensive model)
    songs_to_process = []
    for song_data in songs:
        song_id = song_data["song_id"]
        song = await get_song(session, song_id)
        embedding_file = settings.embeddings_dir / f"{song_id}.json"
        if not force and song and song.embedding_file_path and embedding_file.exists():
            results[song_id]["embeddings"] = "⏸ skipped"
        else:
            songs_to_process.append(song_data)

    # If no songs need processing, skip model loading
    if not songs_to_process:
        if not quiet:
            console.print("[blue]⏸ All embeddings already exist, skipping model load[/blue]\n")
        return

    # Load model only if we have songs to process (expensive operation)
    model_task = None
    if not quiet:
        model_task = progress.add_task(
            "[yellow]📦 Loading embedding model...",
            total=None  # Indeterminate progress
        )

    def model_progress_callback(message: str) -> None:
        """Callback for model loading progress."""
        if model_task is not None and not quiet:
            progress.update(model_task, description=f"[yellow]📦 {message}")

    # Load model (this may download it on first run)
    from src.embeddings.model_loader import get_embedding_model
    model = get_embedding_model(progress_callback=model_progress_callback)

    if model_task is not None and not quiet:
        progress.remove_task(model_task)
        console.print("[green]✓ Model ready[/green]")

    task = progress.add_task("[green]🧠 Generating embeddings...", total=len(songs_to_process))

    for song_data in songs_to_process:
        song_id = song_data["song_id"]
        progress.update(task, description=f"[green]🧠 Embedding: {song_data.get('title', song_id)}")

        # Generate embedding - fail immediately on error
        try:
            await generate_embedding_for_song(session, song_id, model=model, force=force)
            results[song_id]["embeddings"] = "✓ success"
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Embedding generation failed for song {song_id}: {error_msg}",
                exc_info=True,
                extra={"song_id": song_id, "title": song_data.get("title", "unknown")}
            )
            # Fail immediately
            raise RuntimeError(f"Embedding generation failed for {song_data.get('title', song_id)}: {error_msg}") from e

        progress.advance(task)


async def process_index_step(
    session: AsyncSession,
    progress: Progress,
    skip: bool,
    results: dict,
    console: Console,
    quiet: bool,
) -> None:
    """Process index building step.

    Note: This step builds the index from ALL embedding files in the embeddings directory,
    not just the ones from links.json. This ensures existing embeddings are always included
    even if links.json is empty.
    """
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping index building step[/blue]\n")
        return

    if not quiet:
        console.print("\n[magenta]📊 Step 4: Building ANN index from all embeddings...[/magenta]")
        console.print("[dim]Note: Index includes ALL embeddings from directory, not just links.json[/dim]")
    task = progress.add_task("[magenta]📊 Building index...", total=1)

    # Build index - fail immediately on error
    try:
        # Build index from ALL embedding files in the directory
        # This ensures existing embeddings are included even if links.json is empty
        await build_index(session=session)
        results["_index"] = {"status": "✓ success"}
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Index building failed: {error_msg}",
            exc_info=True
        )
        # Fail immediately
        raise RuntimeError(f"Index building failed: {error_msg}") from e

    progress.advance(task)


@app.command()
def main(
    input_file: Path = typer.Option(
        "data/links.json",
        "--input",
        "-i",
        help="Path to links.json file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing"),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip download step"),
    skip_metadata: bool = typer.Option(False, "--skip-metadata", help="Skip metadata extraction"),
    skip_embeddings: bool = typer.Option(
        False, "--skip-embeddings", help="Skip embedding generation"
    ),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip index building"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    all_songs: bool = typer.Option(False, "--all-songs", help="Process ALL songs from database, not just links.json"),
    clean_embeddings: bool = typer.Option(False, "--clean-embeddings", help="Delete all embedding files before regeneration (implies --all-songs --force --skip-download --skip-metadata)"),
) -> None:
    """Process songs through the complete pipeline."""
    start_time = datetime.now()

    # Load links
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r") as f:
        links_data = json.load(f)

    # Convert links to songs format
    songs = []
    if not links_data:
        if not quiet:
            console.print("[yellow]Warning:[/yellow] No links found in input file")
            console.print("[yellow]Note:[/yellow] Will only process index step (if not skipped) to include existing embeddings\n")
    else:
        for link_entry in links_data:
            if isinstance(link_entry, dict) and "url" in link_entry:
                youtube_url = link_entry["url"]
            elif isinstance(link_entry, str):
                youtube_url = link_entry
            else:
                console.print(f"[yellow]Warning:[/yellow] Invalid link entry: {link_entry}")
                continue

            # Generate song_id from YouTube URL
            try:
                video_id = extract_video_id(youtube_url)
                song_id = f"youtube/{video_id}"
                songs.append({
                    "song_id": song_id,
                    "youtube_url": youtube_url,
                    "title": "",  # Will be extracted during download
                })
            except ValueError as e:
                console.print(f"[yellow]Warning:[/yellow] Skipping invalid URL: {youtube_url} - {e}")
                continue

    # If --all-songs is used, load all songs from database instead
    if all_songs:
        async def load_all_songs_from_db():
            async with AsyncSessionLocal() as session:
                from src.database.db import get_all_songs
                all_songs_list = await get_all_songs(session)
                return [
                    {
                        "song_id": song.song_id,
                        "youtube_url": song.youtube_url or "",
                        "title": song.title or "",
                    }
                    for song in all_songs_list
                ]

        if not quiet:
            console.print("[cyan]Loading all songs from database...[/cyan]")
        songs = asyncio.run(load_all_songs_from_db())
        if not quiet:
            console.print(f"[cyan]Found {len(songs)} song(s) in database[/cyan]\n")

    # Initialize results tracking
    results = {
        song["song_id"]: {
            "title": song.get("title", ""),
            "download": "pending",
            "metadata": "pending",
            "embeddings": "pending",
            "index": "pending",
            "errors": [],
        }
        for song in songs
    }

    # Show initial status
    if not quiet:
        console.print(create_header())
        if songs:
            console.print(f"[cyan]Found {len(songs)} song(s) to process[/cyan]\n")
        else:
            console.print("[cyan]No new songs to process from links.json[/cyan]\n")
            console.print("[cyan]Note:[/cyan] Existing data (audio, embeddings, database) will be preserved\n")
        log_file = Path("logs") / "pipeline_errors.log"
        console.print(f"[dim]Error logs will be saved to: {log_file}[/dim]\n")

    logger.info(f"Starting pipeline with {len(songs)} songs from links.json")

    # Initialize database
    if not quiet:
        console.print("[yellow]Initializing database...[/yellow]")
    try:
        asyncio.run(init_db())
        if not quiet:
            console.print("[green]✓ Database initialized[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Database initialization failed: {str(e)}[/red]")
        sys.exit(1)

    # Handle --clean-embeddings flag (shorthand for common cleanup scenario)
    if clean_embeddings:
        if not quiet:
            console.print("[yellow]🧹 --clean-embeddings flag detected[/yellow]")
            console.print("[yellow]This will delete all embedding files and Chroma index, then regenerate them[/yellow]\n")
        all_songs = True
        force = True
        skip_download = True
        skip_metadata = True
        skip_index = True  # Skip index by default - user can rebuild separately
        # Clean embeddings directory
        embeddings_dir = settings.embeddings_dir
        if embeddings_dir.exists():
            deleted_count = 0
            for embedding_file in embeddings_dir.rglob("*.json"):
                try:
                    embedding_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    if not quiet:
                        console.print(f"[yellow]Warning:[/yellow] Could not delete {embedding_file}: {e}")
            if not quiet:
                console.print(f"[green]✓ Deleted {deleted_count} embedding file(s)[/green]")

        # Also delete Chroma index to avoid dimension mismatch errors
        chroma_db_path = settings.index_dir / "chroma_db"
        if chroma_db_path.exists():
            import shutil
            try:
                shutil.rmtree(chroma_db_path)
                if not quiet:
                    console.print(f"[green]✓ Deleted Chroma index directory[/green]\n")
                logger.info(f"Deleted Chroma index directory: {chroma_db_path}")
            except Exception as e:
                if not quiet:
                    console.print(f"[yellow]Warning:[/yellow] Could not delete Chroma index: {e}\n")
                logger.warning(f"Could not delete Chroma index: {e}")

    # Create progress display
    if quiet:
        progress = Progress(console=console, disable=True)
    else:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    # Run pipeline
    async def run_pipeline():
        """Run the async pipeline."""
        async with AsyncSessionLocal() as session:
            # Step 1: Download
            await process_download_step(
                session, songs, progress, force, skip_download, results, console, quiet, verbose
            )

            # Step 2: Metadata
            await process_metadata_step(
                session, songs, progress, force, skip_metadata, results, console, quiet
            )

            # Step 3: Embeddings
            await process_embeddings_step(
                session, songs, progress, force, skip_embeddings, results, console, quiet, all_songs
            )

            # Step 4: Index
            await process_index_step(session, progress, skip_index, results, console, quiet)

    try:
        logger.info("Starting pipeline execution")
        with progress:
            logger.info("Running async pipeline")
            asyncio.run(run_pipeline())
            logger.info("Async pipeline completed")

        if not quiet:
            console.print("\n[green]✓ Pipeline completed![/green]\n")
        logger.info("Pipeline completion message displayed")

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        console.print(f"[red]Critical error:[/red] {error_msg}")
        logger.critical(f"Critical pipeline error: {error_msg}", exc_info=True)
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        sys.exit(2)

    # Calculate statistics
    logger.info("Starting statistics calculation")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Duration calculated: {duration:.2f}s")

    successful = len(songs)  # All succeeded if we got here
    failed = 0  # No failures if we got here (would have raised exception)
    logger.info(f"Statistics: successful={successful}, failed={failed}")

    # Display results
    logger.info("Starting results display")
    if json_output:
        output = {
            "summary": {
                "total": len(songs),
                "successful": successful,
                "failed": failed,
                "duration_seconds": duration,
            },
            "results": results,
        }
        logger.info("Displaying JSON output")
        console.print(json.dumps(output, indent=2))
        logger.info("JSON output displayed")
    else:
        logger.info("Displaying table output")
        # Only show song table if there are songs to process
        if songs:
            # Create summary table
            table = Table(title="Pipeline Results", show_header=True, header_style="bold magenta")
            table.add_column("Song", style="cyan")
            table.add_column("Download", justify="center")
            table.add_column("Metadata", justify="center")
            table.add_column("Embeddings", justify="center")
            table.add_column("Index", justify="center")

            for song_id, result in results.items():
                if song_id == "_index":
                    continue
                table.add_row(
                    f"{result.get('title', song_id)}",
                    result.get("download", "pending"),
                    result.get("metadata", "pending"),
                    result.get("embeddings", "pending"),
                    "✓" if results.get("_index", {}).get("status", "").startswith("✓") else "pending",
                )

        logger.info("Rendering results table")
        console.print("\n")
        console.print(table)
        logger.info("Results table rendered")

        # Statistics
        logger.info("Creating statistics table")
        stats_table = Table(title="Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Songs from links.json", str(len(songs)))
        if songs:
            stats_table.add_row("Successful", str(successful))
        stats_table.add_row("Failed", str(failed))
        stats_table.add_row("Index Status", results.get("_index", {}).get("status", "pending"))
        stats_table.add_row("Duration", f"{duration:.2f}s")

        logger.info("Rendering statistics table")
        console.print("\n")
        console.print(stats_table)
        logger.info("Statistics table rendered")

        if not songs:
            logger.info("Displaying no-songs note")
            console.print("\n[yellow]Note:[/yellow] No songs were processed from links.json, but existing data was preserved.")
            console.print("[yellow]The index was rebuilt from all existing embeddings in the directory.[/yellow]")

    logger.info("All output completed, preparing to exit")
    # Exit code - if we got here, everything succeeded
    logger.info("Calling sys.exit(0)")
    sys.exit(0)


if __name__ == "__main__":
    app()


