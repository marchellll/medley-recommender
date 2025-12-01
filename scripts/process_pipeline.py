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


def compute_song_hash(song_data: dict) -> str:
    """Compute deterministic hash for a song."""
    # Create a stable representation
    stable_repr = json.dumps(
        {
            "song_id": song_data["song_id"],
            "title": song_data["title"],
            "artist": song_data["artist"],
            "youtube_url": song_data["youtube_url"],
            "lyrics": song_data["lyrics"],
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
) -> None:
    """Process download step."""
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping download step[/blue]\n")
        return

    if not quiet:
        console.print("[cyan]📥 Step 1: Downloading audio from YouTube...[/cyan]")
    task = progress.add_task("[cyan]📥 Downloading audio...", total=len(songs))

    for song_data in songs:
        song_id = song_data["song_id"]
        progress.update(task, description=f"[cyan]📥 Downloading: {song_data['title']}")

        # Check if already downloaded
        song = await get_song(session, song_id)
        if not force and song and song.audio_file_path:
            audio_path = Path(song.audio_file_path)
            if audio_path.exists():
                results[song_id]["download"] = "⏸ skipped"
                progress.advance(task)
                continue

        # Download - fail immediately on error
        try:
            await download_audio_for_song(
                session,
                song_id,
                song_data["title"],
                song_data["artist"],
                song_data["youtube_url"],
                song_data["lyrics"],
            )
            results[song_id]["download"] = "✓ success"
        except Exception as e:
            error_msg = str(e)
            # Log to file with full details
            logger.error(
                f"Download failed for song {song_id} ({song_data['title']}): {error_msg}",
                exc_info=True,
                extra={
                    "song_id": song_id,
                    "title": song_data["title"],
                    "artist": song_data["artist"],
                    "youtube_url": song_data["youtube_url"],
                }
            )
            # Fail immediately
            raise RuntimeError(f"Download failed for {song_data['title']}: {error_msg}") from e

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
    """Process metadata extraction step."""
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping metadata extraction step[/blue]\n")
        return

    if not quiet:
        console.print("\n[yellow]🔍 Step 2: Extracting metadata (BPM, key, duration)...[/yellow]")
    task = progress.add_task("[yellow]🔍 Extracting metadata...", total=len(songs))

    for song_data in songs:
        song_id = song_data["song_id"]
        progress.update(task, description=f"[yellow]🔍 Extracting: {song_data['title']}")

        # Check if already extracted
        song = await get_song(session, song_id)
        if not force and song and song.bpm is not None:
            results[song_id]["metadata"] = "⏸ skipped"
            progress.advance(task)
            continue

        # Extract metadata - fail immediately on error
        try:
            await extract_metadata_for_song(session, song_id)
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
) -> None:
    """Process embeddings generation step."""
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping embeddings generation step[/blue]\n")
        return

    if not quiet:
        console.print("\n[green]🧠 Step 3: Generating embeddings from lyrics...[/green]")

    # Load model first (with progress indication)
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

    task = progress.add_task("[green]🧠 Generating embeddings...", total=len(songs))

    for song_data in songs:
        song_id = song_data["song_id"]
        progress.update(task, description=f"[green]🧠 Embedding: {song_data['title']}")

        # Check if already generated
        song = await get_song(session, song_id)
        embedding_file = settings.embeddings_dir / f"{song_id}.json"
        if not force and song and song.embedding_file_path and embedding_file.exists():
            # Verify hash matches
            song_hash = compute_song_hash(song_data)
            # For idempotency, we could check hash, but for now just check existence
            results[song_id]["embeddings"] = "⏸ skipped"
            progress.advance(task)
            continue

        # Generate embedding - fail immediately on error
        try:
            await generate_embedding_for_song(session, song_id, model=model)
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
    progress: Progress,
    skip: bool,
    results: dict,
    console: Console,
    quiet: bool,
) -> None:
    """Process index building step."""
    if skip:
        if not quiet:
            console.print("[blue]⏸ Skipping index building step[/blue]\n")
        return

    if not quiet:
        console.print("\n[magenta]📊 Step 4: Building ANN index...[/magenta]")
    task = progress.add_task("[magenta]📊 Building index...", total=1)

    # Build index - fail immediately on error
    try:
        # Check if index exists
        index_file = settings.index_dir / "hnsw_index.bin"
        mapping_file = settings.index_dir / "id_mapping.json"

        # Build index (will rebuild if embeddings changed)
        build_index()
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
        "data/songs.json",
        "--input",
        "-i",
        help="Path to songs.json file",
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
) -> None:
    """Process songs through the complete pipeline."""
    start_time = datetime.now()

    # Load songs
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r") as f:
        songs = json.load(f)

    if not songs:
        console.print("[yellow]Warning:[/yellow] No songs found in input file")
        sys.exit(0)

    # Initialize results tracking
    results = {
        song["song_id"]: {
            "title": song["title"],
            "artist": song["artist"],
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
        console.print(f"[cyan]Found {len(songs)} song(s) to process[/cyan]\n")
        log_file = Path("logs") / "pipeline_errors.log"
        console.print(f"[dim]Error logs will be saved to: {log_file}[/dim]\n")

    logger.info(f"Starting pipeline with {len(songs)} songs")

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
                session, songs, progress, force, skip_download, results, console, quiet
            )

            # Step 2: Metadata
            await process_metadata_step(
                session, songs, progress, force, skip_metadata, results, console, quiet
            )

            # Step 3: Embeddings
            await process_embeddings_step(
                session, songs, progress, force, skip_embeddings, results, console, quiet
            )

            # Step 4: Index
            await process_index_step(progress, skip_index, results, console, quiet)

    try:
        with progress:
            asyncio.run(run_pipeline())

        if not quiet:
            console.print("\n[green]✓ Pipeline completed![/green]\n")

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
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    successful = len(songs)  # All succeeded if we got here
    failed = 0  # No failures if we got here (would have raised exception)

    # Display results
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
        console.print(json.dumps(output, indent=2))
    else:
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
                f"{result['title']} - {result['artist']}",
                result.get("download", "pending"),
                result.get("metadata", "pending"),
                result.get("embeddings", "pending"),
                "✓" if results.get("_index", {}).get("status", "").startswith("✓") else "pending",
            )

        console.print("\n")
        console.print(table)

        # Statistics
        stats_table = Table(title="Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Total Songs", str(len(songs)))
        stats_table.add_row("Successful", str(successful))
        stats_table.add_row("Failed", str(failed))
        stats_table.add_row("Duration", f"{duration:.2f}s")

        console.print("\n")
        console.print(stats_table)

    # Exit code - if we got here, everything succeeded
    sys.exit(0)


if __name__ == "__main__":
    app()


