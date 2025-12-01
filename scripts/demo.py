"""Demo script showing end-to-end workflow."""

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.database.db import init_db
from src.index.searcher import IndexSearcher

console = Console()


def main():
    """Run demo workflow."""
    console.print(Panel.fit("[bold cyan]🎵 Medley Recommender Demo[/bold cyan]"))

    # Check if pipeline has been run
    songs_file = Path("data/songs.json")
    if not songs_file.exists():
        console.print("[red]Error:[/red] data/songs.json not found. Please create it first.")
        return

    chroma_db_path = Path("data/index/chroma_db")
    if not chroma_db_path.exists():
        console.print(
            "[yellow]Warning:[/yellow] Index not found. Please run the pipeline first:"
        )
        console.print("  [cyan]uv run python scripts/process_pipeline.py[/cyan]")
        return

    # Initialize database
    asyncio.run(init_db())

    # Load searcher
    console.print("\n[green]Loading index...[/green]")
    searcher = IndexSearcher()
    searcher.load()
    console.print("[green]✓ Index loaded[/green]")

    # Demo searches
    console.print("\n[bold]Demo Searches:[/bold]\n")

    queries = [
        "amazing grace",
        "praise and worship",
        "God's love",
    ]

    for query in queries:
        console.print(f"[cyan]Query:[/cyan] '{query}'")
        results = searcher.search_by_text(query, k=3)

        if results:
            for i, (song_id, distance) in enumerate(results, 1):
                similarity = 1.0 - distance
                console.print(
                    f"  {i}. Song ID: {song_id}, Similarity: {similarity:.3f}"
                )
        else:
            console.print("  No results found")

        console.print()

    console.print("[green]✓ Demo complete![/green]")


if __name__ == "__main__":
    main()


