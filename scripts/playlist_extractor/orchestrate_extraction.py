#!/usr/bin/env python3
"""
Simplified orchestrator script for playlist extraction.

This script extracts video URLs from a YouTube playlist and saves them to found_links.json.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

# Import function from extract_playlist_raw
from extract_playlist_raw import extract_playlist_raw

console = Console()


class ExtractionOrchestrator:
    """Orchestrates playlist URL extraction."""

    def __init__(
        self,
        playlist_url: str,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            playlist_url: YouTube playlist URL to extract
            output_dir: Directory to save outputs (default: data/playlist-extraction)
        """
        self.playlist_url = playlist_url

        # Set up paths
        if output_dir is None:
            # Save to data/playlist-extraction folder
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / "data" / "playlist-extraction"
        else:
            self.output_dir = Path(output_dir)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define output file path
        self.found_links_path = self.output_dir / "found_links.json"

        # Track statistics
        self.stats = {
            "videos_extracted": 0,
        }

    def log_step(self, step_name: str, message: str, status: str = "info") -> None:
        """Log a step with consistent formatting."""
        status_colors = {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
        }
        color = status_colors.get(status, "cyan")
        console.print(f"[{color}][{step_name}] {message}[/{color}]")

    def log_error(self, step_name: str, error: Exception, context: Optional[str] = None) -> None:
        """Log an error with full traceback."""
        console.print(f"\n[red]✗ ERROR in {step_name}[/red]")
        if context:
            console.print(f"[red]Context: {context}[/red]")
        console.print(f"[red]Error: {error}[/red]")
        console.print("\n[dim]Full traceback:[/dim]")
        console.print(Panel(traceback.format_exc(), border_style="red", title="Traceback"))

    def step1_extract_urls(self) -> bool:
        """Step 1: Extract video URLs from playlist."""
        self.log_step("STEP 1", f"Extracting video URLs from: {self.playlist_url}")

        try:
            extract_playlist_raw(self.playlist_url, self.found_links_path)

            # Verify output
            if not self.found_links_path.exists():
                raise FileNotFoundError(f"Output file was not created: {self.found_links_path}")

            # Load and count videos
            with open(self.found_links_path, "r", encoding="utf-8") as f:
                videos = json.load(f)

            self.stats["videos_extracted"] = len(videos)
            self.log_step(
                "STEP 1",
                f"✓ Extracted {len(videos)} video URLs → {self.found_links_path}",
                "success",
            )
            return True

        except Exception as e:
            self.log_error("STEP 1", e, f"Failed to extract playlist from {self.playlist_url}")
            return False

    def print_summary(self) -> None:
        """Print a summary of the extraction process."""
        console.print("\n[bold cyan]Extraction Summary:[/bold cyan]")
        console.print(f"  [green]✓[/green] Videos Extracted: {self.stats['videos_extracted']}")
        console.print(f"  [bold green]✓[/bold green] Output File: {self.found_links_path}")

    def run(self) -> bool:
        """Run the extraction pipeline."""
        console.print("\n[bold cyan]=" * 60)
        console.print("[bold cyan]Playlist URL Extraction[/bold cyan]")
        console.print("[bold cyan]=" * 60)
        console.print(f"[cyan]Playlist URL:[/cyan] {self.playlist_url}")
        console.print(f"[cyan]Output Directory:[/cyan] {self.output_dir}")
        console.print()

        if not self.step1_extract_urls():
            console.print(f"\n[red]✗ Pipeline failed[/red]")
            console.print(f"[yellow]Check the error messages above for details.[/yellow]")
            console.print(f"[yellow]Output file location: {self.output_dir}[/yellow]")
            return False

        console.print("\n[bold green]=" * 60)
        console.print("[bold green]✓ Extraction completed successfully![/bold green]")
        console.print("[bold green]=" * 60)

        self.print_summary()
        return True


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract video URLs from YouTube playlist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract URLs from playlist
  uv run python orchestrate_extraction.py "https://www.youtube.com/playlist?list=..."

  # Custom output directory
  uv run python orchestrate_extraction.py --output-dir ./my_output "https://www.youtube.com/playlist?list=..."
        """,
    )

    parser.add_argument(
        "playlist_url",
        nargs="?",
        default="https://www.youtube.com/playlist?list=PLqjpMewv8gCRPg9qLAzABbwliuzKxL5Rc",
        help="YouTube playlist URL to extract (default: example playlist)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: data/playlist-extraction)",
    )

    args = parser.parse_args()

    # Create orchestrator and run
    orchestrator = ExtractionOrchestrator(
        playlist_url=args.playlist_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    success = orchestrator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
