#!/usr/bin/env python3
"""
Step 1: Extract raw playlist data using yt-dlp and save as JSON.

This script extracts all video information from a YouTube playlist and saves
the raw data to data/playlist-extraction/playlist_raw.json for further processing.
"""

import json
import re
import sys
from pathlib import Path

import yt_dlp
from rich.console import Console

console = Console()


class YTDLLogger:
    """Custom logger for yt-dlp to show verbose progress."""

    def __init__(self):
        self.video_count = 0
        self.sabr_warnings = set()  # Track SABR warnings per video

    def debug(self, msg: str) -> None:
        """Handle debug messages."""
        # Show messages about extracting/processing videos
        msg_lower = msg.lower()
        if any(keyword in msg_lower for keyword in [
            "extracting", "processing", "video", "playlist",
            "downloading webpage", "youtube", "fetching"
        ]):
            # Count videos being processed
            if "video" in msg_lower and "extracting" in msg_lower:
                self.video_count += 1
            console.print(f"[dim]  {msg}[/dim]")

    def warning(self, msg: str) -> None:
        """Handle warning messages."""
        # Check for SABR streaming warnings
        if "SABR streaming" in msg or "web_safari" in msg:
            # Extract video ID if present
            video_match = re.search(r"([A-Za-z0-9_-]{11}):", msg)
            if video_match:
                video_id = video_match.group(1)
                if video_id not in self.sabr_warnings:
                    self.sabr_warnings.add(video_id)
                    console.print(
                        f"[yellow]⚠ SABR streaming warning for video {video_id}[/yellow]"
                    )
                    console.print(
                        f"[dim]    (Some formats may be unavailable due to YouTube's SABR streaming)[/dim]"
                    )
                    console.print(
                        f"[dim]    This is a known YouTube limitation, not an error[/dim]"
                    )
            else:
                console.print(f"[yellow]⚠ {msg}[/yellow]")
        else:
            # Show other warnings normally
            console.print(f"[yellow]⚠ {msg}[/yellow]")

    def error(self, msg: str) -> None:
        """Handle error messages."""
        console.print(f"[red]✗ {msg}[/red]")


def extract_playlist_raw(playlist_url: str, output_path: Path) -> None:
    """Extract all video URLs from a YouTube playlist and save to found_links.json."""
    # Custom logger for verbose output
    yt_dlp_logger = YTDLLogger()

    ydl_opts = {
        "quiet": False,  # Don't suppress output
        "no_warnings": False,  # Show warnings
        "extract_flat": False,  # Get full metadata
        "noplaylist": False,  # Extract playlist
        "logger": yt_dlp_logger,  # Use custom logger
        "verbose": True,  # Enable verbose output
        "extractor_args": {
            "youtube": {
                # Use android client to avoid SABR streaming issues with web_safari
                # Falls back to other clients if android fails
                "player_client": ["android", "ios", "web"],
            }
        },
    }

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        console.print(f"[cyan]Extracting playlist information...[/cyan]")
        console.print(f"[dim]This may take a while for large playlists (fetching metadata for each video)...[/dim]")
        try:
            # Get playlist info
            # This is the slow part - yt-dlp fetches metadata for each video
            console.print(f"[dim]  Fetching playlist structure...[/dim]")
            info = ydl.extract_info(playlist_url, download=False, process=True)

            console.print(f"[green]✓ Playlist information extracted[/green]")

            if "entries" not in info:
                raise ValueError("No entries found in playlist")

            # Process each video
            entries = info["entries"]
            total = len(entries) if entries else 0

            console.print(f"[cyan]Processing {total} videos...[/cyan]")

            for i, entry in enumerate(entries, 1):
                if entry is None:
                    continue

                # Extract only URL
                video_id = entry.get("id", "")
                if video_id:
                    video_data = {
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    }
                    videos.append(video_data)

                # Show progress every 10 videos or for the last one
                if i % 10 == 0 or i == total:
                    console.print(f"[dim]  Processed {i}/{total} videos...[/dim]")

        except Exception as e:
            console.print(f"[red]Error extracting playlist: {e}[/red]")
            raise

    # Save raw data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
        f.write("\n")

    console.print(f"[green]✓ Extracted {len(videos)} video URLs to {output_path}[/green]")


def main(playlist_url: str | None = None) -> None:
    """Main function to extract raw playlist data."""
    # Get playlist URL from command line or use default
    if playlist_url is None:
        if len(sys.argv) > 1:
            playlist_url = sys.argv[1]
        else:
            playlist_url = "https://www.youtube.com/playlist?list=PLqjpMewv8gCRPg9qLAzABbwliuzKxL5Rc"

    # Paths
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "playlist-extraction" / "found_links.json"

    console.print(f"[bold]Extracting raw playlist data:[/bold] {playlist_url}")

    try:
        extract_playlist_raw(playlist_url, output_path)
    except Exception as e:
        console.print(f"[red]Failed to extract playlist: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

