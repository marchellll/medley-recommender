# Playlist Extraction Pipeline

This directory contains scripts for extracting video URLs from YouTube playlists.

## Quick Start

The easiest way to extract URLs from a playlist is using the orchestrator script:

```bash
# Extract URLs from a playlist
uv run python scripts/playlist_extractor/orchestrate_extraction.py "PLAYLIST_URL"
```

The orchestrator will:
1. Download playlist metadata
2. Extract all video URLs
3. Save to `found_links.json`

**Output Location**: `data/playlist-extraction/found_links.json`

**Output Format**:
```json
[
  {"url": "https://www.youtube.com/watch?v=4AYM5h4ZuA0"},
  {"url": "https://www.youtube.com/watch?v=5BYM5h4ZuA1"},
  ...
]
```

## Manual Extraction

For more control, you can run the extraction script directly:

```bash
uv run python scripts/playlist_extractor/extract_playlist_raw.py [PLAYLIST_URL]
```

This extracts all video URLs from the playlist and saves them to `data/playlist-extraction/found_links.json`.

## Using Extracted Links

After extracting links, you can use them with the main pipeline:

```bash
# Copy found_links.json to links.json (or use --input flag)
cp data/playlist-extraction/found_links.json data/links.json

# Run the pipeline
uv run python scripts/embeddings/process_pipeline.py
```

The pipeline will:
1. Download audio from each YouTube URL
2. Extract metadata (BPM, key, duration)
3. Generate embeddings (requires lyrics to be added manually to database first)
4. Build search index

## Notes

- The extraction process only collects URLs - no guessing or validation needed
- All processing (title extraction, lyrics, metadata) happens during the main pipeline
- Duplicate URLs are handled automatically by the pipeline
