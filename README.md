# Medley Recommender

A lightweight local system for recommending worship/praise songs based on semantic similarity and musical metadata (BPM, key, etc.).

Medley Recommender helps you discover worship and praise songs by searching through lyrics using semantic similarity. It automatically downloads audio from YouTube, extracts musical metadata like BPM and key, generates embeddings from song lyrics, and provides fast similarity search through a local ANN index. Perfect for worship leaders, musicians, and anyone looking to find songs with similar themes, arrangements, or musical characteristics.

## Project Goal

Build a system that:
- Ingests worship/praise songs (YouTube URLs only)
- Downloads audio from YouTube and converts to MP3 format (192kbps)
- Extracts BPM & metadata (key, duration, etc.)
- Accepts lyrics (manually added to database)
- Creates text embeddings from lyrics
- Stores vectors in a local ANN index (Chroma) with native filtering support
- Stores metadata in SQLite database
- Exposes a query API (filter by BPM + semantic similarity)

## Features

- 🎵 **Semantic Search**: Find songs by lyrics similarity using BGE embeddings
- 🎚️ **Metadata Filtering**: Native filtering by BPM, key, and other musical attributes during search
- 📥 **YouTube Integration**: Automatic audio download from YouTube
- 🧠 **Local Processing**: All embeddings and indexing done locally
- 🚀 **Fast API**: Async FastAPI server with MCP support
- 🎨 **Beautiful CLI**: Premium terminal UI with progress tracking
- 🐳 **Docker Ready**: Single command deployment

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medley-recommender
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables (optional - defaults are provided):
```bash
cp .env.example .env
# Edit .env as needed, or use the default values
```

4. Prepare your songs data:
```bash
# Edit data/links.json with YouTube URLs
# Format: [{"url": "https://www.youtube.com/watch?v=..."}, ...]
```

5. Run the pipeline:
```bash
uv run python scripts/embeddings/process_pipeline.py
```

### Docker Deployment

```bash
docker compose up --build
```

This will start both the API server and Streamlit UI. Ports are configured in `.env` (default: API on 9876, Streamlit on 9877).

## Usage

### Master Pipeline Script

Process all songs through the complete pipeline:

```bash
# Process all songs
uv run python scripts/embeddings/process_pipeline.py

# Use custom input file
uv run python scripts/embeddings/process_pipeline.py --input path/to/links.json

# Force reprocessing
uv run python scripts/embeddings/process_pipeline.py --force

# Skip specific steps
uv run python scripts/embeddings/process_pipeline.py --skip-download --skip-lyrics --skip-metadata

# Verbose output
uv run python scripts/embeddings/process_pipeline.py --verbose

# JSON output for scripting
uv run python scripts/embeddings/process_pipeline.py --json
```

### API Server

Start the API server (uses port from `.env`, default: 9876):

```bash
# Recommended: Use the startup script (reads from .env)
uv run python scripts/servers/run_api.py

# Or manually specify port
uv run uvicorn api.main:app --host 0.0.0.0 --port 9876 --reload
```

#### Endpoints

- `POST /api/search`: Search for songs
  ```bash
  curl -X POST http://localhost:9876/api/search \
    -H "Content-Type: application/json" \
    -d '{
      "query": "amazing grace",
      "bpm_min": 60,
      "bpm_max": 120,
      "limit": 10
    }'
  ```

- `POST /api/add_song`: Add a new song
  ```bash
  curl -X POST http://localhost:9876/api/add_song \
    -H "Content-Type: application/json" \
    -d '{
      "title": "Song Title",
      "youtube_url": "https://youtube.com/watch?v=...",
      "lyrics": "Song lyrics here..."
    }'
  ```

### Streamlit UI

Start the Streamlit UI (uses port from `.env`, default: 9877):

```bash
# Recommended: Use the startup script (reads from .env)
uv run python scripts/servers/run_streamlit.py

# Or manually specify port
uv run streamlit run ui/app.py --server.port 9877
```

## Project Structure

```
medley-recommender/
├── data/                    # Data files
│   ├── audio/              # Downloaded audio files (MP3 format)
│   ├── links.json          # Input YouTube URLs
│   ├── songs_embeddings/        # Embedding files
│   ├── index/              # ANN index files
│   └── medley.db           # SQLite database
├── scripts/                 # Utility scripts
│   ├── playlist_extractor/ # Playlist extraction scripts
│   ├── embeddings/         # Embedding and pipeline scripts
│   └── servers/            # Server startup scripts
├── api/                     # API server
├── src/                     # Core library
│   ├── audio/              # Audio processing
│   ├── embeddings/         # Embedding generation
│   ├── index/              # ANN index
│   ├── database/           # Database models
│   └── utils/              # Utilities
├── ui/                      # Streamlit UI
└── tests/                   # Test suite
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Master Pipeline Script

The master pipeline script (`scripts/embeddings/process_pipeline.py`) orchestrates the entire data processing workflow:

1. **Download**: Downloads audio from YouTube URLs (extracts title automatically)
2. **Metadata**: Extracts BPM, key, duration from audio files
3. **Embeddings**: Generates semantic embeddings from lyrics (lyrics must be added manually to database)
4. **Index**: Builds ANN index for fast similarity search

### Features

- **Idempotent**: Same input produces same output
- **Progress Tracking**: Beautiful CLI with progress bars and status updates
- **Error Handling**: Continues processing even if individual songs fail
- **Resumable**: Skips already processed steps

### Example Output

```sh
uv run python scripts/embeddings/process_pipeline.py
🎵 Medley Recommender Pipeline
Processing worship songs through the complete pipeline

📥 Downloading audio... [████████████] 100%
📝 Extracting lyrics... [████████████] 100%
🔍 Extracting metadata... [████████████] 100%
🧠 Generating embeddings... [████████████] 100%
📊 Building index... [████████████] 100%
```


## License

[Add your license here]

