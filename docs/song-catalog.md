# Song Catalog

Medley stores worship songs in **SQLite** (`data/medley.db`). The committed database is the catalog source of truth; the Rust binary adds **FTS5** keyword search on top without removing existing rows.

## Required fields

| Field | Type | Rules |
|-------|------|-------|
| `title` | string | Non-empty |
| `youtube_url` | string | Valid YouTube watch or youtu.be URL |
| `lyrics` | string | Non-empty (required for embedding/search) |
| `bpm` | number | Must be > 0 and ≤ 300 |
| `key` | string | One of: `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `A#`, `B` |

## Song ID

Each song gets a **UUID v7** primary key at creation time (time-ordered, URL-safe). The YouTube URL is stored separately and must be unique across the catalog.

Legacy databases with `youtube/{video_id}` keys are rewritten to UUID v7 automatically on startup (row data and FTS preserved).

## Existing `medley.db`

The Rust migrations are **additive only**:

- `CREATE TABLE IF NOT EXISTS` — does not replace a legacy schema (extra columns such as `embedding_file_path` are kept)
- FTS5 virtual table + triggers are added if missing
- FTS **rebuild** backfills search index from existing rows; it does **not** delete catalog data

`medley serve` runs additive migrations on startup and backfills FTS when needed.

## Adding songs

### REST API

```bash
curl -X POST http://localhost:9876/api/songs \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Song Title",
    "youtube_url": "https://www.youtube.com/watch?v=...",
    "lyrics": "...",
    "bpm": 120,
    "key": "G"
  }'
```

Create/update/delete embed and sync the vector index inline (Voyage + Qdrant Edge). No batch pipeline.

### Web UI

Open `http://localhost:9876` — search, catalog browse (FTS + cursor pagination), add/edit/delete.

### MCP

Streamable HTTP at `http://localhost:9876/mcp` — tools: `add_song`, `update_song`, `delete_song`, `get_song`, `list_songs`, `search_songs`.

## Catalog list / keyword search

`GET /api/songs` supports filters and stable cursor pagination:

| Param | Purpose |
|-------|---------|
| `q` | FTS keyword search (title + lyrics) |
| `key`, `bpm_min`, `bpm_max` | Filters |
| `limit` | Page size (default 20, max 100) |
| `last_id` | Cursor for page 2+ |
| `last_rank` | Required with `last_id` when `q` is set (BM25 tiebreaker) |

## Validation

`SongService` rejects incomplete songs before embedding. Missing lyrics, invalid BPM, or missing key block indexing.
