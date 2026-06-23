# Medley Recommender

Worship song recommender: **Voyage AI** lyric embeddings (`voyage-4-large`, 2048-dim), manual BPM/key metadata, **SQLite FTS5** catalog search, **Qdrant Edge** semantic search.

Single Rust binary — REST, web UI, and MCP on one port.

See [docs/song-catalog.md](docs/song-catalog.md) for catalog format and API details.

## Features

- Semantic search (Voyage query embeddings + Qdrant Edge)
- FTS5 keyword catalog browse with stable cursor pagination
- Song CRUD via REST, web UI, and MCP (`/mcp`)
- Existing `data/medley.db` preserved on upgrade (additive migrations)
- Docker-ready

## Prerequisites

- Rust 1.76+ (stable)
- `VOYAGE_API_KEY` from [Voyage AI](https://dash.voyageai.com/) (required for search and new writes)

## Quick start

```bash
git clone <repository-url>
cd medley-recommender
cp .env.example .env
# Set VOYAGE_API_KEY in .env

# Run server (API + UI + MCP on :9876)
cargo run -p medley-server

# Rebuild vector index from catalog (wipes edge_shard, re-embeds via Voyage)
cargo run -p medley-server -- reindex
```

Open `http://localhost:9876` · MCP at `http://localhost:9876/mcp`

## Data layout

```
data/
  medley.db           # SQLite catalog (committed)
  edge_shard/         # Qdrant Edge vectors (committed; semantic search works out of the box)
```

Both `medley.db` and `edge_shard/` are checked in so a fresh clone is deployable without calling Voyage for every song. After adding or bulk-updating songs, run `reindex` and commit the updated `edge_shard/`.

```bash
cargo run -p medley-server -- reindex
```

This deletes `EDGE_SHARD_PATH` and re-embeds every ready song via Voyage.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/api/songs` | Catalog list / FTS (`q`, cursor pagination) |
| POST | `/api/songs` | Create song (embed + index inline) |
| GET | `/api/songs/:id` | Get song |
| PATCH | `/api/songs/:id` | Update song |
| DELETE | `/api/songs/:id` | Delete song |
| POST | `/api/search` | Semantic search |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `data/medley.db` | SQLite catalog |
| `EDGE_SHARD_PATH` | `data/edge_shard` | Qdrant Edge shard directory |
| `BIND_HOST` / `API_HOST` | `0.0.0.0` | Listen host |
| `BIND_PORT` / `API_PORT` | `9876` | Listen port |
| `VOYAGE_API_KEY` | — | Voyage API key |
| `EMBEDDING_MODEL` | `voyage-4-large` | Embedding model |
| `EMBEDDING_OUTPUT_DIMENSION` | `2048` | Vector size |
| `LOG_LEVEL` / `RUST_LOG` | `info` | Log filter (see `main.rs` defaults) |

## Development

```bash
cargo build --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p medley-core --test legacy_db -- --ignored  # uses committed data/medley.db
```

## Docker

```bash
docker compose up --build
```

## Project structure

```
medley-recommender/
├── crates/
│   ├── medley-core/     # domain, repo, services, Voyage, Qdrant Edge
│   └── medley-server/   # axum REST, askama UI, MCP
├── data/
├── docs/
└── Cargo.toml
```

## License

[Add your license here]
