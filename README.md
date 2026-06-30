# Medley Recommender

Worship song recommender. **Voyage AI** lyric embeddings (2048-dim), **Tantivy** keyword search, **Qdrant Edge** semantic search. Single Rust binary — REST, web UI, MCP on one port.

## Quick start

```bash
# set VOYAGE_API_KEY
cp .env.example .env
# or at least this
export VOYAGE_API_KEY=<insert your own>


# run the server
cargo run -p medley-server
```

Open `http://localhost:9876` · MCP at `http://localhost:9876/mcp`

```bash
docker pull --platform linux/amd64 ghcr.io/marchellll/medley-recommender:latest
docker run --platform linux/amd64 -e VOYAGE_API_KEY -p 9876:9876 ghcr.io/marchellll/medley-recommender:latest
```
or
```bash
docker compose up --build   # build from source, containerized
```

Voyage key required at runtime for query embedding. Pre-committed `data/edge_shard/` skips the batch re-index step.

## Re-index

After adding or updating songs, rebuild vectors from catalog:

```bash
cargo run -p medley-server -- reindex   # wipes edge_shard, re-embeds via Voyage
```

Commit updated `data/edge_shard/` so others skip the call.

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/api/songs` | List / keyword search (`q`, cursor) |
| POST | `/api/songs` | Create song (embed + index) |
| GET | `/api/songs/:id` | Get song |
| PATCH | `/api/songs/:id` | Update song |
| DELETE | `/api/songs/:id` | Delete song |
| POST | `/api/search` | Semantic search |
| POST | `/api/auth/login` | Exchange `ADMIN_TOKEN` for JWT |

Unauthenticated: **10 req/min per IP**. Admin JWT bypasses.

## Auth

| Token | Env var | Use |
|-------|---------|-----|
| Admin | `ADMIN_TOKEN` | HTTP login → JWT cookie/Bearer. Unlocks REST/UI CRUD. |
| API | `API_TOKEN` | MCP Bearer on `/mcp`. Unlocks mutation tools. |

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `data/medley.db` | SQLite catalog |
| `EDGE_SHARD_PATH` | `data/edge_shard` | Qdrant Edge shard |
| `TEXT_INDEX_PATH` | `data/text_index` | Tantivy keyword index |
| `BIND_HOST` / `API_HOST` | `0.0.0.0` | Listen host |
| `BIND_PORT` / `API_PORT` | `9876` | Listen port |
| `VOYAGE_API_KEY` | — | Voyage API key |
| `EMBEDDING_MODEL` | `voyage-4-large` | Embedding model |
| `EMBEDDING_OUTPUT_DIMENSION` | `2048` | Vector size |
| `ADMIN_TOKEN` | — | HTTP admin login |
| `API_TOKEN` | — | MCP Bearer |
| `LOG_LEVEL` / `RUST_LOG` | `info` | Log filter |

## Dev

```bash
cargo build --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p medley-core --test legacy_db -- --ignored  # uses committed data/medley.db
```

## Layout

```
medley-recommender/
├── crates/
│   ├── medley-core/     # domain, repo, services, Voyage, Qdrant Edge
│   └── medley-server/   # axum REST, askama UI, MCP
├── data/
│   ├── medley.db        # SQLite catalog (committed)
│   ├── edge_shard/      # Qdrant Edge vectors (committed)
│   └── text_index/      # Tantivy index (rebuilt on startup)
├── docs/
└── Cargo.toml
```

See [docs/song-catalog.md](docs/song-catalog.md) for catalog format and full API details.

## License

MIT — see [LICENSE](LICENSE)
