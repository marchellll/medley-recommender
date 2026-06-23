# Catalog Write Sequence

Songs are written through **REST**, **UI**, or **MCP**; `SongService` keeps SQLite, Tantivy, Voyage, and Qdrant Edge in sync.

## Storage

| Asset | Location | Notes |
|-------|----------|--------|
| Catalog | `data/medley.db` | SQLite rows only |
| Keywords | `data/text_index/` | Tantivy index (synced on writes; rebuilt on startup if out of sync) |
| Vectors | `data/edge_shard/` | Qdrant Edge shard (created/updated on each write) |

## Write path (create / update / delete)

```mermaid
sequenceDiagram
    participant Client as REST / UI / MCP
    participant SongSvc as SongService
    participant Validation as domain/validation
    participant Repo as SqliteSongRepository
    participant Voyage as VoyageClient
    participant Edge as Qdrant Edge
    participant Text as Tantivy
    participant DB as SQLite

    Client->>SongSvc: create / update / delete

    SongSvc->>Validation: validate fields
    alt Invalid
        Validation-->>Client: 400
    end

    alt create
        SongSvc->>Repo: insert
        Repo->>DB: INSERT songs
        SongSvc->>Text: upsert(song)
        SongSvc->>Voyage: embed(lyrics, document)
        SongSvc->>Edge: upsert(song_id, vector, key, bpm)
        SongSvc->>Edge: flush
    else update (title/lyrics/bpm/key changed)
        SongSvc->>Repo: update
        Repo->>DB: UPDATE songs
        SongSvc->>Text: upsert(song)
        opt lyrics/bpm/key changed
            SongSvc->>Voyage: re-embed
            SongSvc->>Edge: upsert
            SongSvc->>Edge: flush
        end
    else delete
        SongSvc->>Repo: delete
        Repo->>DB: DELETE songs
        SongSvc->>Text: delete(song_id)
        SongSvc->>Edge: delete(song_id)
        SongSvc->>Edge: flush
    end

    SongSvc-->>Client: success
```

## Running the server

```bash
cargo run -p medley-server
```

Single process on `:9876` — REST (`/api/*`), web UI, MCP (`/mcp`), health (`/health`).
