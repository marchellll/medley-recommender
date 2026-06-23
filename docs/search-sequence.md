# Search Sequence Diagram

Semantic search via `POST /api/search` — Voyage query embedding, Qdrant Edge nearest-neighbor search, optional key/BPM payload filters, SQLite metadata join.

## Prerequisites

- Qdrant Edge shard at `EDGE_SHARD_PATH` (default `data/edge_shard`)
- `VOYAGE_API_KEY` set (each search embeds the query)
- Catalog rows in `DATABASE_PATH` (default `data/medley.db`)
- Vectors indexed in Edge (`data/edge_shard/`, committed for deployment alongside `medley.db`)

After catalog changes, rebuild and commit the shard:

```bash
cargo run -p medley-server -- reindex
```

```mermaid
sequenceDiagram
    participant User as User / Client
    participant API as POST /api/search
    participant SearchSvc as SearchService
    participant Voyage as VoyageClient
    participant VoyageAPI as Voyage API
    participant Edge as Qdrant Edge
    participant DB as SQLite (medley.db)

    User->>API: SearchQuery<br/>(query, limit, optional keys, bpm_min, bpm_max)

    API->>SearchSvc: search(query)
    SearchSvc->>Voyage: embed(query, input_type=query)
    Voyage->>VoyageAPI: POST /v1/embeddings
    Note over Voyage,VoyageAPI: voyage-4-large, 2048 dims, L2-normalized
    VoyageAPI-->>Voyage: Query vector
    Voyage-->>SearchSvc: Normalized query vector

    SearchSvc->>Edge: search(vector, filters)
    Note over Edge: Dot distance on normalized vectors<br/>payload filter: key, bpm
    Edge-->>SearchSvc: song_ids + scores

    SearchSvc->>DB: get_many(song_ids)
    DB-->>SearchSvc: title, bpm, key, youtube_url

    SearchSvc-->>API: SearchResult list
    API-->>User: { results, total }
```

## Similarity score

Edge returns a dot-product score on L2-normalized vectors (higher = more similar). Results preserve Edge rank order.
