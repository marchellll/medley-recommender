# Multi-stage build for medley (Rust single binary)
FROM rust:1.88-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release -p medley-server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/medley /usr/local/bin/medley

ENV DATABASE_PATH=data/medley.db \
    EDGE_SHARD_PATH=data/edge_shard \
    BIND_HOST=0.0.0.0 \
    BIND_PORT=9876

EXPOSE 9876

CMD ["medley"]
