use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use medley_core::config::Config;
use medley_core::embed::voyage::VoyageClient;
use medley_core::index::qdrant_edge::EdgeVectorIndex;
use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::service::search_service::SearchService;
use medley_core::service::song_service::SongService;

use crate::routes;
use crate::state::AppState;

struct CoreServices {
    repo: Arc<dyn medley_core::repo::SongRepository>,
    embedder: Arc<dyn medley_core::embed::EmbeddingProvider>,
    index: Arc<dyn medley_core::index::VectorIndex>,
}

async fn build_core_services(config: &Config) -> anyhow::Result<CoreServices> {
    let repo = SqliteSongRepository::connect(&config.database_path)
        .await
        .context("sqlite connect")?;
    repo.migrate().await.context("sqlite migrate")?;
    let repo: Arc<dyn medley_core::repo::SongRepository> = Arc::new(repo);

    let embedder: Arc<dyn medley_core::embed::EmbeddingProvider> = Arc::new(VoyageClient::new(
        config.voyage_api_key.clone(),
        config.embedding_model.clone(),
        config.embedding_dimension,
        config.voyage_base_url.clone(),
    ));

    let index: Arc<dyn medley_core::index::VectorIndex> = Arc::new(
        EdgeVectorIndex::open(&config.edge_shard_path, config.embedding_dimension)
            .context("edge index open")?,
    );

    Ok(CoreServices {
        repo,
        embedder,
        index,
    })
}

pub async fn build_song_service(config: &Config) -> anyhow::Result<Arc<SongService>> {
    let services = build_core_services(config).await?;
    Ok(Arc::new(SongService::new(
        services.repo,
        services.embedder,
        services.index,
    )))
}

pub async fn build_state(config: &Config) -> anyhow::Result<AppState> {
    let services = build_core_services(config).await?;
    let songs = Arc::new(SongService::new(
        services.repo.clone(),
        services.embedder.clone(),
        services.index.clone(),
    ));
    let search = Arc::new(SearchService::new(
        services.repo,
        services.embedder,
        services.index,
    ));

    Ok(AppState::new(songs, search))
}

pub fn test_router(state: AppState) -> Router {
    Router::new()
        .merge(routes::api_router(state))
        .route("/health", axum::routing::get(routes::health))
}

pub fn test_config(
    database_path: std::path::PathBuf,
    edge_shard_path: std::path::PathBuf,
    voyage_base_url: impl Into<String>,
) -> Config {
    Config {
        database_path,
        edge_shard_path,
        bind_addr: "127.0.0.1:0".into(),
        voyage_api_key: "test-key".into(),
        embedding_model: "voyage-4-large".into(),
        embedding_dimension: 8,
        voyage_base_url: voyage_base_url.into(),
    }
}
