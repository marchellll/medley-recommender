use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use medley_core::config::Config;
use medley_core::embed::voyage::VoyageClient;
use medley_core::index::qdrant_edge::EdgeVectorIndex;
use medley_core::index::tantivy_text::TantivyTextIndex;
use medley_core::index::{TextIndex, VectorIndex};
use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::repo::submission_sqlite::SubmissionRepository;
use medley_core::service::search_service::SearchService;
use medley_core::service::song_service::SongService;
use medley_core::service::submission_service::SubmissionService;

use crate::auth::{AdminAuth, McpAuth};
use crate::rate_limit::RateLimiter;
use crate::routes;
use crate::state::AppState;

struct CoreServices {
    repo: Arc<dyn medley_core::repo::SongRepository>,
    embedder: Arc<dyn medley_core::embed::EmbeddingProvider>,
    vector_index: Arc<dyn VectorIndex>,
    text_index: Arc<dyn TextIndex>,
    submission_repo: SubmissionRepository,
}

async fn build_core_services(config: &Config) -> anyhow::Result<CoreServices> {
    let sqlite = SqliteSongRepository::connect(&config.database_path)
        .await
        .context("sqlite connect")?;
    sqlite.migrate().await.context("sqlite migrate")?;
    let submission_repo = SubmissionRepository::new(sqlite.pool().clone());
    let repo: Arc<dyn medley_core::repo::SongRepository> = Arc::new(sqlite);

    let embedder: Arc<dyn medley_core::embed::EmbeddingProvider> = Arc::new(VoyageClient::new(
        config.voyage_api_key.clone(),
        config.embedding_model.clone(),
        config.embedding_dimension,
        config.voyage_base_url.clone(),
    ));

    let vector_index: Arc<dyn VectorIndex> = Arc::new(
        EdgeVectorIndex::open(&config.edge_shard_path, config.embedding_dimension)
            .context("edge index open")?,
    );

    let text_index: Arc<dyn TextIndex> =
        Arc::new(TantivyTextIndex::open(&config.text_index_path).context("text index open")?);

    Ok(CoreServices {
        repo,
        embedder,
        vector_index,
        text_index,
        submission_repo,
    })
}

pub async fn build_song_service(config: &Config) -> anyhow::Result<Arc<SongService>> {
    let services = build_core_services(config).await?;
    let songs = Arc::new(SongService::new(
        services.repo,
        services.embedder,
        services.vector_index,
        services.text_index,
    ));
    songs
        .ensure_text_index_synced()
        .await
        .context("text index sync")?;
    Ok(songs)
}

pub async fn build_state(config: &Config) -> anyhow::Result<AppState> {
    if config.admin_token.is_empty() {
        tracing::warn!("ADMIN_TOKEN is not set; HTTP admin login and mutations disabled");
    }
    if config.api_token.is_empty() {
        tracing::warn!("API_TOKEN is not set; MCP mutations disabled");
    }

    let services = build_core_services(config).await?;
    let songs = Arc::new(SongService::new(
        services.repo.clone(),
        services.embedder.clone(),
        services.vector_index.clone(),
        services.text_index.clone(),
    ));
    songs
        .ensure_text_index_synced()
        .await
        .context("text index sync")?;
    let submissions = Arc::new(SubmissionService::new(
        services.submission_repo,
        services.repo.clone(),
    ));
    let search = Arc::new(SearchService::new(
        services.repo,
        services.embedder,
        services.vector_index,
    ));

    Ok(AppState::new(
        songs,
        submissions,
        search,
        Arc::new(AdminAuth::new(config.admin_token.clone())),
        Arc::new(McpAuth::new(config.api_token.clone())),
        Arc::new(RateLimiter::default()),
        Arc::new(RateLimiter::per_minute(30)),
        Arc::new(RateLimiter::per_minute(5)),
        Arc::new(RateLimiter::per_minute(200)),
    ))
}

pub fn test_router(state: AppState) -> Router {
    routes::api_router(state).route("/health", axum::routing::get(routes::health))
}

pub fn test_ui_router(state: AppState) -> Router {
    routes::ui_router(state)
}

pub fn admin_bearer(state: &AppState) -> String {
    format!(
        "Bearer {}",
        state.admin_auth.issue_jwt().expect("admin jwt")
    )
}

pub fn admin_cookie(state: &AppState) -> String {
    format!(
        "medley_token={}",
        state.admin_auth.issue_jwt().expect("admin jwt")
    )
}

pub fn test_config(
    database_path: std::path::PathBuf,
    edge_shard_path: std::path::PathBuf,
    voyage_base_url: impl Into<String>,
) -> Config {
    let text_index_path = database_path
        .parent()
        .map(|dir| dir.join("text_index"))
        .unwrap_or_else(|| std::path::PathBuf::from("data/text_index"));
    Config {
        database_path,
        edge_shard_path,
        text_index_path,
        bind_addr: "127.0.0.1:0".into(),
        voyage_api_key: "test-key".into(),
        embedding_model: "voyage-4-large".into(),
        embedding_dimension: 8,
        voyage_base_url: voyage_base_url.into(),
        admin_token: "test-admin-token".into(),
        api_token: "test-api-token".into(),
    }
}
