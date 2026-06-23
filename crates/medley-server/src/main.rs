use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use clap::{Parser, Subcommand};
use medley_core::config::Config;
use medley_server::app::build_state;
use medley_server::mcp::MedleyMcp;
use medley_server::reindex;
use medley_server::routes;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService,
    session::local::LocalSessionManager,
};
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "medley", about = "Medley song catalog and semantic search")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start HTTP API, UI, and MCP on one port (default)
    Serve,
    /// Wipe vector index and re-embed all catalog songs from the database
    Reindex,
}

fn init_logging() {
    let log_filter = std::env::var("LOG_LEVEL").ok().map(|level| {
        format!(
            "{level},qdrant_edge=error,medley_core::service=debug,medley_core::repo=debug,medley_server=info,tower_http=info"
        )
    });
    let env_filter = match log_filter {
        Some(filter) => EnvFilter::new(filter),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,qdrant_edge=error,medley_core::service=debug,medley_core::repo=debug,medley_server=info,tower_http=info",
            )
        }),
    };
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => serve().await,
        Commands::Reindex => reindex::run(&Config::from_env()).await,
    }
}

async fn serve() -> anyhow::Result<()> {
    let config = Config::from_env();
    let app_state = build_state(&config).await?;

    let ct = CancellationToken::new();
    let mcp_state = app_state.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(MedleyMcp::new(mcp_state.clone())),
        Arc::new(LocalSessionManager::default()),
        {
            let mut cfg = StreamableHttpServerConfig::default();
            cfg.stateful_mode = true;
            cfg.cancellation_token = ct.child_token();
            cfg
        },
    );

    let app = Router::new()
        .merge(routes::api_router(app_state.clone()))
        .merge(routes::ui_router(app_state))
        .route("/health", axum::routing::get(routes::health))
        .nest_service("/mcp", mcp_service)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&config.bind_addr)
        .await
        .with_context(|| format!("failed to bind {}", config.bind_addr))?;
    tracing::info!("medley listening on http://{}", config.bind_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.ok();
            ct.cancel();
        })
        .await?;

    Ok(())
}
