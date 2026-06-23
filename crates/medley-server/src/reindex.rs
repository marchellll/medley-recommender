use anyhow::Context;
use medley_core::config::Config;

pub async fn run(config: &Config) -> anyhow::Result<()> {
    if config.voyage_api_key.is_empty() {
        anyhow::bail!("VOYAGE_API_KEY is required to re-embed catalog songs");
    }

    if config.edge_shard_path.exists() {
        tracing::info!(
            path = %config.edge_shard_path.display(),
            "removing existing edge shard"
        );
        std::fs::remove_dir_all(&config.edge_shard_path)
            .with_context(|| format!("remove {}", config.edge_shard_path.display()))?;
    }

    let songs = crate::app::build_song_service(config)
        .await
        .context("build song service")?;

    tracing::info!(
        database = %config.database_path.display(),
        edge_shard = %config.edge_shard_path.display(),
        "reindex started"
    );

    let report = songs.reindex_all().await.context("reindex catalog")?;

    println!("reindex complete");
    println!("  database path:   {}", config.database_path.display());
    println!("  edge shard path: {}", config.edge_shard_path.display());
    println!("  catalog songs:   {}", report.total_songs);
    println!("  vectors indexed: {}", report.indexed);
    println!("  skipped:         {}", report.skipped);
    println!();
    println!("  tip: commit data/edge_shard after reindex if deploying from git");

    Ok(())
}
