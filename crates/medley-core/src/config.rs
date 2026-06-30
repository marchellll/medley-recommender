use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub database_path: PathBuf,
    pub edge_shard_path: PathBuf,
    pub text_index_path: PathBuf,
    pub bind_addr: String,
    pub voyage_api_key: String,
    pub embedding_model: String,
    pub embedding_dimension: usize,
    /// Override for tests (`VOYAGE_BASE_URL` in production tooling).
    pub voyage_base_url: String,
    pub admin_token: String,
    pub api_token: String,
}

impl Config {
    pub fn load_env() {
        let _ = dotenvy::dotenv();
    }

    pub fn from_env() -> Self {
        Self::load_env();
        let database_path =
            std::env::var("DATABASE_PATH").unwrap_or_else(|_| "data/medley.db".into());
        let edge_shard_path =
            std::env::var("EDGE_SHARD_PATH").unwrap_or_else(|_| "data/edge_shard".into());
        let text_index_path =
            std::env::var("TEXT_INDEX_PATH").unwrap_or_else(|_| "data/text_index".into());
        let bind_host = std::env::var("BIND_HOST")
            .or_else(|_| std::env::var("API_HOST"))
            .unwrap_or_else(|_| "0.0.0.0".into());
        let bind_port = std::env::var("BIND_PORT")
            .or_else(|_| std::env::var("API_PORT"))
            .unwrap_or_else(|_| "9876".into());
        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .expect("VOYAGE_API_KEY must be set in env variablle");
        let embedding_model =
            std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "voyage-4-large".into());
        let embedding_dimension = std::env::var("EMBEDDING_OUTPUT_DIMENSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let voyage_base_url =
            std::env::var("VOYAGE_BASE_URL").unwrap_or_else(|_| "https://api.voyageai.com".into());
        let admin_token = std::env::var("ADMIN_TOKEN").unwrap_or_else(|_| "admin".into());
        let api_token = std::env::var("API_TOKEN").unwrap_or_else(|_| "apitoken".into());

        Self {
            database_path: PathBuf::from(database_path),
            edge_shard_path: PathBuf::from(edge_shard_path),
            text_index_path: PathBuf::from(text_index_path),
            bind_addr: format!("{bind_host}:{bind_port}"),
            voyage_api_key,
            embedding_model,
            embedding_dimension,
            voyage_base_url,
            admin_token,
            api_token,
        }
    }

    pub fn data_dir(&self) -> &Path {
        self.database_path.parent().unwrap_or(Path::new("data"))
    }
}
