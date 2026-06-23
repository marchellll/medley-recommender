use async_trait::async_trait;

use crate::domain::error::AppError;
use crate::domain::models::{Song, SongListQuery};

pub mod qdrant_edge;
pub mod tantivy_text;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub song_id: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct TextSearchHit {
    pub song_id: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct TextSearchPage {
    pub hits: Vec<TextSearchHit>,
    pub limit: u32,
    pub next_last_id: Option<String>,
    pub next_last_rank: Option<f64>,
    pub has_more: bool,
    pub total: i64,
}

#[async_trait]
#[mockall::automock]
pub trait VectorIndex: Send + Sync {
    async fn upsert(
        &self,
        song_id: &str,
        vector: Vec<f32>,
        key: &str,
        bpm: f64,
    ) -> Result<(), AppError>;

    async fn delete(&self, song_id: &str) -> Result<(), AppError>;

    async fn search(
        &self,
        vector: &[f32],
        limit: usize,
        keys: Option<&[String]>,
        bpm_min: Option<f64>,
        bpm_max: Option<f64>,
    ) -> Result<Vec<VectorHit>, AppError>;

    async fn flush(&self) -> Result<(), AppError>;
}

#[async_trait]
#[mockall::automock]
pub trait TextIndex: Send + Sync {
    async fn upsert(&self, song: &Song) -> Result<(), AppError>;
    async fn delete(&self, song_id: &str) -> Result<(), AppError>;
    async fn search(&self, query: &SongListQuery) -> Result<TextSearchPage, AppError>;
    async fn rebuild(&self, songs: &[Song]) -> Result<(), AppError>;
    async fn doc_count(&self) -> Result<u64, AppError>;
}
