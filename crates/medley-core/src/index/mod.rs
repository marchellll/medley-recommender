use async_trait::async_trait;

use crate::domain::error::AppError;

pub mod qdrant_edge;

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub song_id: String,
    pub score: f32,
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
