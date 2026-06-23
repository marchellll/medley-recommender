use async_trait::async_trait;

use crate::domain::error::AppError;
use crate::domain::models::{Song, SongListQuery, SongPatch};
use crate::domain::pagination::CursorPage;

pub mod sqlite;

#[async_trait]
#[mockall::automock]
pub trait SongRepository: Send + Sync {
    async fn get(&self, song_id: &str) -> Result<Option<Song>, AppError>;
    async fn get_many(&self, song_ids: &[String]) -> Result<Vec<Song>, AppError>;
    async fn insert(&self, song: &Song) -> Result<(), AppError>;
    async fn update(&self, song_id: &str, patch: &SongPatch) -> Result<Song, AppError>;
    async fn delete(&self, song_id: &str) -> Result<(), AppError>;
    async fn exists(&self, song_id: &str) -> Result<bool, AppError>;
    async fn exists_by_youtube_url(&self, youtube_url: &str) -> Result<bool, AppError>;
    async fn get_by_youtube_url(&self, youtube_url: &str) -> Result<Option<Song>, AppError>;
    async fn list(&self, query: &SongListQuery) -> Result<CursorPage<Song>, AppError>;
}

#[derive(Debug, Clone)]
pub struct SongRow {
    pub song: Song,
    pub rank: Option<f64>,
}
