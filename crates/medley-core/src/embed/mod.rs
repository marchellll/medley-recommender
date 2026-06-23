use async_trait::async_trait;

use crate::domain::error::AppError;

pub mod voyage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Query,
    Document,
}

#[async_trait]
#[mockall::automock]
pub trait EmbeddingProvider: Send + Sync {
    fn dimension(&self) -> usize;
    async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>, AppError>;
}
