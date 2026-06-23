use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("validation: {0}")]
    Validation(String),
    #[error("invalid cursor: {0}")]
    InvalidCursor(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("index unavailable: {0}")]
    IndexUnavailable(String),
    #[error("embedding: {0}")]
    Embedding(String),
    #[error("database: {0}")]
    Database(#[from] sqlx::Error),
    #[error("internal: {0}")]
    Internal(String),
}

impl AppError {
    pub fn status_code(&self) -> u16 {
        match self {
            Self::NotFound(_) => 404,
            Self::Validation(_) | Self::InvalidCursor(_) => 400,
            Self::Conflict(_) => 409,
            Self::IndexUnavailable(_) => 503,
            Self::Embedding(_) => 502,
            Self::Database(_) | Self::Internal(_) => 500,
        }
    }
}
