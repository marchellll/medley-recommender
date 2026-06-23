pub mod config;
pub mod domain;
pub mod embed;
pub mod index;
pub mod repo;
pub mod service;

pub use config::Config;
pub use domain::error::AppError;
pub use service::search_service::SearchService;
pub use service::song_service::SongService;
