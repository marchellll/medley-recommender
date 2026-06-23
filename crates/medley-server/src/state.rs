use std::sync::Arc;

use medley_core::service::search_service::SearchService;
use medley_core::service::song_service::SongService;

#[derive(Clone)]
pub struct AppState {
    pub songs: Arc<SongService>,
    pub search: Arc<SearchService>,
}

impl AppState {
    pub fn new(songs: Arc<SongService>, search: Arc<SearchService>) -> Self {
        Self { songs, search }
    }
}
