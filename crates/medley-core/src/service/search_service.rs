use std::sync::Arc;

use crate::domain::error::AppError;
use crate::domain::models::{SearchQuery, SearchResult};
use crate::domain::pagination::clamp_limit;
use crate::embed::{EmbeddingProvider, InputType};
use crate::index::VectorIndex;
use crate::repo::SongRepository;

pub struct SearchService {
    repo: Arc<dyn SongRepository>,
    embedder: Arc<dyn EmbeddingProvider>,
    index: Arc<dyn VectorIndex>,
}

impl SearchService {
    pub fn new(
        repo: Arc<dyn SongRepository>,
        embedder: Arc<dyn EmbeddingProvider>,
        index: Arc<dyn VectorIndex>,
    ) -> Self {
        Self {
            repo,
            embedder,
            index,
        }
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>, AppError> {
        tracing::info!(
            query = %query.query,
            bpm_min = ?query.bpm_min,
            bpm_max = ?query.bpm_max,
            limit = ?query.limit,
            keys = ?query.keys,
            "search_service.search"
        );
        let q = query.query.trim();
        if q.is_empty() {
            tracing::warn!("search_service.search empty query");
            return Err(AppError::Validation("query is required".into()));
        }

        let limit = clamp_limit(query.limit, 10, 50);
        let vectors = self
            .embedder
            .embed(&[q.to_string()], InputType::Query)
            .await?;
        let vector = vectors
            .into_iter()
            .next()
            .ok_or_else(|| AppError::Embedding("empty embedding response".into()))?;

        let keys_ref = query.keys.as_deref();
        let hits = self
            .index
            .search(
                &vector,
                limit as usize,
                keys_ref,
                query.bpm_min,
                query.bpm_max,
            )
            .await?;

        if hits.is_empty() {
            tracing::info!(query = %q, "search_service.search ok (no hits)");
            return Ok(vec![]);
        }

        let song_ids: Vec<String> = hits.iter().map(|h| h.song_id.clone()).collect();
        let songs = self.repo.get_many(&song_ids).await?;
        let by_id: std::collections::HashMap<_, _> =
            songs.into_iter().map(|s| (s.song_id.clone(), s)).collect();

        let mut results = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some(song) = by_id.get(&hit.song_id) else {
                tracing::warn!(song_id = %hit.song_id, "search_service.search missing catalog row");
                continue;
            };
            results.push(SearchResult {
                song_id: song.song_id.clone(),
                title: song.title.clone(),
                bpm: song.bpm,
                key: song.key.clone(),
                similarity_score: hit.score as f64,
                youtube_url: song.youtube_url.clone(),
            });
        }

        tracing::info!(query = %q, returned = results.len(), "search_service.search ok");
        Ok(results)
    }
}
