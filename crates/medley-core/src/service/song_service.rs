use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;

use crate::domain::error::AppError;
use crate::domain::id::new_song_id;
use crate::domain::lyrics::clean_lyrics;
use crate::domain::models::{NewSong, Song, SongListQuery, SongPatch};
use crate::domain::pagination::CursorPage;
use crate::domain::validation::{song_readiness_issues, validate_new_song, validate_patch};
use crate::domain::youtube::normalize_youtube_url;
use crate::embed::{EmbeddingProvider, InputType};
use crate::index::{TextIndex, VectorIndex};
use crate::repo::SongRepository;

#[derive(Debug, Clone, Copy)]
pub struct ReindexReport {
    pub total_songs: i64,
    pub indexed: u64,
    pub skipped: u64,
}

pub struct SongService {
    repo: Arc<dyn SongRepository>,
    embedder: Arc<dyn EmbeddingProvider>,
    vector_index: Arc<dyn VectorIndex>,
    text_index: Arc<dyn TextIndex>,
}

impl SongService {
    pub fn new(
        repo: Arc<dyn SongRepository>,
        embedder: Arc<dyn EmbeddingProvider>,
        vector_index: Arc<dyn VectorIndex>,
        text_index: Arc<dyn TextIndex>,
    ) -> Self {
        Self {
            repo,
            embedder,
            vector_index,
            text_index,
        }
    }

    pub async fn get(&self, song_id: &str) -> Result<Song, AppError> {
        tracing::debug!(%song_id, "song_service.get");
        let song = self
            .repo
            .get(song_id)
            .await?
            .ok_or_else(|| AppError::NotFound(song_id.into()))?;
        tracing::info!(%song_id, title = %song.title, "song_service.get ok");
        Ok(song)
    }

    pub async fn list(&self, query: SongListQuery) -> Result<CursorPage<Song>, AppError> {
        tracing::debug!(?query, "song_service.list");
        let page = if query.q.as_ref().is_some_and(|q| !q.trim().is_empty()) {
            self.list_text(query).await?
        } else {
            self.repo.list(&query).await?
        };
        tracing::info!(
            returned = page.items.len(),
            total = page.total,
            has_more = page.has_more,
            "song_service.list ok"
        );
        Ok(page)
    }

    async fn list_text(&self, query: SongListQuery) -> Result<CursorPage<Song>, AppError> {
        let text_page = self.text_index.search(&query).await?;
        let song_ids: Vec<String> = text_page
            .hits
            .iter()
            .map(|hit| hit.song_id.clone())
            .collect();
        let songs = self.repo.get_many(&song_ids).await?;
        let mut by_id: HashMap<String, Song> = songs
            .into_iter()
            .map(|song| (song.song_id.clone(), song))
            .collect();
        let items: Vec<Song> = song_ids.iter().filter_map(|id| by_id.remove(id)).collect();

        Ok(CursorPage {
            items,
            limit: text_page.limit,
            next_last_id: text_page.next_last_id,
            next_last_rank: text_page.next_last_rank,
            has_more: text_page.has_more,
            total: text_page.total,
        })
    }

    pub async fn create(&self, mut input: NewSong) -> Result<Song, AppError> {
        input.youtube_url = normalize_youtube_url(&input.youtube_url)?;
        tracing::info!(title = %input.title, youtube_url = %input.youtube_url, "song_service.create");
        validate_new_song(&input)?;
        if self.repo.exists_by_youtube_url(&input.youtube_url).await? {
            tracing::warn!(youtube_url = %input.youtube_url, "song_service.create conflict");
            return Err(AppError::Conflict(
                "a song with this YouTube URL already exists".into(),
            ));
        }

        input.lyrics = clean_lyrics(&input.lyrics);
        let now = Utc::now();
        let song = Song {
            song_id: new_song_id(),
            title: input.title.trim().to_string(),
            youtube_url: input.youtube_url,
            lyrics: input.lyrics.clone(),
            bpm: input.bpm,
            key: input.key.clone(),
            created_at: now,
            updated_at: now,
        };

        let issues = song_readiness_issues(&song);
        if !issues.is_empty() {
            tracing::warn!(?issues, "song_service.create validation failed");
            return Err(AppError::Validation(format!(
                "song not ready for indexing: {}",
                issues.join(", ")
            )));
        }

        self.repo.insert(&song).await?;
        self.text_index.upsert(&song).await?;
        self.index_song(&song, &input.lyrics).await?;
        self.vector_index.flush().await?;
        tracing::info!(song_id = %song.song_id, title = %song.title, "song_service.create ok");
        Ok(song)
    }

    pub async fn update(&self, song_id: &str, mut patch: SongPatch) -> Result<Song, AppError> {
        if let Some(url) = patch.youtube_url.take() {
            patch.youtube_url = Some(normalize_youtube_url(&url)?);
        }
        tracing::info!(%song_id, ?patch, "song_service.update");
        validate_patch(&patch)?;
        if let Some(url) = &patch.youtube_url {
            if let Some(existing) = self.repo.get_by_youtube_url(url).await? {
                if existing.song_id != song_id {
                    tracing::warn!(%song_id, youtube_url = %url, "song_service.update conflict");
                    return Err(AppError::Conflict(
                        "a song with this YouTube URL already exists".into(),
                    ));
                }
            }
        }
        let updated = self.repo.update(song_id, &patch).await?;

        let needs_text_index = patch.title.is_some()
            || patch.lyrics.is_some()
            || patch.bpm.is_some()
            || patch.key.is_some();
        if needs_text_index {
            self.text_index.upsert(&updated).await?;
        }

        let needs_vector_reindex =
            patch.lyrics.is_some() || patch.bpm.is_some() || patch.key.is_some();
        if needs_vector_reindex {
            let issues = song_readiness_issues(&updated);
            if !issues.is_empty() {
                tracing::warn!(%song_id, ?issues, "song_service.update validation failed");
                return Err(AppError::Validation(format!(
                    "song not ready for indexing: {}",
                    issues.join(", ")
                )));
            }
            self.index_song(&updated, &updated.lyrics).await?;
            self.vector_index.flush().await?;
        }

        tracing::info!(
            %song_id,
            title = %updated.title,
            reindexed = needs_vector_reindex,
            "song_service.update ok"
        );
        Ok(updated)
    }

    pub async fn delete(&self, song_id: &str) -> Result<(), AppError> {
        tracing::info!(%song_id, "song_service.delete");
        self.repo.delete(song_id).await?;
        self.text_index.delete(song_id).await?;
        self.vector_index.delete(song_id).await?;
        self.vector_index.flush().await?;
        tracing::info!(%song_id, "song_service.delete ok");
        Ok(())
    }

    /// Re-embed every catalog song and upsert into the vector index.
    ///
    /// Assumes the caller has already wiped the Edge shard if a full rebuild is desired.
    pub async fn reindex_all(&self) -> Result<ReindexReport, AppError> {
        tracing::info!("song_service.reindex_all start");
        let mut indexed = 0u64;
        let mut skipped = 0u64;
        let mut total_songs = 0i64;
        let mut last_id: Option<String> = None;
        let mut last_rank: Option<f64> = None;

        loop {
            let page = self
                .list(SongListQuery {
                    q: None,
                    key: None,
                    bpm_min: None,
                    bpm_max: None,
                    limit: Some(100),
                    last_id: last_id.clone(),
                    last_rank,
                })
                .await?;

            if total_songs == 0 {
                total_songs = page.total;
            }

            let mut ready = Vec::new();
            for song in &page.items {
                let issues = song_readiness_issues(song);
                if issues.is_empty() {
                    ready.push(song);
                } else {
                    skipped += 1;
                    tracing::warn!(
                        song_id = %song.song_id,
                        title = %song.title,
                        ?issues,
                        "song_service.reindex_all skip"
                    );
                }
            }

            if !ready.is_empty() {
                let lyrics: Vec<String> = ready.iter().map(|s| s.lyrics.clone()).collect();
                let vectors = self.embedder.embed(&lyrics, InputType::Document).await?;
                if vectors.len() != ready.len() {
                    return Err(AppError::Embedding(format!(
                        "expected {} embeddings, got {}",
                        ready.len(),
                        vectors.len()
                    )));
                }
                for (song, vector) in ready.iter().zip(vectors) {
                    self.vector_index
                        .upsert(&song.song_id, vector, &song.key, song.bpm)
                        .await?;
                    indexed += 1;
                }
            }

            if !page.has_more {
                break;
            }
            last_id = page.next_last_id;
            last_rank = page.next_last_rank;
        }

        self.vector_index.flush().await?;
        tracing::info!(total_songs, indexed, skipped, "song_service.reindex_all ok");
        Ok(ReindexReport {
            total_songs,
            indexed,
            skipped,
        })
    }

    pub async fn ensure_text_index_synced(&self) -> Result<(), AppError> {
        let catalog_count = self
            .repo
            .list(&SongListQuery {
                q: None,
                key: None,
                bpm_min: None,
                bpm_max: None,
                limit: Some(1),
                last_id: None,
                last_rank: None,
            })
            .await?
            .total as u64;
        let index_count = self.text_index.doc_count().await?;
        if catalog_count == index_count {
            return Ok(());
        }

        tracing::info!(
            catalog_count,
            index_count,
            "rebuilding text index from catalog"
        );
        let songs = self.load_all_songs().await?;
        self.text_index.rebuild(&songs).await?;
        Ok(())
    }

    async fn load_all_songs(&self) -> Result<Vec<Song>, AppError> {
        let mut songs = Vec::new();
        let mut last_id: Option<String> = None;
        loop {
            let page = self
                .repo
                .list(&SongListQuery {
                    q: None,
                    key: None,
                    bpm_min: None,
                    bpm_max: None,
                    limit: Some(100),
                    last_id: last_id.clone(),
                    last_rank: None,
                })
                .await?;
            songs.extend(page.items);
            if !page.has_more {
                break;
            }
            last_id = page.next_last_id;
        }
        Ok(songs)
    }

    async fn index_song(&self, song: &Song, lyrics: &str) -> Result<(), AppError> {
        tracing::debug!(song_id = %song.song_id, "song_service.index_song");
        let vectors = self
            .embedder
            .embed(&[lyrics.to_string()], InputType::Document)
            .await?;
        let vector = vectors
            .into_iter()
            .next()
            .ok_or_else(|| AppError::Embedding("empty embedding response".into()))?;
        self.vector_index
            .upsert(&song.song_id, vector, &song.key, song.bpm)
            .await
    }
}
