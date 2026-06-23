use std::sync::Arc;

use chrono::Utc;

use crate::domain::error::AppError;
use crate::domain::id::new_song_id;
use crate::domain::lyrics::clean_lyrics;
use crate::domain::models::{NewSong, Song, SongSubmission, SubmissionListQuery};
use crate::domain::pagination::CursorPage;
use crate::domain::validation::validate_new_song;
use crate::domain::youtube::normalize_youtube_url;
use crate::repo::submission_sqlite::SubmissionRepository;
use crate::repo::SongRepository;
use crate::service::song_service::SongService;

pub struct SubmissionService {
    repo: SubmissionRepository,
    songs: Arc<dyn SongRepository>,
}

impl SubmissionService {
    pub fn new(repo: SubmissionRepository, songs: Arc<dyn SongRepository>) -> Self {
        Self { repo, songs }
    }

    pub async fn get(&self, submission_id: &str) -> Result<SongSubmission, AppError> {
        self.repo
            .get(submission_id)
            .await?
            .ok_or_else(|| AppError::NotFound(submission_id.into()))
    }

    pub async fn list(
        &self,
        query: SubmissionListQuery,
    ) -> Result<CursorPage<SongSubmission>, AppError> {
        self.repo.list(&query).await
    }

    pub async fn submit(&self, input: NewSong) -> Result<SongSubmission, AppError> {
        let input = self.prepare_new_song(input, None).await?;
        let now = Utc::now();
        let submission = SongSubmission {
            submission_id: new_song_id(),
            title: input.title,
            youtube_url: input.youtube_url,
            lyrics: input.lyrics,
            bpm: input.bpm,
            key: input.key,
            submitted_at: now,
        };
        self.repo.insert(&submission).await?;
        Ok(submission)
    }

    pub async fn update(&self, submission_id: &str, input: NewSong) -> Result<(), AppError> {
        if !self.repo.exists(submission_id).await? {
            return Err(AppError::NotFound(submission_id.into()));
        }
        let input = self.prepare_new_song(input, Some(submission_id)).await?;
        self.repo.update(submission_id, &input).await
    }

    pub async fn approve(
        &self,
        submission_id: &str,
        input: NewSong,
        songs: &SongService,
    ) -> Result<Song, AppError> {
        if !self.repo.exists(submission_id).await? {
            return Err(AppError::NotFound(submission_id.into()));
        }
        let input = self.prepare_new_song(input, Some(submission_id)).await?;
        let song = songs.create(input).await?;
        self.repo.delete(submission_id).await?;
        Ok(song)
    }

    pub async fn delete(&self, submission_id: &str) -> Result<(), AppError> {
        self.repo.delete(submission_id).await
    }

    async fn prepare_new_song(
        &self,
        mut input: NewSong,
        exclude_submission_id: Option<&str>,
    ) -> Result<NewSong, AppError> {
        input.youtube_url = normalize_youtube_url(&input.youtube_url)?;
        validate_new_song(&input)?;

        if self.songs.exists_by_youtube_url(&input.youtube_url).await? {
            return Err(AppError::Conflict(
                "a song with this YouTube URL already exists".into(),
            ));
        }

        if let Some(existing) = self.repo.get_by_youtube_url(&input.youtube_url).await? {
            if exclude_submission_id != Some(existing.submission_id.as_str()) {
                return Err(AppError::Conflict(
                    "a submission with this YouTube URL already exists".into(),
                ));
            }
        }

        input.lyrics = clean_lyrics(&input.lyrics);
        input.title = input.title.trim().to_string();
        input.key = input.key.trim().to_string();
        Ok(input)
    }
}
