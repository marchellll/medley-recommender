use chrono::{DateTime, Utc};
use sqlx::{Pool, Row, Sqlite};

use crate::domain::error::AppError;
use crate::domain::models::{NewSong, SongSubmission, SubmissionListQuery};
use crate::domain::pagination::{clamp_limit, CursorPage};

pub struct SubmissionRepository {
    pool: Pool<Sqlite>,
}

impl SubmissionRepository {
    pub fn new(pool: Pool<Sqlite>) -> Self {
        Self { pool }
    }

    fn map_row(row: &sqlx::sqlite::SqliteRow) -> Result<SongSubmission, AppError> {
        Ok(SongSubmission {
            submission_id: row.try_get("submission_id")?,
            title: row.try_get("title")?,
            youtube_url: row.try_get("youtube_url")?,
            lyrics: row.try_get("lyrics")?,
            bpm: row.try_get("bpm")?,
            key: row.try_get("key")?,
            submitted_at: parse_dt(row.try_get::<String, _>("submitted_at")?)?,
        })
    }

    pub async fn insert(&self, submission: &SongSubmission) -> Result<(), AppError> {
        sqlx::query(
            "INSERT INTO song_submissions (submission_id, title, youtube_url, lyrics, bpm, key, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&submission.submission_id)
        .bind(&submission.title)
        .bind(&submission.youtube_url)
        .bind(&submission.lyrics)
        .bind(submission.bpm)
        .bind(&submission.key)
        .bind(submission.submitted_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn get(&self, submission_id: &str) -> Result<Option<SongSubmission>, AppError> {
        let row = sqlx::query(
            "SELECT submission_id, title, youtube_url, lyrics, bpm, key, submitted_at FROM song_submissions WHERE submission_id = ?",
        )
        .bind(submission_id)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::map_row).transpose()
    }

    pub async fn exists(&self, submission_id: &str) -> Result<bool, AppError> {
        let row: (i64,) =
            sqlx::query_as("SELECT COUNT(1) FROM song_submissions WHERE submission_id = ?")
                .bind(submission_id)
                .fetch_one(&self.pool)
                .await?;
        Ok(row.0 > 0)
    }

    pub async fn exists_by_youtube_url(&self, youtube_url: &str) -> Result<bool, AppError> {
        let row: (i64,) =
            sqlx::query_as("SELECT COUNT(1) FROM song_submissions WHERE youtube_url = ?")
                .bind(youtube_url)
                .fetch_one(&self.pool)
                .await?;
        Ok(row.0 > 0)
    }

    pub async fn get_by_youtube_url(
        &self,
        youtube_url: &str,
    ) -> Result<Option<SongSubmission>, AppError> {
        let row = sqlx::query(
            "SELECT submission_id, title, youtube_url, lyrics, bpm, key, submitted_at FROM song_submissions WHERE youtube_url = ?",
        )
        .bind(youtube_url)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::map_row).transpose()
    }

    pub async fn update(&self, submission_id: &str, input: &NewSong) -> Result<(), AppError> {
        let result = sqlx::query(
            "UPDATE song_submissions SET title = ?, youtube_url = ?, lyrics = ?, bpm = ?, key = ? WHERE submission_id = ?",
        )
        .bind(&input.title)
        .bind(&input.youtube_url)
        .bind(&input.lyrics)
        .bind(input.bpm)
        .bind(&input.key)
        .bind(submission_id)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            return Err(AppError::NotFound(submission_id.into()));
        }
        Ok(())
    }

    pub async fn delete(&self, submission_id: &str) -> Result<(), AppError> {
        let result = sqlx::query("DELETE FROM song_submissions WHERE submission_id = ?")
            .bind(submission_id)
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            return Err(AppError::NotFound(submission_id.into()));
        }
        Ok(())
    }

    pub async fn count(&self) -> Result<i64, AppError> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(1) FROM song_submissions")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.0)
    }

    pub async fn list(
        &self,
        query: &SubmissionListQuery,
    ) -> Result<CursorPage<SongSubmission>, AppError> {
        let limit = clamp_limit(query.limit, 20, 100);
        let fetch = limit + 1;

        if let Some(last_id) = &query.last_id {
            if !self.exists(last_id).await? {
                return Err(AppError::InvalidCursor(format!(
                    "unknown last_id: {last_id}"
                )));
            }
        }

        let mut sql = String::from(
            "SELECT submission_id, title, youtube_url, lyrics, bpm, key, submitted_at FROM song_submissions WHERE 1=1",
        );
        if query.last_id.is_some() {
            sql.push_str(" AND submission_id < ?");
        }
        sql.push_str(" ORDER BY submission_id DESC LIMIT ?");

        let mut q = sqlx::query(&sql);
        if let Some(last_id) = &query.last_id {
            q = q.bind(last_id);
        }
        q = q.bind(fetch as i64);

        let rows = q.fetch_all(&self.pool).await?;
        let items: Result<Vec<SongSubmission>, AppError> = rows.iter().map(Self::map_row).collect();
        let items = items?;
        let total = self.count().await?;

        Ok(CursorPage::from_rows(
            items,
            limit,
            total,
            |s| &s.submission_id,
            None::<fn(&SongSubmission) -> f64>,
        ))
    }
}

fn parse_dt(s: String) -> Result<DateTime<Utc>, AppError> {
    use chrono::NaiveDateTime;

    DateTime::parse_from_rfc3339(&s)
        .map(|d| d.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S"))
                .map(|n| n.and_utc())
        })
        .map_err(|e| AppError::Internal(format!("bad datetime: {e}")))
}
