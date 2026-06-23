use std::path::Path;

use chrono::{DateTime, Utc};
use sqlx::{sqlite::SqlitePoolOptions, Pool, Row, Sqlite};

use crate::domain::error::AppError;
use crate::domain::youtube::normalize_youtube_url;
use crate::domain::models::{Song, SongListQuery, SongPatch};
use crate::domain::pagination::{clamp_limit, CursorPage};

pub struct SqliteSongRepository {
    pool: Pool<Sqlite>,
}

impl SqliteSongRepository {
    pub async fn connect(database_path: &Path) -> Result<Self, AppError> {
        if let Some(parent) = database_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AppError::Internal(e.to_string()))?;
        }
        let url = format!("sqlite:{}?mode=rwc", database_path.display());
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await?;
        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> Result<(), AppError> {
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?;
        migrate_legacy_song_ids_to_uuid(&self.pool).await?;
        normalize_youtube_urls(&self.pool).await?;
        Ok(())
    }

    pub fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }

    fn map_row(row: &sqlx::sqlite::SqliteRow) -> Result<Song, AppError> {
        Ok(Song {
            song_id: row.try_get("song_id")?,
            title: row.try_get("title")?,
            youtube_url: row.try_get("youtube_url")?,
            lyrics: row.try_get("lyrics")?,
            bpm: row.try_get("bpm")?,
            key: row.try_get("key")?,
            created_at: parse_dt(row.try_get::<String, _>("created_at")?)?,
            updated_at: parse_dt(row.try_get::<String, _>("updated_at")?)?,
        })
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

/// Rewrite `youtube/{video_id}` primary keys to UUID v7.
///
/// Uses insert-then-delete (not in-place PK UPDATE) to avoid edge cases with
/// dependent rows during primary-key changes.
async fn migrate_legacy_song_ids_to_uuid(pool: &Pool<Sqlite>) -> Result<(), AppError> {
    use crate::domain::id::{is_legacy_song_id, new_song_id_from_created_at};

    let legacy_count: (i64,) =
        sqlx::query_as("SELECT COUNT(1) FROM songs WHERE song_id LIKE 'youtube/%'")
            .fetch_one(pool)
            .await?;
    if legacy_count.0 == 0 {
        return Ok(());
    }

    tracing::info!(
        "migrating {} legacy song id(s) from youtube/* to UUID v7",
        legacy_count.0
    );

    let rows = sqlx::query(
        "SELECT song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at \
         FROM songs WHERE song_id LIKE 'youtube/%' ORDER BY created_at, song_id",
    )
    .fetch_all(pool)
    .await?;

    let mut tx = pool
        .begin()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    // Allow duplicate youtube_url briefly while both old and new rows exist.
    sqlx::query("DROP INDEX IF EXISTS idx_songs_youtube_url")
        .execute(&mut *tx)
        .await?;

    for row in rows {
        let old_id: String = row.try_get("song_id")?;
        if !is_legacy_song_id(&old_id) {
            continue;
        }
        let created_at: String = row.try_get("created_at")?;
        let new_id = new_song_id_from_created_at(&created_at);

        sqlx::query(
            "INSERT INTO songs (song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&new_id)
        .bind(row.try_get::<String, _>("title")?)
        .bind(row.try_get::<String, _>("youtube_url")?)
        .bind(row.try_get::<String, _>("lyrics")?)
        .bind(row.try_get::<f64, _>("bpm")?)
        .bind(row.try_get::<String, _>("key")?)
        .bind(&created_at)
        .bind(row.try_get::<String, _>("updated_at")?)
        .execute(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM songs WHERE song_id = ?")
            .bind(&old_id)
            .execute(&mut *tx)
            .await?;
    }

    sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_youtube_url ON songs(youtube_url)")
        .execute(&mut *tx)
        .await?;

    tx.commit()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;

    Ok(())
}

async fn normalize_youtube_urls(pool: &Pool<Sqlite>) -> Result<(), AppError> {
    let rows = sqlx::query("SELECT song_id, youtube_url FROM songs")
        .fetch_all(pool)
        .await?;

    for row in rows {
        let song_id: String = row.try_get("song_id")?;
        let url: String = row.try_get("youtube_url")?;
        let normalized = match normalize_youtube_url(&url) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if normalized == url {
            continue;
        }
        sqlx::query("UPDATE songs SET youtube_url = ? WHERE song_id = ?")
            .bind(&normalized)
            .bind(&song_id)
            .execute(pool)
            .await?;
    }

    Ok(())
}

#[async_trait::async_trait]
impl super::SongRepository for SqliteSongRepository {
    async fn get(&self, song_id: &str) -> Result<Option<Song>, AppError> {
        tracing::debug!(%song_id, "repo.get");
        let row = sqlx::query(
            "SELECT song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at FROM songs WHERE song_id = ?",
        )
        .bind(song_id)
        .fetch_optional(&self.pool)
        .await?;
        let found = row.is_some();
        tracing::debug!(%song_id, found, "repo.get done");
        row.as_ref().map(Self::map_row).transpose()
    }

    async fn get_many(&self, song_ids: &[String]) -> Result<Vec<Song>, AppError> {
        tracing::debug!(count = song_ids.len(), "repo.get_many");
        if song_ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders = song_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at FROM songs WHERE song_id IN ({placeholders})"
        );
        let mut q = sqlx::query(&sql);
        for id in song_ids {
            q = q.bind(id);
        }
        let rows = q.fetch_all(&self.pool).await?;
        tracing::debug!(requested = song_ids.len(), found = rows.len(), "repo.get_many done");
        rows.iter().map(Self::map_row).collect()
    }

    async fn insert(&self, song: &Song) -> Result<(), AppError> {
        tracing::info!(song_id = %song.song_id, title = %song.title, "repo.insert");
        sqlx::query(
            "INSERT INTO songs (song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&song.song_id)
        .bind(&song.title)
        .bind(&song.youtube_url)
        .bind(&song.lyrics)
        .bind(song.bpm)
        .bind(&song.key)
        .bind(song.created_at.to_rfc3339())
        .bind(song.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        tracing::info!(song_id = %song.song_id, "repo.insert ok");
        Ok(())
    }

    async fn update(&self, song_id: &str, patch: &SongPatch) -> Result<Song, AppError> {
        tracing::info!(%song_id, ?patch, "repo.update");
        let existing = self
            .get(song_id)
            .await?
            .ok_or_else(|| AppError::NotFound(song_id.into()))?;

        let title = patch.title.as_ref().unwrap_or(&existing.title);
        let youtube_url = patch.youtube_url.as_ref().unwrap_or(&existing.youtube_url);
        let lyrics = patch.lyrics.as_ref().unwrap_or(&existing.lyrics);
        let bpm = patch.bpm.unwrap_or(existing.bpm);
        let key = patch.key.as_ref().unwrap_or(&existing.key);
        let updated_at = Utc::now();

        sqlx::query(
            "UPDATE songs SET title = ?, youtube_url = ?, lyrics = ?, bpm = ?, key = ?, updated_at = ? WHERE song_id = ?",
        )
        .bind(title)
        .bind(youtube_url)
        .bind(lyrics)
        .bind(bpm)
        .bind(key)
        .bind(updated_at.to_rfc3339())
        .bind(song_id)
        .execute(&self.pool)
        .await?;

        tracing::info!(%song_id, "repo.update ok");
        Ok(Song {
            song_id: song_id.to_string(),
            title: title.clone(),
            youtube_url: youtube_url.clone(),
            lyrics: lyrics.clone(),
            bpm,
            key: key.clone(),
            created_at: existing.created_at,
            updated_at,
        })
    }

    async fn delete(&self, song_id: &str) -> Result<(), AppError> {
        tracing::info!(%song_id, "repo.delete");
        let result = sqlx::query("DELETE FROM songs WHERE song_id = ?")
            .bind(song_id)
            .execute(&self.pool)
            .await?;
        if result.rows_affected() == 0 {
            tracing::warn!(%song_id, "repo.delete not found");
            return Err(AppError::NotFound(song_id.into()));
        }
        tracing::info!(%song_id, "repo.delete ok");
        Ok(())
    }

    async fn exists(&self, song_id: &str) -> Result<bool, AppError> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(1) FROM songs WHERE song_id = ?")
            .bind(song_id)
            .fetch_one(&self.pool)
            .await?;
        Ok(row.0 > 0)
    }

    async fn exists_by_youtube_url(&self, youtube_url: &str) -> Result<bool, AppError> {
        let row: (i64,) = sqlx::query_as("SELECT COUNT(1) FROM songs WHERE youtube_url = ?")
            .bind(youtube_url)
            .fetch_one(&self.pool)
            .await?;
        Ok(row.0 > 0)
    }

    async fn get_by_youtube_url(&self, youtube_url: &str) -> Result<Option<Song>, AppError> {
        let row = sqlx::query(
            "SELECT song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at FROM songs WHERE youtube_url = ?",
        )
        .bind(youtube_url)
        .fetch_optional(&self.pool)
        .await?;
        row.as_ref().map(Self::map_row).transpose()
    }

    async fn list(&self, query: &SongListQuery) -> Result<CursorPage<Song>, AppError> {
        tracing::debug!(?query, "repo.list");
        let limit = clamp_limit(query.limit, 20, 100);
        let fetch = limit + 1;

        if let Some(last_id) = &query.last_id {
            if !super::SongRepository::exists(self, last_id).await? {
                return Err(AppError::InvalidCursor(format!("unknown last_id: {last_id}")));
            }
        }

        let mut sql = String::from(
            "SELECT song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at FROM songs WHERE 1=1",
        );
        if query.last_id.is_some() {
            sql.push_str(" AND song_id > ?");
        }
        if query.key.is_some() {
            sql.push_str(" AND key = ?");
        }
        if query.bpm_min.is_some() {
            sql.push_str(" AND bpm >= ?");
        }
        if query.bpm_max.is_some() {
            sql.push_str(" AND bpm <= ?");
        }
        sql.push_str(" ORDER BY song_id ASC LIMIT ?");

        let mut q = sqlx::query(&sql);
        if let Some(last_id) = &query.last_id {
            q = q.bind(last_id);
        }
        if let Some(key) = &query.key {
            q = q.bind(key);
        }
        if let Some(bpm_min) = query.bpm_min {
            q = q.bind(bpm_min);
        }
        if let Some(bpm_max) = query.bpm_max {
            q = q.bind(bpm_max);
        }
        q = q.bind(fetch as i64);

        let rows = q.fetch_all(&self.pool).await?;
        let songs: Result<Vec<Song>, AppError> = rows.iter().map(Self::map_row).collect();
        let songs = songs?;

        let total = self.count_filtered(query).await?;
        let page = CursorPage::from_rows(
            songs,
            limit,
            total,
            |s| &s.song_id,
            None::<fn(&Song) -> f64>,
        );
        tracing::debug!(
            returned = page.items.len(),
            total = page.total,
            has_more = page.has_more,
            "repo.list done"
        );
        Ok(page)
    }
}

impl SqliteSongRepository {
    async fn count_filtered(&self, query: &SongListQuery) -> Result<i64, AppError> {
        let mut sql = String::from("SELECT COUNT(1) as cnt FROM songs WHERE 1=1");
        if query.key.is_some() {
            sql.push_str(" AND key = ?");
        }
        if query.bpm_min.is_some() {
            sql.push_str(" AND bpm >= ?");
        }
        if query.bpm_max.is_some() {
            sql.push_str(" AND bpm <= ?");
        }
        let mut q = sqlx::query(&sql);
        if let Some(key) = &query.key {
            q = q.bind(key);
        }
        if let Some(bpm_min) = query.bpm_min {
            q = q.bind(bpm_min);
        }
        if let Some(bpm_max) = query.bpm_max {
            q = q.bind(bpm_max);
        }
        let row = q.fetch_one(&self.pool).await?;
        Ok(row.try_get("cnt")?)
    }
}
