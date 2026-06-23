use chrono::Utc;
use medley_core::domain::id::new_song_id;
use medley_core::domain::models::{Song, SongListQuery};
use medley_core::repo::SongRepository;
use medley_core::repo::sqlite::SqliteSongRepository;
use tempfile::TempDir;
use uuid::Uuid;

async fn setup_repo() -> (TempDir, SqliteSongRepository) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let repo = SqliteSongRepository::connect(&db_path).await.unwrap();
    repo.migrate().await.unwrap();
    (dir, repo)
}

fn sample_song(video_id: &str, title: &str, lyrics: &str) -> Song {
    let now = Utc::now();
    Song {
        song_id: new_song_id(),
        title: title.into(),
        youtube_url: format!("https://www.youtube.com/watch?v={video_id}"),
        lyrics: lyrics.into(),
        bpm: 120.0,
        key: "G".into(),
        created_at: now,
        updated_at: now,
    }
}

#[tokio::test]
async fn crud_round_trip() {
    let (_dir, repo) = setup_repo().await;
    let song = sample_song("test1234567", "Amazing Grace", "Line one\nLine two");
    let song_id = song.song_id.clone();
    repo.insert(&song).await.unwrap();

    let fetched = repo.get(&song_id).await.unwrap().unwrap();
    assert_eq!(fetched.title, "Amazing Grace");
    assert!(Uuid::parse_str(&fetched.song_id).is_ok());

    repo.delete(&song_id).await.unwrap();
    assert!(repo.get(&song_id).await.unwrap().is_none());
}

#[tokio::test]
async fn stable_plain_pagination() {
    let (_dir, repo) = setup_repo().await;
    for i in 0..5 {
        repo.insert(&sample_song(
            &format!("vid{i:02}"),
            &format!("Song {i}"),
            "lyrics",
        ))
        .await
        .unwrap();
    }

    let page1 = repo
        .list(&SongListQuery {
            q: None,
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(2),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();
    assert_eq!(page1.items.len(), 2);
    assert!(page1.has_more);
    let last = page1.next_last_id.clone().unwrap();

    let page2 = repo
        .list(&SongListQuery {
            q: None,
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(2),
            last_id: Some(last),
            last_rank: None,
        })
        .await
        .unwrap();
    assert_ne!(page1.items[1].song_id, page2.items[0].song_id);
}

#[tokio::test]
async fn invalid_cursor_errors() {
    let (_dir, repo) = setup_repo().await;
    repo.insert(&sample_song("x", "X", "lyrics")).await.unwrap();

    let err = repo
        .list(&SongListQuery {
            q: None,
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: Some(new_song_id()),
            last_rank: None,
        })
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown last_id"));
}

#[tokio::test]
async fn legacy_youtube_ids_migrated_to_uuid() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("legacy.db");
    let repo = SqliteSongRepository::connect(&db_path).await.unwrap();
    sqlx::migrate!("./migrations")
        .run(repo.pool())
        .await
        .unwrap();

    let now = Utc::now();
    sqlx::query(
        "INSERT INTO songs (song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("youtube/legacy12345")
    .bind("Legacy Song")
    .bind("https://www.youtube.com/watch?v=legacy12345")
    .bind("old lyrics")
    .bind(100.0)
    .bind("C")
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(repo.pool())
    .await
    .unwrap();

    repo.migrate().await.unwrap();

    assert!(
        repo.get("youtube/legacy12345")
            .await
            .unwrap()
            .is_none()
    );
    let song = repo
        .get_by_youtube_url("https://www.youtube.com/watch?v=legacy12345")
        .await
        .unwrap()
        .expect("song preserved");
    assert!(Uuid::parse_str(&song.song_id).is_ok());
    assert_eq!(song.title, "Legacy Song");
}

#[tokio::test]
async fn migration_normalizes_youtube_urls() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("medley.db");
    sqlx::migrate!("./migrations")
        .run(&sqlx::sqlite::SqlitePoolOptions::new()
            .connect(&format!("sqlite:{}?mode=rwc", db_path.display()))
            .await
            .unwrap())
        .await
        .unwrap();

    let now = Utc::now();
    sqlx::query(
        "INSERT INTO songs (song_id, title, youtube_url, lyrics, bpm, key, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind("019ade09-0000-7000-8000-000000000099")
    .bind("Query Params Song")
    .bind("https://www.youtube.com/watch?v=query123456&list=PLabc&t=10")
    .bind("lyrics")
    .bind(100.0)
    .bind("G")
    .bind(now.to_rfc3339())
    .bind(now.to_rfc3339())
    .execute(
        &sqlx::sqlite::SqlitePoolOptions::new()
            .connect(&format!("sqlite:{}?mode=rwc", db_path.display()))
            .await
            .unwrap(),
    )
    .await
    .unwrap();

    let repo = medley_core::repo::sqlite::SqliteSongRepository::connect(&db_path)
        .await
        .unwrap();
    repo.migrate().await.unwrap();

    let song = repo
        .get("019ade09-0000-7000-8000-000000000099")
        .await
        .unwrap()
        .expect("song preserved");
    assert_eq!(
        song.youtube_url,
        "https://www.youtube.com/watch?v=query123456"
    );
}

#[tokio::test]
async fn migration_drops_songs_fts() {
    let (_dir, repo) = setup_repo().await;
    let fts_exists: (i64,) = sqlx::query_as(
        "SELECT COUNT(1) FROM sqlite_master WHERE type = 'table' AND name = 'songs_fts'",
    )
    .fetch_one(repo.pool())
    .await
    .unwrap();
    assert_eq!(fts_exists.0, 0);
}
