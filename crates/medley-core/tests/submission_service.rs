use std::sync::Arc;

use medley_core::domain::error::AppError;
use medley_core::domain::models::NewSong;
use medley_core::embed::MockEmbeddingProvider;
use medley_core::index::{MockTextIndex, MockVectorIndex};
use medley_core::repo::submission_sqlite::SubmissionRepository;
use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::service::song_service::SongService;
use medley_core::service::submission_service::SubmissionService;
use tempfile::TempDir;

async fn setup() -> (TempDir, SubmissionService, Arc<SongService>) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let songs_repo = SqliteSongRepository::connect(&db_path).await.unwrap();
    songs_repo.migrate().await.unwrap();
    let submission_repo = SubmissionRepository::new(songs_repo.pool().clone());
    let songs_arc: Arc<SqliteSongRepository> = Arc::new(songs_repo);

    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));
    let mut vector_index = MockVectorIndex::new();
    vector_index
        .expect_upsert()
        .returning(|_, _, _, _| Box::pin(async { Ok(()) }));
    vector_index
        .expect_flush()
        .returning(|| Box::pin(async { Ok(()) }));
    let mut text_index = MockTextIndex::new();
    text_index
        .expect_upsert()
        .returning(|_| Box::pin(async { Ok(()) }));

    let song_service = Arc::new(SongService::new(
        songs_arc.clone(),
        Arc::new(embedder),
        Arc::new(vector_index),
        Arc::new(text_index),
    ));
    let submission_service = SubmissionService::new(submission_repo, songs_arc.clone());

    (dir, submission_service, song_service)
}

fn sample_new_song(video_id: &str) -> NewSong {
    NewSong {
        title: "Test Song".into(),
        youtube_url: format!("https://www.youtube.com/watch?v={video_id}"),
        lyrics: "line one".into(),
        bpm: 120.0,
        key: "G".into(),
    }
}

#[tokio::test]
async fn submit_stores_without_indexing() {
    let (_dir, submissions, songs) = setup().await;
    let submission = submissions
        .submit(sample_new_song("sub00000001"))
        .await
        .unwrap();
    assert!(!submission.submission_id.is_empty());
    assert!(songs.get(&submission.submission_id).await.is_err());
}

#[tokio::test]
async fn submit_rejects_catalog_url() {
    let (_dir, submissions, songs) = setup().await;
    let input = sample_new_song("cat00000001");
    songs.create(input.clone()).await.unwrap();
    let err = submissions.submit(input).await.unwrap_err();
    assert!(matches!(err, AppError::Conflict(_)));
}

#[tokio::test]
async fn submit_rejects_pending_duplicate_url() {
    let (_dir, submissions, _songs) = setup().await;
    let input = sample_new_song("dup00000001");
    submissions.submit(input.clone()).await.unwrap();
    let err = submissions.submit(input).await.unwrap_err();
    assert!(matches!(err, AppError::Conflict(_)));
}

#[tokio::test]
async fn submit_rejects_invalid_key() {
    let (_dir, submissions, _songs) = setup().await;
    let mut input = sample_new_song("bad00000001");
    input.key = "H".into();
    let err = submissions.submit(input).await.unwrap_err();
    assert!(matches!(err, AppError::Validation(_)));
}

#[tokio::test]
async fn update_excludes_self_from_url_conflict() {
    let (_dir, submissions, _songs) = setup().await;
    let submission = submissions
        .submit(sample_new_song("self0000001"))
        .await
        .unwrap();
    let mut updated = sample_new_song("self0000001");
    updated.title = "Renamed".into();
    submissions
        .update(&submission.submission_id, updated)
        .await
        .unwrap();
    let fetched = submissions.get(&submission.submission_id).await.unwrap();
    assert_eq!(fetched.title, "Renamed");
}

#[tokio::test]
async fn approve_creates_catalog_entry_and_removes_submission() {
    let (_dir, submissions, songs) = setup().await;
    let submission = submissions
        .submit(sample_new_song("app00000001"))
        .await
        .unwrap();
    let id = submission.submission_id.clone();
    let input = sample_new_song("app00000001");

    let song = submissions.approve(&id, input, songs.as_ref()).await.unwrap();
    assert_eq!(song.title, "Test Song");
    assert!(submissions.get(&id).await.is_err());
    assert!(songs.get(&song.song_id).await.is_ok());
}

#[tokio::test]
async fn approve_uses_posted_values_not_stale_db() {
    let (_dir, submissions, songs) = setup().await;
    let submission = submissions
        .submit(sample_new_song("edit0000001"))
        .await
        .unwrap();
    let id = submission.submission_id.clone();
    let mut input = sample_new_song("edit0000001");
    input.title = "Approved Title".into();

    let song = submissions.approve(&id, input, songs.as_ref()).await.unwrap();
    assert_eq!(song.title, "Approved Title");
}
