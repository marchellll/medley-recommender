use chrono::Utc;
use medley_core::domain::id::new_song_id;
use medley_core::domain::models::{NewSong, SongSubmission, SubmissionListQuery};
use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::repo::submission_sqlite::SubmissionRepository;
use tempfile::TempDir;

async fn setup() -> (TempDir, SubmissionRepository, SqliteSongRepository) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let songs = SqliteSongRepository::connect(&db_path).await.unwrap();
    songs.migrate().await.unwrap();
    let submissions = SubmissionRepository::new(songs.pool().clone());
    (dir, submissions, songs)
}

fn sample_submission(video_id: &str) -> SongSubmission {
    SongSubmission {
        submission_id: new_song_id(),
        title: "Test Song".into(),
        youtube_url: format!("https://www.youtube.com/watch?v={video_id}"),
        lyrics: "hello world".into(),
        bpm: 120.0,
        key: "G".into(),
        submitted_at: Utc::now(),
    }
}

fn sample_new_song(video_id: &str) -> NewSong {
    NewSong {
        title: "Updated".into(),
        youtube_url: format!("https://www.youtube.com/watch?v={video_id}"),
        lyrics: "updated lyrics".into(),
        bpm: 100.0,
        key: "C".into(),
    }
}

#[tokio::test]
async fn insert_get_delete_round_trip() {
    let (_dir, repo, _songs) = setup().await;
    let submission = sample_submission("sub12345678");
    let id = submission.submission_id.clone();
    repo.insert(&submission).await.unwrap();

    let fetched = repo.get(&id).await.unwrap().unwrap();
    assert_eq!(fetched.title, "Test Song");

    repo.delete(&id).await.unwrap();
    assert!(repo.get(&id).await.unwrap().is_none());
}

#[tokio::test]
async fn update_changes_fields() {
    let (_dir, repo, _songs) = setup().await;
    let submission = sample_submission("upd12345678");
    let id = submission.submission_id.clone();
    repo.insert(&submission).await.unwrap();

    repo.update(&id, &sample_new_song("upd12345678"))
        .await
        .unwrap();
    let fetched = repo.get(&id).await.unwrap().unwrap();
    assert_eq!(fetched.title, "Updated");
    assert_eq!(fetched.bpm, 100.0);
}

#[tokio::test]
async fn duplicate_youtube_url_rejected_by_db() {
    let (_dir, repo, _songs) = setup().await;
    repo.insert(&sample_submission("dup12345678"))
        .await
        .unwrap();
    let mut second = sample_submission("dup12345678");
    second.submission_id = new_song_id();
    assert!(repo.insert(&second).await.is_err());
}

#[tokio::test]
async fn list_pagination() {
    let (_dir, repo, _songs) = setup().await;
    for i in 0..5 {
        let mut s = sample_submission(&format!("pag{i:02}234567"));
        s.title = format!("Song {i}");
        repo.insert(&s).await.unwrap();
    }

    let page1 = repo
        .list(&SubmissionListQuery {
            limit: Some(2),
            last_id: None,
        })
        .await
        .unwrap();
    assert_eq!(page1.items.len(), 2);
    assert!(page1.has_more);

    let page2 = repo
        .list(&SubmissionListQuery {
            limit: Some(2),
            last_id: page1.next_last_id,
        })
        .await
        .unwrap();
    assert_eq!(page2.items.len(), 2);
    assert!(page2.items[0].submission_id != page1.items[0].submission_id);
}

#[tokio::test]
async fn exists_by_youtube_url() {
    let (_dir, repo, _songs) = setup().await;
    let submission = sample_submission("exi12345678");
    let url = submission.youtube_url.clone();
    repo.insert(&submission).await.unwrap();
    assert!(repo.exists_by_youtube_url(&url).await.unwrap());
    assert!(!repo
        .exists_by_youtube_url("https://www.youtube.com/watch?v=other123456")
        .await
        .unwrap());
}
