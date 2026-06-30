use std::sync::Arc;

use chrono::Utc;
use medley_core::domain::error::AppError;
use medley_core::domain::models::{NewSong, Song, SongPatch};
use medley_core::embed::{InputType, MockEmbeddingProvider};
use medley_core::index::{MockTextIndex, MockVectorIndex};
use medley_core::repo::MockSongRepository;
use medley_core::service::song_service::SongService;

fn sample_song(id: &str) -> Song {
    let now = Utc::now();
    Song {
        song_id: id.into(),
        title: "Amazing Grace".into(),
        youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".into(),
        lyrics: "hello world".into(),
        bpm: 120.0,
        key: "G".into(),
        created_at: now,
        updated_at: now,
    }
}

fn sample_new_song() -> NewSong {
    NewSong {
        title: "Amazing Grace".into(),
        youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".into(),
        lyrics: "hello world".into(),
        bpm: 120.0,
        key: "G".into(),
    }
}

fn noop_text_index() -> MockTextIndex {
    MockTextIndex::new()
}

#[tokio::test]
async fn create_inserts_embeds_upserts_and_flushes() {
    let mut repo = MockSongRepository::new();
    repo.expect_exists_by_youtube_url()
        .times(1)
        .returning(|_| Box::pin(async { Ok(false) }));
    repo.expect_insert()
        .times(1)
        .withf(|song: &Song| song.title == "Amazing Grace" && song.lyrics == "hello world")
        .returning(|_| Box::pin(async { Ok(()) }));

    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .times(1)
        .withf(|texts, input_type| {
            texts == ["hello world".to_string()] && *input_type == InputType::Document
        })
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));

    let mut index = MockVectorIndex::new();
    index
        .expect_upsert()
        .times(1)
        .withf(|_, _, key, bpm| key == "G" && (*bpm - 120.0).abs() < f64::EPSILON)
        .returning(|_, _, _, _| Box::pin(async { Ok(()) }));
    index
        .expect_flush()
        .times(1)
        .returning(|| Box::pin(async { Ok(()) }));

    let mut text_index = noop_text_index();
    text_index
        .expect_upsert()
        .times(1)
        .returning(|_| Box::pin(async { Ok(()) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(embedder),
        Arc::new(index),
        Arc::new(text_index),
    );
    let song: Song = service.create(sample_new_song()).await.unwrap();
    assert_eq!(song.title, "Amazing Grace");
}

#[tokio::test]
async fn update_title_only_skips_vector_reindex() {
    let song = sample_song("019ade09-0000-7000-8000-000000000001");
    let mut repo = MockSongRepository::new();
    repo.expect_update().times(1).returning({
        let song = song.clone();
        move |id, patch| {
            assert_eq!(id, "019ade09-0000-7000-8000-000000000001");
            assert_eq!(patch.title.as_deref(), Some("New Title"));
            assert!(patch.lyrics.is_none());
            let song = song.clone();
            Box::pin(async move {
                Ok(Song {
                    title: "New Title".into(),
                    ..song
                })
            })
        }
    });

    let mut embedder = MockEmbeddingProvider::new();
    embedder.expect_embed().times(0);

    let mut index = MockVectorIndex::new();
    index.expect_upsert().times(0);
    index.expect_flush().times(0);

    let mut text_index = noop_text_index();
    text_index
        .expect_upsert()
        .times(1)
        .returning(|_| Box::pin(async { Ok(()) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(embedder),
        Arc::new(index),
        Arc::new(text_index),
    );
    let updated = service
        .update(
            "019ade09-0000-7000-8000-000000000001",
            SongPatch {
                title: Some("New Title".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.title, "New Title");
}

#[tokio::test]
async fn update_lyrics_reindexes() {
    let song = sample_song("019ade09-0000-7000-8000-000000000001");
    let mut repo = MockSongRepository::new();
    repo.expect_update().times(1).returning({
        let song = song.clone();
        move |_, patch| {
            assert!(patch.lyrics.is_some());
            let song = song.clone();
            Box::pin(async move {
                Ok(Song {
                    lyrics: "updated lyrics".into(),
                    ..song
                })
            })
        }
    });

    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .times(1)
        .withf(|texts, input_type| {
            texts == ["updated lyrics".to_string()] && *input_type == InputType::Document
        })
        .returning(|_, _| Box::pin(async { Ok(vec![vec![0.0, 1.0, 0.0, 0.0]]) }));

    let mut index = MockVectorIndex::new();
    index
        .expect_upsert()
        .times(1)
        .returning(|_, _, _, _| Box::pin(async { Ok(()) }));
    index
        .expect_flush()
        .times(1)
        .returning(|| Box::pin(async { Ok(()) }));

    let mut text_index = noop_text_index();
    text_index
        .expect_upsert()
        .times(1)
        .returning(|_| Box::pin(async { Ok(()) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(embedder),
        Arc::new(index),
        Arc::new(text_index),
    );
    service
        .update(
            "019ade09-0000-7000-8000-000000000001",
            SongPatch {
                lyrics: Some("updated lyrics".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_removes_from_repo_and_index() {
    let mut repo = MockSongRepository::new();
    repo.expect_delete()
        .times(1)
        .with(mockall::predicate::eq("song-1"))
        .returning(|_| Box::pin(async { Ok(()) }));

    let mut embedder = MockEmbeddingProvider::new();
    embedder.expect_embed().times(0);

    let mut index = MockVectorIndex::new();
    index
        .expect_delete()
        .times(1)
        .with(mockall::predicate::eq("song-1"))
        .returning(|_| Box::pin(async { Ok(()) }));
    index
        .expect_flush()
        .times(1)
        .returning(|| Box::pin(async { Ok(()) }));

    let mut text_index = noop_text_index();
    text_index
        .expect_delete()
        .times(1)
        .with(mockall::predicate::eq("song-1"))
        .returning(|_| Box::pin(async { Ok(()) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(embedder),
        Arc::new(index),
        Arc::new(text_index),
    );
    service.delete("song-1").await.unwrap();
}

#[tokio::test]
async fn create_strips_youtube_query_params() {
    let mut repo = MockSongRepository::new();
    repo.expect_exists_by_youtube_url()
        .times(1)
        .with(mockall::predicate::eq(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ))
        .returning(|_| Box::pin(async { Ok(false) }));
    repo.expect_insert()
        .times(1)
        .withf(|song: &Song| song.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        .returning(|_| Box::pin(async { Ok(()) }));

    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .times(1)
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));

    let mut index = MockVectorIndex::new();
    index
        .expect_upsert()
        .times(1)
        .returning(|_, _, _, _| Box::pin(async { Ok(()) }));
    index
        .expect_flush()
        .times(1)
        .returning(|| Box::pin(async { Ok(()) }));

    let mut text_index = noop_text_index();
    text_index
        .expect_upsert()
        .times(1)
        .returning(|_| Box::pin(async { Ok(()) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(embedder),
        Arc::new(index),
        Arc::new(text_index),
    );

    let mut input = sample_new_song();
    input.youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc".into();
    let song = service.create(input).await.unwrap();
    assert_eq!(
        song.youtube_url,
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    );
}

#[tokio::test]
async fn create_rejects_duplicate_when_only_query_params_differ() {
    let mut repo = MockSongRepository::new();
    repo.expect_exists_by_youtube_url()
        .times(1)
        .with(mockall::predicate::eq(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ))
        .returning(|_| Box::pin(async { Ok(true) }));

    let service = SongService::new(
        Arc::new(repo),
        Arc::new(MockEmbeddingProvider::new()),
        Arc::new(MockVectorIndex::new()),
        Arc::new(noop_text_index()),
    );

    let mut input = sample_new_song();
    input.youtube_url = "https://youtu.be/dQw4w9WgXcQ?t=30".into();
    let err = service.create(input).await.unwrap_err();
    assert!(matches!(err, AppError::Conflict(_)));
}
