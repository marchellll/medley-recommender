use std::sync::Arc;

use chrono::Utc;
use medley_core::domain::error::AppError;
use medley_core::domain::models::{SearchQuery, Song};
use medley_core::embed::{InputType, MockEmbeddingProvider};
use medley_core::index::{MockVectorIndex, VectorHit};
use medley_core::repo::MockSongRepository;
use medley_core::service::search_service::SearchService;

fn sample_song(id: &str) -> Song {
    let now = Utc::now();
    Song {
        song_id: id.into(),
        title: "Test Song".into(),
        youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ".into(),
        lyrics: "line one".into(),
        bpm: 120.0,
        key: "G".into(),
        created_at: now,
        updated_at: now,
    }
}

#[tokio::test]
async fn search_rejects_empty_query() {
    let repo = MockSongRepository::new();
    let embedder = MockEmbeddingProvider::new();
    let index = MockVectorIndex::new();
    let service = SearchService::new(Arc::new(repo), Arc::new(embedder), Arc::new(index));

    let err = service
        .search(SearchQuery {
            query: "   ".into(),
            bpm_min: None,
            bpm_max: None,
            keys: None,
            limit: None,
        })
        .await
        .unwrap_err();

    assert!(matches!(err, AppError::Validation(_)));
}

#[tokio::test]
async fn search_embeds_queries_and_joins_catalog_rows() {
    let song = sample_song("song-1");
    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .times(1)
        .withf(|texts, input_type| texts == ["grace".to_string()] && *input_type == InputType::Query)
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));

    let mut index = MockVectorIndex::new();
    index
        .expect_search()
        .times(1)
        .withf(|_, limit, _, _, _| *limit == 10)
        .returning(|_, _, _, _, _| {
            Box::pin(async {
                Ok(vec![VectorHit {
                    song_id: "song-1".into(),
                    score: 0.92,
                }])
            })
        });

    let mut repo = MockSongRepository::new();
    repo.expect_get_many()
        .times(1)
        .withf(|ids| ids == ["song-1".to_string()])
        .returning({
            let song = song.clone();
            move |_| {
                let song = song.clone();
                Box::pin(async move { Ok(vec![song]) })
            }
        });

    let service = SearchService::new(Arc::new(repo), Arc::new(embedder), Arc::new(index));
    let results = service
        .search(SearchQuery {
            query: "grace".into(),
            bpm_min: None,
            bpm_max: None,
            keys: None,
            limit: None,
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].song_id, "song-1");
    assert_eq!(results[0].title, "Test Song");
    assert!((results[0].similarity_score - 0.92).abs() < 1e-5);
}

#[tokio::test]
async fn search_skips_index_hits_missing_from_catalog() {
    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .times(1)
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));

    let mut index = MockVectorIndex::new();
    index.expect_search().times(1).returning(|_, _, _, _, _| {
        Box::pin(async {
            Ok(vec![
                VectorHit {
                    song_id: "missing".into(),
                    score: 0.5,
                },
                VectorHit {
                    song_id: "present".into(),
                    score: 0.4,
                },
            ])
        })
    });

    let mut repo = MockSongRepository::new();
    repo.expect_get_many()
        .times(1)
        .returning(|_| Box::pin(async { Ok(vec![sample_song("present")]) }));

    let service = SearchService::new(Arc::new(repo), Arc::new(embedder), Arc::new(index));
    let results = service
        .search(SearchQuery {
            query: "test".into(),
            bpm_min: None,
            bpm_max: None,
            keys: None,
            limit: Some(2),
        })
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].song_id, "present");
}
