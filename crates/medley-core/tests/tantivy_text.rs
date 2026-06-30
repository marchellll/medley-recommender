use std::sync::Arc;

use chrono::Utc;
use medley_core::domain::id::new_song_id;
use medley_core::domain::models::{Song, SongListQuery};
use medley_core::embed::MockEmbeddingProvider;
use medley_core::index::tantivy_text::TantivyTextIndex;
use medley_core::index::MockVectorIndex;
use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::repo::SongRepository;
use medley_core::service::song_service::SongService;
use tempfile::TempDir;

fn sample_song(title: &str, lyrics: &str, key: &str, bpm: f64) -> Song {
    let now = Utc::now();
    Song {
        song_id: new_song_id(),
        title: title.into(),
        youtube_url: format!("https://www.youtube.com/watch?v={title}"),
        lyrics: lyrics.into(),
        bpm,
        key: key.into(),
        created_at: now,
        updated_at: now,
    }
}

async fn setup_service() -> (TempDir, SongService, Arc<SqliteSongRepository>) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let text_path = dir.path().join("text_index");
    let repo = Arc::new(SqliteSongRepository::connect(&db_path).await.unwrap());
    repo.migrate().await.unwrap();
    let text_index = TantivyTextIndex::open(&text_path).unwrap();
    let repo_dyn: Arc<dyn SongRepository> = repo.clone();

    let mut vector_index = MockVectorIndex::new();
    vector_index
        .expect_flush()
        .returning(|| Box::pin(async { Ok(()) }));
    vector_index
        .expect_upsert()
        .returning(|_, _, _, _| Box::pin(async { Ok(()) }));

    let mut embedder = MockEmbeddingProvider::new();
    embedder
        .expect_embed()
        .returning(|_, _| Box::pin(async { Ok(vec![vec![1.0, 0.0, 0.0, 0.0]]) }));

    let service = SongService::new(
        repo_dyn,
        Arc::new(embedder),
        Arc::new(vector_index),
        Arc::new(text_index),
    );
    (dir, service, repo)
}

#[tokio::test]
async fn prefix_search_matches_partial_word() {
    let (_dir, service, repo) = setup_service().await;
    let song = sample_song("Amazing Grace", "shalom peace everywhere", "G", 120.0);
    repo.insert(&song).await.unwrap();
    service.ensure_text_index_synced().await.unwrap();

    let page = service
        .list(SongListQuery {
            q: Some("shal".into()),
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].song_id, song.song_id);
}

#[tokio::test]
async fn typo_search_matches_near_word() {
    let (_dir, service, repo) = setup_service().await;
    let song = sample_song("Peace Song", "shalom peace everywhere", "G", 120.0);
    repo.insert(&song).await.unwrap();
    service.ensure_text_index_synced().await.unwrap();

    let page = service
        .list(SongListQuery {
            q: Some("shalum".into()),
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].song_id, song.song_id);
}

#[tokio::test]
async fn multi_word_search_requires_all_terms() {
    let (_dir, service, repo) = setup_service().await;
    let matching = sample_song("Amazing Grace", "amazing grace how sweet", "G", 120.0);
    let other = sample_song("Only Amazing", "amazing day today", "G", 120.0);
    repo.insert(&matching).await.unwrap();
    repo.insert(&other).await.unwrap();
    service.ensure_text_index_synced().await.unwrap();

    let page = service
        .list(SongListQuery {
            q: Some("amazing grace".into()),
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].song_id, matching.song_id);
}

#[tokio::test]
async fn key_filter_applies_to_text_search() {
    let (_dir, service, repo) = setup_service().await;
    let in_g = sample_song("Grace in G", "amazing grace", "G", 120.0);
    let in_c = sample_song("Grace in C", "amazing grace", "C", 120.0);
    repo.insert(&in_g).await.unwrap();
    repo.insert(&in_c).await.unwrap();
    service.ensure_text_index_synced().await.unwrap();

    let page = service
        .list(SongListQuery {
            q: Some("grace".into()),
            key: Some("G".into()),
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].song_id, in_g.song_id);
}

#[tokio::test]
async fn text_search_pagination_is_stable() {
    let (_dir, service, repo) = setup_service().await;
    for i in 0..5 {
        repo.insert(&sample_song(
            &format!("Song {i}"),
            &format!("common keyword line {i}"),
            "G",
            120.0,
        ))
        .await
        .unwrap();
    }
    service.ensure_text_index_synced().await.unwrap();

    let page1 = service
        .list(SongListQuery {
            q: Some("keyword".into()),
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

    let page2 = service
        .list(SongListQuery {
            q: Some("keyword".into()),
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(2),
            last_id: page1.next_last_id.clone(),
            last_rank: page1.next_last_rank,
        })
        .await
        .unwrap();
    assert!(!page2.items.is_empty());
    assert_ne!(page1.items[1].song_id, page2.items[0].song_id);
}

#[tokio::test]
async fn short_query_does_not_fuzzy_match_amin_for_iman() {
    let (_dir, service, repo) = setup_service().await;
    let iman_song = sample_song(
        "Faith Song",
        "hidup oleh iman bukan penglihatan",
        "G",
        120.0,
    );
    let amin_song = sample_song("Amen Song", "pagi siang malam amin amin", "G", 120.0);
    repo.insert(&iman_song).await.unwrap();
    repo.insert(&amin_song).await.unwrap();
    service.ensure_text_index_synced().await.unwrap();

    let page = service
        .list(SongListQuery {
            q: Some("iman".into()),
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(10),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].song_id, iman_song.song_id);
}

#[tokio::test]
async fn rebuild_matches_catalog_count() {
    let (_dir, service, repo) = setup_service().await;
    for i in 0..3 {
        repo.insert(&sample_song(
            &format!("Song {i}"),
            "shared lyrics",
            "G",
            120.0,
        ))
        .await
        .unwrap();
    }
    service.ensure_text_index_synced().await.unwrap();
    let count = service
        .list(SongListQuery {
            q: None,
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(1),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap()
        .total;
    assert_eq!(count, 3);
}
