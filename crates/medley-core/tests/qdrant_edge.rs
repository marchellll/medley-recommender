use medley_core::domain::id::new_song_id;
use medley_core::index::qdrant_edge::EdgeVectorIndex;
use medley_core::index::VectorIndex;
use tempfile::TempDir;

#[tokio::test]
async fn edge_upsert_search_delete() {
    let dir = TempDir::new().unwrap();
    let index = EdgeVectorIndex::open(&dir.path().join("shard"), 4).unwrap();
    let song_id = new_song_id();

    let vector = vec![1.0, 0.0, 0.0, 0.0];
    index
        .upsert(&song_id, vector.clone(), "G", 120.0)
        .await
        .unwrap();

    let hits = index.search(&vector, 5, None, None, None).await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].song_id, song_id);

    index.delete(&song_id).await.unwrap();
    index.flush().await.unwrap();
}
