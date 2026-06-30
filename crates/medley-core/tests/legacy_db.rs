use std::path::Path;

use medley_core::repo::sqlite::SqliteSongRepository;
use medley_core::repo::SongRepository;

#[tokio::test]
#[ignore = "requires committed data/medley.db"]
async fn migrate_preserves_existing_medley_db() {
    let source = Path::new("data/medley.db");
    if !source.exists() {
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("medley.db");
    std::fs::copy(source, &db_path).unwrap();

    let count_before: (i64,) = sqlx::query_as("SELECT COUNT(1) FROM songs")
        .fetch_one(
            &sqlx::sqlite::SqlitePoolOptions::new()
                .connect(&format!("sqlite:{}?mode=ro", db_path.display()))
                .await
                .unwrap(),
        )
        .await
        .unwrap();

    let repo = SqliteSongRepository::connect(&db_path).await.unwrap();
    repo.migrate().await.unwrap();

    let count_after: (i64,) = sqlx::query_as("SELECT COUNT(1) FROM songs")
        .fetch_one(repo.pool())
        .await
        .unwrap();

    assert_eq!(count_before.0, count_after.0);
    assert!(count_after.0 > 0);

    let page = repo
        .list(&medley_core::domain::models::SongListQuery {
            q: None,
            key: None,
            bpm_min: None,
            bpm_max: None,
            limit: Some(1),
            last_id: None,
            last_rank: None,
        })
        .await
        .unwrap();
    assert!(
        !page.items.is_empty(),
        "expected songs to remain readable after migrate"
    );
    assert!(
        uuid::Uuid::parse_str(&page.items[0].song_id).is_ok(),
        "expected UUID v7 song ids after migrate"
    );
}
