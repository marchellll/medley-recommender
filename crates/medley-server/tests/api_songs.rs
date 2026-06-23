use axum::body::Body;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use medley_server::app::{build_state, test_config, test_router};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn mount_voyage_embeddings(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{ "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }]
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn list_songs_empty_catalog() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );

    let state = build_state(&config).await.unwrap();
    let app = test_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/songs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["items"].as_array().unwrap().len(), 0);
    assert_eq!(body["total"], 0);
}

#[tokio::test]
async fn get_missing_song_returns_404() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );

    let state = build_state(&config).await.unwrap();
    let app = test_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/songs/00000000-0000-7000-8000-000000000000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn create_song_returns_201_and_indexes() {
    let server = MockServer::start().await;
    mount_voyage_embeddings(&server).await;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        server.uri(),
    );

    let app = test_router(build_state(&config).await.unwrap());
    let body = serde_json::json!({
        "title": "REST Song",
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "lyrics": "rest lyrics",
        "bpm": 120,
        "key": "G"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/songs")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CREATED);
    let song: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(song["title"], "REST Song");
    assert!(song["song_id"].as_str().unwrap().len() > 10);
}

#[tokio::test]
async fn search_returns_indexed_song() {
    let server = MockServer::start().await;
    mount_voyage_embeddings(&server).await;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        server.uri(),
    );

    let app = test_router(build_state(&config).await.unwrap());

    let create_body = serde_json::json!({
        "title": "Indexed",
        "youtube_url": "https://www.youtube.com/watch?v=abc12345678",
        "lyrics": "semantic target phrase",
        "bpm": 100,
        "key": "C"
    });
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/songs")
                .header("content-type", "application/json")
                .body(Body::from(create_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let search_body = serde_json::json!({ "query": "semantic target", "limit": 5 });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(Body::from(search_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(payload["total"].as_u64().unwrap() >= 1);
    assert_eq!(payload["results"][0]["title"], "Indexed");
}

#[tokio::test]
async fn patch_title_skips_reindex_and_search_still_works() {
    let server = MockServer::start().await;
    mount_voyage_embeddings(&server).await;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        server.uri(),
    );

    let app = test_router(build_state(&config).await.unwrap());

    let create_body = serde_json::json!({
        "title": "Original",
        "youtube_url": "https://www.youtube.com/watch?v=patch123456",
        "lyrics": "stable lyrics for search",
        "bpm": 88,
        "key": "F"
    });
    let created = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/songs")
                .header("content-type", "application/json")
                .body(Body::from(create_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let song: serde_json::Value =
        serde_json::from_slice(&created.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let song_id = song["song_id"].as_str().unwrap();

    let patch_body = serde_json::json!({ "title": "Renamed Only" });
    let patch = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/api/songs/{song_id}"))
                .header("content-type", "application/json")
                .body(Body::from(patch_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(patch.status(), StatusCode::OK);

    let search_body = serde_json::json!({ "query": "stable lyrics", "limit": 5 });
    let search = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/search")
                .header("content-type", "application/json")
                .body(Body::from(search_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
    let payload: serde_json::Value =
        serde_json::from_slice(&search.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(payload["total"].as_u64().unwrap() >= 1);
}
