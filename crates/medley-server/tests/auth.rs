use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use medley_server::app::{admin_bearer, build_state, test_config, test_router};
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

fn test_app(state: medley_server::state::AppState) -> axum::Router {
    test_router(state)
}

async fn oneshot(app: axum::Router, request: Request<Body>) -> http::Response<axum::body::Body> {
    app.oneshot(request).await.unwrap()
}

#[tokio::test]
async fn login_rejects_wrong_token() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_app(build_state(&config).await.unwrap());

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/api/auth/login")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"token":"wrong"}"#))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn login_returns_jwt_for_admin_token() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_app(build_state(&config).await.unwrap());

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/api/auth/login")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"token":"test-admin-token"}"#))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(body["token"].as_str().unwrap().len() > 20);
}

#[tokio::test]
async fn create_song_requires_admin_jwt() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_app(build_state(&config).await.unwrap());

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/api/songs")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "title": "Blocked",
                    "youtube_url": "https://www.youtube.com/watch?v=blocked1234",
                    "lyrics": "nope",
                    "bpm": 100,
                    "key": "C"
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn create_song_with_admin_jwt_succeeds() {
    let server = MockServer::start().await;
    mount_voyage_embeddings(&server).await;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        server.uri(),
    );
    let state = build_state(&config).await.unwrap();
    let auth = admin_bearer(&state);
    let app = test_app(state);

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/api/songs")
            .header("content-type", "application/json")
            .header("authorization", auth)
            .body(Body::from(
                serde_json::json!({
                    "title": "Authed",
                    "youtube_url": "https://www.youtube.com/watch?v=authed12345",
                    "lyrics": "auth lyrics",
                    "bpm": 100,
                    "key": "C"
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn rate_limits_unauthenticated_queries() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_app(build_state(&config).await.unwrap());

    for _ in 0..30 {
        let response = oneshot(
            app.clone(),
            Request::builder()
                .uri("/api/songs")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 1], 1234))))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let response = oneshot(
        app,
        Request::builder()
            .uri("/api/songs")
            .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 1], 1234))))
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn admin_jwt_bypasses_rate_limit() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let state = build_state(&config).await.unwrap();
    let auth = admin_bearer(&state);
    let app = test_app(state);

    for _ in 0..12 {
        let response = oneshot(
            app.clone(),
            Request::builder()
                .uri("/api/songs")
                .header("authorization", auth.clone())
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 2], 1234))))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn mcp_middleware_blocks_mutation_without_api_token() {
    use axum::{middleware, Router};
    use medley_server::rate_limit::mcp_auth_middleware;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let state = build_state(&config).await.unwrap();

    let app = Router::new()
        .route("/mcp", axum::routing::post(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            mcp_auth_middleware,
        ))
        .with_state(state);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "add_song" }
    });

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn mcp_middleware_allows_mutation_with_api_token() {
    use axum::{middleware, Router};
    use medley_server::rate_limit::mcp_auth_middleware;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let state = build_state(&config).await.unwrap();

    let app = Router::new()
        .route("/mcp", axum::routing::post(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            mcp_auth_middleware,
        ))
        .with_state(state);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "add_song" }
    });

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", "Bearer test-api-token")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn mcp_rate_limits_even_with_api_token() {
    use axum::{middleware, Router};
    use medley_server::rate_limit::mcp_auth_middleware;

    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let state = build_state(&config).await.unwrap();

    let app = Router::new()
        .route("/mcp", axum::routing::post(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            mcp_auth_middleware,
        ))
        .with_state(state);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": "search_songs" }
    })
    .to_string();

    for _ in 0..10 {
        let response = oneshot(
            app.clone(),
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", "Bearer test-api-token")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 3], 1234))))
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    let response = oneshot(
        app,
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", "Bearer test-api-token")
            .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 3], 1234))))
            .body(Body::from(body))
            .unwrap(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}
