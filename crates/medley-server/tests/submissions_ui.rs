use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use medley_server::app::{admin_cookie, build_state, test_config, test_ui_router};
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

fn contribute_form_body(video_id: &str) -> String {
    format!(
        "title=Public+Song&youtube_url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D{video_id}&lyrics=hello&bpm=120&key=G"
    )
}

#[tokio::test]
async fn public_contribute_submission() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let state = build_state(&config).await.unwrap();
    let app = test_ui_router(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/contribute")
                .header("content-type", "application/x-www-form-urlencoded")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 50], 1234))))
                .body(Body::from(contribute_form_body("pub00000001")))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SEE_OTHER);
    let location = response
        .headers()
        .get("location")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(location.contains("submitted=1"));
}

#[tokio::test]
async fn contribution_rate_limiter_allows_multiple_before_throttle() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_ui_router(build_state(&config).await.unwrap());

    let body = contribute_form_body("rat00000001");
    let addr = ConnectInfo(SocketAddr::from(([10, 0, 0, 51], 1234)));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/contribute")
                .header("content-type", "application/x-www-form-urlencoded")
                .extension(addr)
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SEE_OTHER);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/contribute")
                .header("content-type", "application/x-www-form-urlencoded")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 51], 1234))))
                .body(Body::from(contribute_form_body("rat00000002")))
                .unwrap(),
        )
        .await
        .unwrap();
    // submission_rate_limit is 5/min, so 2nd request from same IP still passes
    assert_eq!(response.status(), StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn admin_submissions_requires_login() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_ui_router(build_state(&config).await.unwrap());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/submissions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SEE_OTHER);
}

#[tokio::test]
async fn approve_submission_creates_catalog_song() {
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
    let cookie = admin_cookie(&state);
    let app = test_ui_router(state.clone());

    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/contribute")
                .header("content-type", "application/x-www-form-urlencoded")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 52], 1234))))
                .body(Body::from(contribute_form_body("adm00000001")))
                .unwrap(),
        )
        .await
        .unwrap();

    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/submissions")
                .header("cookie", cookie.clone())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), StatusCode::OK);
    let html = String::from_utf8(
        list_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(html.contains("Public Song"));

    let detail_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/submissions")
                .header("cookie", cookie.clone())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let detail_html = String::from_utf8(
        detail_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    let review_path = detail_html
        .split("href=\"/submissions/")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("review link");
    let submission_id = review_path.trim_end_matches("/edit").trim_end_matches('"');

    let approve_body = contribute_form_body("adm00000001").replace("Public+Song", "Approved+Song");
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/submissions/{submission_id}/approve"))
                .header("content-type", "application/x-www-form-urlencoded")
                .header("cookie", cookie.clone())
                .body(Body::from(approve_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SEE_OTHER);

    let catalog = app
        .oneshot(
            Request::builder()
                .uri("/catalog")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let catalog_html = String::from_utf8(
        catalog
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(catalog_html.contains("Approved Song"));
}

#[tokio::test]
async fn contribute_validation_rerenders_form() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let edge_dir = tempfile::TempDir::new().unwrap();
    let config = test_config(
        db_dir.path().join("medley.db"),
        edge_dir.path().join("edge_shard"),
        "http://127.0.0.1:9",
    );
    let app = test_ui_router(build_state(&config).await.unwrap());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/contribute")
                .header("content-type", "application/x-www-form-urlencoded")
                .extension(ConnectInfo(SocketAddr::from(([10, 0, 0, 53], 1234))))
                .body(Body::from(
                    "title=Bad&youtube_url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Dbad00000001&lyrics=hi&bpm=120&key=H",
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let html = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(html.contains("key must be one of"));
}
