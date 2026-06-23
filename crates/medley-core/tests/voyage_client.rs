use medley_core::domain::error::AppError;
use medley_core::embed::voyage::VoyageClient;
use medley_core::embed::{EmbeddingProvider, InputType};
use wiremock::matchers::{bearer_token, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn voyage_embeds_and_normalizes_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(bearer_token("test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{ "embedding": [3.0, 4.0] }]
        })))
        .mount(&server)
        .await;

    let client = VoyageClient::new(
        "test-key".into(),
        "voyage-4-large".into(),
        2,
        server.uri(),
    );

    let vectors = client
        .embed(&["hello".into()], InputType::Document)
        .await
        .unwrap();

    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 2);
    let norm: f32 = vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn voyage_surfaces_api_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid key"))
        .mount(&server)
        .await;

    let client = VoyageClient::new(
        "bad-key".into(),
        "voyage-4-large".into(),
        4,
        server.uri(),
    );

    let err = client
        .embed(&["hello".into()], InputType::Query)
        .await
        .unwrap_err();
    assert!(matches!(err, AppError::Embedding(_)));
}

#[tokio::test]
async fn voyage_requires_api_key() {
    let client = VoyageClient::new(
        String::new(),
        "voyage-4-large".into(),
        4,
        "http://127.0.0.1:9".into(),
    );

    let err = client
        .embed(&["hello".into()], InputType::Query)
        .await
        .unwrap_err();
    assert!(matches!(err, AppError::Embedding(_)));
}
