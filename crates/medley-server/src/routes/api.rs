use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use medley_core::domain::error::AppError;
use medley_core::domain::models::{NewSong, SearchQuery, SongListQuery, SongPatch};
use medley_core::domain::pagination::CursorPage;
use serde::Serialize;

use crate::state::AppState;

struct ApiError(AppError);

impl From<AppError> for ApiError {
    fn from(value: AppError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.0.status_code())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        if status.is_server_error() {
            tracing::error!(status = %status, error = %self.0, "api request failed");
        } else if status.is_client_error() {
            tracing::warn!(status = %status, error = %self.0, "api request rejected");
        }
        let body = serde_json::json!({ "error": self.0.to_string() });
        (status, Json(body)).into_response()
    }
}

pub fn api_router(state: AppState) -> Router {
    Router::new()
        .route("/api/songs", get(list_songs).post(create_song))
        .route(
            "/api/songs/{song_id}",
            get(get_song).patch(update_song).delete(delete_song),
        )
        .route("/api/search", post(search_songs))
        .with_state(state)
}

pub async fn health() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<medley_core::domain::models::SearchResult>,
    total: usize,
}

async fn list_songs(
    State(state): State<AppState>,
    Query(query): Query<SongListQuery>,
) -> Result<Json<CursorPage<medley_core::domain::models::Song>>, ApiError> {
    tracing::info!(?query, "api GET /api/songs");
    let page = state.songs.list(query).await?;
    tracing::info!(
        returned = page.items.len(),
        total = page.total,
        has_more = page.has_more,
        "api GET /api/songs ok"
    );
    Ok(Json(page))
}

async fn get_song(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
) -> Result<Json<medley_core::domain::models::Song>, ApiError> {
    tracing::info!(%song_id, "api GET /api/songs/:id");
    let song = state.songs.get(&song_id).await?;
    tracing::info!(%song_id, title = %song.title, "api GET /api/songs/:id ok");
    Ok(Json(song))
}

async fn create_song(
    State(state): State<AppState>,
    Json(body): Json<NewSong>,
) -> Result<(StatusCode, Json<medley_core::domain::models::Song>), ApiError> {
    tracing::info!(title = %body.title, youtube_url = %body.youtube_url, "api POST /api/songs");
    let song = state.songs.create(body).await?;
    tracing::info!(song_id = %song.song_id, title = %song.title, "api POST /api/songs ok");
    Ok((StatusCode::CREATED, Json(song)))
}

async fn update_song(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
    Json(patch): Json<SongPatch>,
) -> Result<Json<medley_core::domain::models::Song>, ApiError> {
    tracing::info!(%song_id, ?patch, "api PATCH /api/songs/:id");
    let song = state.songs.update(&song_id, patch).await?;
    tracing::info!(%song_id, title = %song.title, "api PATCH /api/songs/:id ok");
    Ok(Json(song))
}

async fn delete_song(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    tracing::info!(%song_id, "api DELETE /api/songs/:id");
    state.songs.delete(&song_id).await?;
    tracing::info!(%song_id, "api DELETE /api/songs/:id ok");
    Ok(StatusCode::NO_CONTENT)
}

async fn search_songs(
    State(state): State<AppState>,
    Json(query): Json<SearchQuery>,
) -> Result<Json<SearchResponse>, ApiError> {
    tracing::info!(
        query = %query.query,
        bpm_min = ?query.bpm_min,
        bpm_max = ?query.bpm_max,
        limit = ?query.limit,
        "api POST /api/search"
    );
    let results = state.search.search(query).await?;
    let total = results.len();
    tracing::info!(total, "api POST /api/search ok");
    Ok(Json(SearchResponse { results, total }))
}
