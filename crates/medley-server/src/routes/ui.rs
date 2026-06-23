use askama::Template;
use axum::{
    Form, Router,
    extract::{Path, Query, State},
    response::{Html, IntoResponse, Redirect},
    routing::{get, post},
};
use medley_core::domain::models::{NewSong, SearchQuery, SongListQuery};

use crate::state::AppState;

#[derive(Template)]
#[template(path = "home.html")]
struct HomeTemplate {
    q: String,
    bpm_min: String,
    bpm_max: String,
    results: Vec<medley_core::domain::models::SearchResult>,
    error: Option<String>,
}

#[derive(Template)]
#[template(path = "catalog.html")]
struct CatalogTemplate {
    songs: Vec<CatalogSong>,
    q: String,
    next_last_id: Option<String>,
    next_last_rank: Option<f64>,
    has_more: bool,
    total: i64,
}

struct CatalogSong {
    title: String,
    key: String,
    bpm: f64,
    youtube_url: String,
    edit_url: String,
    delete_url: String,
}

#[derive(Template)]
#[template(path = "song_form.html")]
struct SongFormTemplate {
    heading: String,
    action: String,
    is_edit: bool,
    title: String,
    youtube_url: String,
    lyrics: String,
    bpm: String,
    key: String,
    error: Option<String>,
}

pub fn ui_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/search", post(search_form))
        .route("/catalog", get(catalog))
        .route("/songs/new", get(new_song_form).post(create_song_form))
        .route(
            "/songs/{song_id}/edit",
            get(edit_song_get).post(update_song_post),
        )
        .route("/songs/{song_id}/delete", post(delete_song_post))
        .with_state(state)
}

fn song_edit_url(song_id: &str) -> String {
    format!("/songs/{song_id}/edit")
}

fn song_delete_url(song_id: &str) -> String {
    format!("/songs/{song_id}/delete")
}

async fn edit_song_get(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
) -> impl IntoResponse {
    edit_song_form(state, song_id).await
}

async fn update_song_post(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    update_song_form(state, song_id, form).await
}

async fn delete_song_post(
    State(state): State<AppState>,
    Path(song_id): Path<String>,
) -> impl IntoResponse {
    delete_song_form(state, song_id).await
}

fn parse_optional_f64(raw: &str) -> Result<Option<f64>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse()
        .map(Some)
        .map_err(|_| format!("\"{trimmed}\" is not a valid number"))
}

fn parse_required_f64(raw: &str, field: &str) -> Result<f64, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} is required"));
    }
    trimmed
        .parse()
        .map_err(|_| format!("{field} must be a number"))
}

async fn home() -> Html<String> {
    Html(
        HomeTemplate {
            q: String::new(),
            bpm_min: String::new(),
            bpm_max: String::new(),
            results: vec![],
            error: None,
        }
        .render()
        .unwrap_or_else(|e| e.to_string()),
    )
}

#[derive(serde::Deserialize, Default)]
struct SearchForm {
    #[serde(default)]
    q: String,
    #[serde(default)]
    bpm_min: String,
    #[serde(default)]
    bpm_max: String,
}

async fn search_form(
    State(state): State<AppState>,
    Form(form): Form<SearchForm>,
) -> Html<String> {
    let bpm_min = match parse_optional_f64(&form.bpm_min) {
        Ok(v) => v,
        Err(err) => {
            return Html(
                HomeTemplate {
                    q: form.q,
                    bpm_min: form.bpm_min,
                    bpm_max: form.bpm_max,
                    results: vec![],
                    error: Some(format!("BPM min: {err}")),
                }
                .render()
                .unwrap_or_else(|e| e.to_string()),
            );
        }
    };
    let bpm_max = match parse_optional_f64(&form.bpm_max) {
        Ok(v) => v,
        Err(err) => {
            return Html(
                HomeTemplate {
                    q: form.q,
                    bpm_min: form.bpm_min,
                    bpm_max: form.bpm_max,
                    results: vec![],
                    error: Some(format!("BPM max: {err}")),
                }
                .render()
                .unwrap_or_else(|e| e.to_string()),
            );
        }
    };

    if form.q.trim().is_empty() {
        tracing::warn!("ui POST /search empty query");
        return Html(
            HomeTemplate {
                q: form.q,
                bpm_min: form.bpm_min,
                bpm_max: form.bpm_max,
                results: vec![],
                error: Some("Enter a search query".into()),
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
        );
    }

    let query = SearchQuery {
        query: form.q.clone(),
        bpm_min,
        bpm_max,
        keys: None,
        limit: Some(20),
    };
    tracing::info!(
        query = %form.q,
        bpm_min = ?bpm_min,
        bpm_max = ?bpm_max,
        "ui POST /search"
    );
    match state.search.search(query).await {
        Ok(results) => {
            tracing::info!(count = results.len(), "ui POST /search ok");
            Html(
            HomeTemplate {
                q: form.q,
                bpm_min: form.bpm_min,
                bpm_max: form.bpm_max,
                results,
                error: None,
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
            )
        }
        Err(err) => {
            tracing::warn!(error = %err, "ui POST /search failed");
            Html(
            HomeTemplate {
                q: form.q,
                bpm_min: form.bpm_min,
                bpm_max: form.bpm_max,
                results: vec![],
                error: Some(err.to_string()),
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
            )
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct CatalogQuery {
    q: Option<String>,
    last_id: Option<String>,
    last_rank: Option<f64>,
}

async fn catalog(
    State(state): State<AppState>,
    Query(query): Query<CatalogQuery>,
) -> Html<String> {
    tracing::info!(?query, "ui GET /catalog");
    let list_query = SongListQuery {
        q: query.q.clone(),
        key: None,
        bpm_min: None,
        bpm_max: None,
        limit: Some(20),
        last_id: query.last_id,
        last_rank: query.last_rank,
    };
    let page = state.songs.list(list_query).await.unwrap_or_default_page();
    let songs: Vec<CatalogSong> = page
        .items
        .into_iter()
        .map(|s| CatalogSong {
            title: s.title,
            key: s.key,
            bpm: s.bpm,
            youtube_url: s.youtube_url,
            edit_url: song_edit_url(&s.song_id),
            delete_url: song_delete_url(&s.song_id),
        })
        .collect();
    tracing::info!(
        returned = songs.len(),
        total = page.total,
        has_more = page.has_more,
        "ui GET /catalog ok"
    );
    Html(
        CatalogTemplate {
            songs,
            q: query.q.unwrap_or_default(),
            next_last_id: page.next_last_id,
            next_last_rank: page.next_last_rank,
            has_more: page.has_more,
            total: page.total,
        }
        .render()
        .unwrap_or_else(|e| e.to_string()),
    )
}

trait DefaultPage {
    fn unwrap_or_default_page(
        self,
    ) -> medley_core::domain::pagination::CursorPage<medley_core::domain::models::Song>;
}

impl DefaultPage
    for Result<
        medley_core::domain::pagination::CursorPage<medley_core::domain::models::Song>,
        medley_core::domain::error::AppError,
    >
{
    fn unwrap_or_default_page(
        self,
    ) -> medley_core::domain::pagination::CursorPage<medley_core::domain::models::Song> {
        self.unwrap_or(medley_core::domain::pagination::CursorPage {
            items: vec![],
            limit: 20,
            next_last_id: None,
            next_last_rank: None,
            has_more: false,
            total: 0,
        })
    }
}

fn new_song_template(error: Option<String>) -> SongFormTemplate {
    SongFormTemplate {
        heading: "Add song".into(),
        action: "/songs/new".into(),
        is_edit: false,
        title: String::new(),
        youtube_url: String::new(),
        lyrics: String::new(),
        bpm: String::new(),
        key: "C".into(),
        error,
    }
}

async fn new_song_form() -> Html<String> {
    Html(
        new_song_template(None)
            .render()
            .unwrap_or_else(|e| e.to_string()),
    )
}

#[derive(serde::Deserialize, Default)]
struct SongForm {
    #[serde(default)]
    title: String,
    #[serde(default)]
    youtube_url: String,
    #[serde(default)]
    lyrics: String,
    #[serde(default)]
    bpm: String,
    #[serde(default)]
    key: String,
}

fn form_template(form: &SongForm) -> SongFormTemplate {
    SongFormTemplate {
        heading: "Add song".into(),
        action: "/songs/new".into(),
        is_edit: false,
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: form.key.clone(),
        error: None,
    }
}

#[allow(clippy::result_large_err)]
fn validate_song_form(form: &SongForm) -> Result<NewSong, (SongFormTemplate, String)> {
    let bpm = parse_required_f64(&form.bpm, "BPM").map_err(|e| (form_template(form), e))?;

    if form.title.trim().is_empty() {
        return Err((form_template(form), "Title is required".into()));
    }

    Ok(NewSong {
        title: form.title.trim().to_string(),
        youtube_url: form.youtube_url.trim().to_string(),
        lyrics: form.lyrics.trim().to_string(),
        bpm,
        key: form.key.trim().to_string(),
    })
}

async fn create_song_form(
    State(state): State<AppState>,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    tracing::info!(title = %form.title, youtube_url = %form.youtube_url, "ui POST /songs/new");
    let new_song = match validate_song_form(&form) {
        Ok(v) => v,
        Err((mut tpl, err)) => {
            tpl.error = Some(err);
            return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response();
        }
    };

    match state.songs.create(new_song).await {
        Ok(song) => {
            tracing::info!(song_id = %song.song_id, "ui POST /songs/new ok");
            Redirect::to("/catalog").into_response()
        }
        Err(err) => {
            tracing::warn!(error = %err, "ui POST /songs/new failed");
            let mut tpl = form_template(&form);
            tpl.error = Some(err.to_string());
            Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    }
}

async fn edit_song_form(state: AppState, song_id: String) -> impl IntoResponse {
    tracing::info!(%song_id, "ui GET /songs/:id/edit");
    match state.songs.get(&song_id).await {
        Ok(song) => Html(
            SongFormTemplate {
                heading: format!("Edit: {}", song.title),
                action: song_edit_url(&song_id),
                is_edit: true,
                title: song.title,
                youtube_url: song.youtube_url,
                lyrics: song.lyrics,
                bpm: song.bpm.to_string(),
                key: song.key,
                error: None,
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
        )
        .into_response(),
        Err(err) => (axum::http::StatusCode::NOT_FOUND, err.to_string()).into_response(),
    }
}

fn edit_song_template(song_id: &str, form: &SongForm, error: Option<String>) -> SongFormTemplate {
    SongFormTemplate {
        heading: "Edit song".into(),
        action: song_edit_url(song_id),
        is_edit: true,
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: form.key.clone(),
        error,
    }
}

async fn update_song_form(
    state: AppState,
    song_id: String,
    form: SongForm,
) -> impl IntoResponse {
    tracing::info!(%song_id, title = %form.title, "ui POST /songs/:id/edit");
    let bpm = match parse_required_f64(&form.bpm, "BPM") {
        Ok(v) => v,
        Err(err) => {
            let tpl = edit_song_template(&song_id, &form, Some(err));
            return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response();
        }
    };

    if form.title.trim().is_empty() {
        let tpl = edit_song_template(&song_id, &form, Some("Title is required".into()));
        return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response();
    }

    let patch = medley_core::domain::models::SongPatch {
        title: Some(form.title.trim().to_string()),
        youtube_url: Some(form.youtube_url.trim().to_string()),
        lyrics: Some(form.lyrics.trim().to_string()),
        bpm: Some(bpm),
        key: Some(form.key.trim().to_string()),
    };
    match state.songs.update(&song_id, patch).await {
        Ok(_) => {
            tracing::info!(%song_id, "ui POST /songs/:id/edit ok");
            Redirect::to("/catalog").into_response()
        }
        Err(err) => {
            tracing::warn!(%song_id, error = %err, "ui POST /songs/:id/edit failed");
            let tpl = edit_song_template(&song_id, &form, Some(err.to_string()));
            Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    }
}

async fn delete_song_form(state: AppState, song_id: String) -> impl IntoResponse {
    tracing::info!(%song_id, "ui POST /songs/:id/delete");
    match state.songs.delete(&song_id).await {
        Ok(()) => {
            tracing::info!(%song_id, "ui POST /songs/:id/delete ok");
            Redirect::to("/catalog").into_response()
        }
        Err(err) => {
            tracing::warn!(%song_id, error = %err, "ui POST /songs/:id/delete failed");
            (axum::http::StatusCode::BAD_REQUEST, err.to_string()).into_response()
        }
    }
}
