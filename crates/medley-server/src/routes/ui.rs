use askama::Template;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::Request,
    http::{header, HeaderMap, StatusCode},
    middleware,
    middleware::Next,
    response::{Html, IntoResponse, Redirect, Response},
    routing::{get, post},
    Form, Router,
};
use medley_core::domain::models::{NewSong, SearchQuery, SongListQuery, SubmissionListQuery};

use crate::auth::{admin_cookie_value, clear_admin_cookie, optional_admin_from_request, AdminUser};
use crate::rate_limit::{submission_rate_limit_middleware, ui_rate_limit_middleware};
use crate::state::AppState;

#[derive(Template)]
#[template(path = "home.html")]
struct HomeTemplate {
    q: String,
    bpm_min: String,
    bpm_max: String,
    results: Vec<medley_core::domain::models::SearchResult>,
    error: Option<String>,
    is_admin: bool,
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
    is_admin: bool,
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
    subtitle: String,
    action: String,
    submit_label: String,
    cancel_href: String,
    title: String,
    youtube_url: String,
    lyrics: String,
    bpm: String,
    key: String,
    error: Option<String>,
    success: Option<String>,
    is_admin: bool,
    contribute_active: bool,
}

#[derive(Template)]
#[template(path = "submissions.html")]
struct SubmissionsTemplate {
    submissions: Vec<SubmissionRow>,
    next_last_id: Option<String>,
    has_more: bool,
    total: i64,
}

struct SubmissionRow {
    title: String,
    key: String,
    bpm: f64,
    youtube_url: String,
    submitted_at: String,
    review_url: String,
    delete_url: String,
}

#[derive(Template)]
#[template(path = "submission_detail.html")]
struct SubmissionDetailTemplate {
    submitted_at: String,
    title: String,
    youtube_url: String,
    lyrics: String,
    bpm: String,
    key: String,
    edit_action: String,
    approve_action: String,
    delete_action: String,
    error: Option<String>,
    success: Option<String>,
}

#[derive(Template)]
#[template(path = "login.html")]
struct LoginTemplate {
    error: Option<String>,
}

pub fn ui_router(state: AppState) -> Router {
    let public = Router::new()
        .route("/", get(home))
        .route("/search", post(search_form))
        .route("/catalog", get(catalog))
        .route("/contribute", get(contribute_form))
        .route("/login", get(login_form).post(login_post))
        .route("/logout", post(logout_post))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ui_rate_limit_middleware,
        ));

    let contribute_post = Router::new()
        .route("/contribute", post(contribute_post_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            submission_rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            ui_rate_limit_middleware,
        ));

    let admin = Router::new()
        .route("/songs/new", get(new_song_form).post(create_song_form))
        .route(
            "/songs/{song_id}/edit",
            get(edit_song_get).post(update_song_post),
        )
        .route("/songs/{song_id}/delete", post(delete_song_post))
        .route("/submissions", get(submissions_list))
        .route("/submissions/{submission_id}", get(submission_detail_get))
        .route(
            "/submissions/{submission_id}/edit",
            post(submission_edit_post),
        )
        .route(
            "/submissions/{submission_id}/approve",
            post(submission_approve_post),
        )
        .route(
            "/submissions/{submission_id}/delete",
            post(submission_delete_post),
        )
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_admin_ui_middleware,
        ));

    Router::new()
        .merge(public)
        .merge(contribute_post)
        .merge(admin)
        .with_state(state)
}

async fn require_admin_ui_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if optional_admin_from_request(&req, &state.admin_auth) {
        return next.run(req).await;
    }
    Redirect::to("/login").into_response()
}

fn is_admin(headers: &HeaderMap, state: &AppState) -> bool {
    state.is_admin_from_cookie_header(
        headers
            .get(header::COOKIE)
            .and_then(|value| value.to_str().ok()),
    )
}

fn song_edit_url(song_id: &str) -> String {
    format!("/songs/{song_id}/edit")
}

fn song_delete_url(song_id: &str) -> String {
    format!("/songs/{song_id}/delete")
}

fn submission_review_url(submission_id: &str) -> String {
    format!("/submissions/{submission_id}")
}

fn submission_delete_url(submission_id: &str) -> String {
    format!("/submissions/{submission_id}/delete")
}

fn submission_edit_url(submission_id: &str) -> String {
    format!("/submissions/{submission_id}/edit")
}

fn submission_approve_url(submission_id: &str) -> String {
    format!("/submissions/{submission_id}/approve")
}

async fn edit_song_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(song_id): Path<String>,
) -> impl IntoResponse {
    edit_song_form(state, headers, song_id).await
}

async fn update_song_post(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(song_id): Path<String>,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    update_song_form(state, song_id, form).await
}

async fn delete_song_post(
    State(state): State<AppState>,
    _admin: AdminUser,
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

async fn home(headers: HeaderMap, State(state): State<AppState>) -> Html<String> {
    Html(
        HomeTemplate {
            q: String::new(),
            bpm_min: String::new(),
            bpm_max: String::new(),
            results: vec![],
            error: None,
            is_admin: is_admin(&headers, &state),
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
    headers: HeaderMap,
    Form(form): Form<SearchForm>,
) -> Html<String> {
    let admin = is_admin(&headers, &state);
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
                    is_admin: admin,
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
                    is_admin: admin,
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
                is_admin: admin,
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
                    is_admin: admin,
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
                    is_admin: admin,
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
    headers: HeaderMap,
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
            is_admin: is_admin(&headers, &state),
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

async fn login_form() -> Html<String> {
    Html(
        LoginTemplate { error: None }
            .render()
            .unwrap_or_else(|e| e.to_string()),
    )
}

#[derive(serde::Deserialize, Default)]
struct LoginForm {
    #[serde(default)]
    token: String,
}

async fn login_post(
    State(state): State<AppState>,
    Form(form): Form<LoginForm>,
) -> impl IntoResponse {
    if !state.admin_auth.enabled() {
        return Html(
            LoginTemplate {
                error: Some("ADMIN_TOKEN is not set".into()),
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
        )
        .into_response();
    }
    if !state.admin_auth.verify_login(&form.token) {
        return Html(
            LoginTemplate {
                error: Some("Invalid admin token".into()),
            }
            .render()
            .unwrap_or_else(|e| e.to_string()),
        )
        .into_response();
    }

    let jwt = match state.admin_auth.issue_jwt() {
        Ok(token) => token,
        Err(err) => {
            return Html(
                LoginTemplate {
                    error: Some(err.to_string()),
                }
                .render()
                .unwrap_or_else(|e| e.to_string()),
            )
            .into_response();
        }
    };

    (
        StatusCode::SEE_OTHER,
        [(header::SET_COOKIE, admin_cookie_value(&jwt))],
        Redirect::to("/catalog"),
    )
        .into_response()
}

async fn logout_post() -> impl IntoResponse {
    (
        StatusCode::SEE_OTHER,
        [(header::SET_COOKIE, clear_admin_cookie())],
        Redirect::to("/"),
    )
}

fn new_song_template(error: Option<String>) -> SongFormTemplate {
    SongFormTemplate {
        heading: "Add song".into(),
        subtitle: "Add a new song to the catalog.".into(),
        action: "/songs/new".into(),
        submit_label: "Save song".into(),
        cancel_href: "/catalog".into(),
        title: String::new(),
        youtube_url: String::new(),
        lyrics: String::new(),
        bpm: String::new(),
        key: "C".into(),
        error,
        success: None,
        is_admin: true,
        contribute_active: false,
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
        subtitle: "Add a new song to the catalog.".into(),
        action: "/songs/new".into(),
        submit_label: "Save song".into(),
        cancel_href: "/catalog".into(),
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: form.key.clone(),
        error: None,
        success: None,
        is_admin: true,
        contribute_active: false,
    }
}

fn contribute_template(
    form: &SongForm,
    error: Option<String>,
    success: Option<String>,
    is_admin: bool,
) -> SongFormTemplate {
    SongFormTemplate {
        heading: "Contribute a song".into(),
        subtitle: "Suggest a song for the catalog. Submissions are reviewed before publishing."
            .into(),
        action: "/contribute".into(),
        submit_label: "Submit for review".into(),
        cancel_href: "/".into(),
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: if form.key.is_empty() {
            "C".into()
        } else {
            form.key.clone()
        },
        error,
        success,
        is_admin,
        contribute_active: true,
    }
}

fn parse_song_form(form: &SongForm) -> Result<NewSong, String> {
    let bpm = parse_required_f64(&form.bpm, "BPM")?;
    if form.title.trim().is_empty() {
        return Err("Title is required".into());
    }
    Ok(NewSong {
        title: form.title.trim().to_string(),
        youtube_url: form.youtube_url.trim().to_string(),
        lyrics: form.lyrics.trim().to_string(),
        bpm,
        key: form.key.trim().to_string(),
    })
}

#[allow(clippy::result_large_err)]
fn validate_song_form(form: &SongForm) -> Result<NewSong, (SongFormTemplate, String)> {
    parse_song_form(form).map_err(|e| (form_template(form), e))
}

#[allow(clippy::result_large_err)]
fn validate_contribute_form(
    form: &SongForm,
    is_admin: bool,
) -> Result<NewSong, (SongFormTemplate, String)> {
    parse_song_form(form).map_err(|e| {
        (
            contribute_template(form, Some(e.clone()), None, is_admin),
            e,
        )
    })
}

async fn create_song_form(
    State(state): State<AppState>,
    _admin: AdminUser,
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

async fn edit_song_form(state: AppState, headers: HeaderMap, song_id: String) -> impl IntoResponse {
    tracing::info!(%song_id, "ui GET /songs/:id/edit");
    match state.songs.get(&song_id).await {
        Ok(song) => Html(
            SongFormTemplate {
                heading: format!("Edit: {}", song.title),
                subtitle: "Update song details and lyrics.".into(),
                action: song_edit_url(&song_id),
                submit_label: "Save song".into(),
                cancel_href: "/catalog".into(),
                title: song.title,
                youtube_url: song.youtube_url,
                lyrics: song.lyrics,
                bpm: song.bpm.to_string(),
                key: song.key,
                error: None,
                success: None,
                is_admin: is_admin(&headers, &state),
                contribute_active: false,
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
        subtitle: "Update song details and lyrics.".into(),
        action: song_edit_url(song_id),
        submit_label: "Save song".into(),
        cancel_href: "/catalog".into(),
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: form.key.clone(),
        error,
        success: None,
        is_admin: true,
        contribute_active: false,
    }
}

async fn update_song_form(state: AppState, song_id: String, form: SongForm) -> impl IntoResponse {
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

#[derive(serde::Deserialize, Default)]
struct ContributeQuery {
    #[serde(default)]
    submitted: Option<String>,
}

async fn contribute_form(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ContributeQuery>,
) -> Html<String> {
    let success = query
        .submitted
        .as_deref()
        .is_some_and(|v| v == "1")
        .then(|| "Thanks! Your submission is in the review queue.".into());
    Html(
        contribute_template(
            &SongForm::default(),
            None,
            success,
            is_admin(&headers, &state),
        )
        .render()
        .unwrap_or_else(|e| e.to_string()),
    )
}

async fn contribute_post_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    let admin = is_admin(&headers, &state);
    tracing::info!(title = %form.title, youtube_url = %form.youtube_url, "ui POST /contribute");
    let new_song = match validate_contribute_form(&form, admin) {
        Ok(v) => v,
        Err((tpl, _)) => {
            return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    };

    match state.submissions.submit(new_song).await {
        Ok(submission) => {
            tracing::info!(submission_id = %submission.submission_id, "ui POST /contribute ok");
            Redirect::to("/contribute?submitted=1").into_response()
        }
        Err(err) => {
            tracing::warn!(error = %err, "ui POST /contribute failed");
            let tpl = contribute_template(&form, Some(err.to_string()), None, admin);
            Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct SubmissionsQuery {
    last_id: Option<String>,
}

async fn submissions_list(
    State(state): State<AppState>,
    Query(query): Query<SubmissionsQuery>,
) -> Html<String> {
    let list_query = SubmissionListQuery {
        limit: Some(20),
        last_id: query.last_id,
    };
    let page = state
        .submissions
        .list(list_query)
        .await
        .unwrap_or_else(|_| medley_core::domain::pagination::CursorPage {
            items: vec![],
            limit: 20,
            next_last_id: None,
            next_last_rank: None,
            has_more: false,
            total: 0,
        });
    let submissions: Vec<SubmissionRow> = page
        .items
        .into_iter()
        .map(|s| SubmissionRow {
            title: s.title,
            key: s.key,
            bpm: s.bpm,
            youtube_url: s.youtube_url,
            submitted_at: s.submitted_at.to_rfc3339(),
            review_url: submission_review_url(&s.submission_id),
            delete_url: submission_delete_url(&s.submission_id),
        })
        .collect();
    Html(
        SubmissionsTemplate {
            submissions,
            next_last_id: page.next_last_id,
            has_more: page.has_more,
            total: page.total,
        }
        .render()
        .unwrap_or_else(|e| e.to_string()),
    )
}

fn submission_detail_template(
    submission_id: &str,
    submitted_at: &str,
    form: &SongForm,
    error: Option<String>,
    success: Option<String>,
) -> SubmissionDetailTemplate {
    SubmissionDetailTemplate {
        submitted_at: submitted_at.into(),
        title: form.title.clone(),
        youtube_url: form.youtube_url.clone(),
        lyrics: form.lyrics.clone(),
        bpm: form.bpm.clone(),
        key: form.key.clone(),
        edit_action: submission_edit_url(submission_id),
        approve_action: submission_approve_url(submission_id),
        delete_action: submission_delete_url(submission_id),
        error,
        success,
    }
}

fn submission_from_record(
    submission: &medley_core::domain::models::SongSubmission,
) -> (String, SongForm) {
    (
        submission.submitted_at.to_rfc3339(),
        SongForm {
            title: submission.title.clone(),
            youtube_url: submission.youtube_url.clone(),
            lyrics: submission.lyrics.clone(),
            bpm: submission.bpm.to_string(),
            key: submission.key.clone(),
        },
    )
}

async fn submission_detail_get(
    State(state): State<AppState>,
    Path(submission_id): Path<String>,
    Query(query): Query<ContributeQuery>,
) -> impl IntoResponse {
    match state.submissions.get(&submission_id).await {
        Ok(submission) => {
            let (submitted_at, form) = submission_from_record(&submission);
            let success = query
                .submitted
                .as_deref()
                .is_some_and(|v| v == "saved")
                .then(|| "Changes saved.".into());
            Html(
                submission_detail_template(&submission_id, &submitted_at, &form, None, success)
                    .render()
                    .unwrap_or_else(|e| e.to_string()),
            )
            .into_response()
        }
        Err(err) => (StatusCode::NOT_FOUND, err.to_string()).into_response(),
    }
}

async fn submission_edit_post(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(submission_id): Path<String>,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    let submitted_at = match state.submissions.get(&submission_id).await {
        Ok(s) => s.submitted_at.to_rfc3339(),
        Err(err) => return (StatusCode::NOT_FOUND, err.to_string()).into_response(),
    };

    let new_song = match parse_song_form(&form) {
        Ok(v) => v,
        Err(err) => {
            let tpl =
                submission_detail_template(&submission_id, &submitted_at, &form, Some(err), None);
            return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response();
        }
    };

    match state.submissions.update(&submission_id, new_song).await {
        Ok(()) => Redirect::to(&format!("/submissions/{submission_id}?saved=1")).into_response(),
        Err(err) => {
            let tpl = submission_detail_template(
                &submission_id,
                &submitted_at,
                &form,
                Some(err.to_string()),
                None,
            );
            Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    }
}

async fn submission_approve_post(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(submission_id): Path<String>,
    Form(form): Form<SongForm>,
) -> impl IntoResponse {
    let submitted_at = match state.submissions.get(&submission_id).await {
        Ok(s) => s.submitted_at.to_rfc3339(),
        Err(err) => return (StatusCode::NOT_FOUND, err.to_string()).into_response(),
    };

    let new_song = match parse_song_form(&form) {
        Ok(v) => v,
        Err(err) => {
            let tpl =
                submission_detail_template(&submission_id, &submitted_at, &form, Some(err), None);
            return Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response();
        }
    };

    match state
        .submissions
        .approve(&submission_id, new_song, state.songs.as_ref())
        .await
    {
        Ok(song) => {
            tracing::info!(song_id = %song.song_id, "ui POST /submissions/:id/approve ok");
            Redirect::to("/catalog").into_response()
        }
        Err(err) => {
            let tpl = submission_detail_template(
                &submission_id,
                &submitted_at,
                &form,
                Some(err.to_string()),
                None,
            );
            Html(tpl.render().unwrap_or_else(|e| e.to_string())).into_response()
        }
    }
}

async fn submission_delete_post(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(submission_id): Path<String>,
) -> impl IntoResponse {
    match state.submissions.delete(&submission_id).await {
        Ok(()) => Redirect::to("/submissions").into_response(),
        Err(err) => (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    }
}
