use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    sync::Mutex,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::Request,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use axum::http::StatusCode;

use crate::auth::{is_mcp_mutation_tool, mcp_mutation_tool_name, optional_admin_from_request, with_mcp_authenticated};
use crate::state::AppState;

const WINDOW: Duration = Duration::from_secs(60);
const MAX_REQUESTS: usize = 10;

#[derive(Clone, Default)]
pub struct RateLimiter {
    inner: std::sync::Arc<Mutex<HashMap<IpAddr, Vec<Instant>>>>,
}

impl RateLimiter {
    pub fn check(&self, ip: IpAddr) -> Result<(), StatusCode> {
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("rate limiter lock");

        let count = guard
            .get_mut(&ip)
            .map(|entries| {
                entries.retain(|t| now.duration_since(*t) < WINDOW);
                entries.len()
            })
            .unwrap_or(0);

        if count == 0 {
            guard.remove(&ip);
        } else if count >= MAX_REQUESTS {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }

        guard.entry(ip).or_default().push(now);
        Ok(())
    }
}

pub fn client_ip<B>(req: &Request<B>) -> IpAddr {
    if let Some(forwarded) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(first) = forwarded.split(',').next() {
            if let Ok(ip) = first.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.ip())
        .unwrap_or(IpAddr::from([127, 0, 0, 1]))
}

pub async fn http_rate_limit_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    if optional_admin_from_request(&req, &state.admin_auth) {
        return next.run(req).await;
    }

    let ip = client_ip(&req);
    if state.rate_limit.check(ip).is_err() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": "rate limit exceeded (10/min per IP)" })),
        )
            .into_response();
    }

    next.run(req).await
}

pub async fn mcp_auth_middleware(
    State(state): State<AppState>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let authenticated = state.mcp_auth.is_authenticated_headers(req.headers());
    let ip = client_ip(&req);

    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid request body" })),
            )
                .into_response();
        }
    };

    if state.rate_limit.check(ip).is_err() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": "rate limit exceeded (10/min per IP)" })),
        )
            .into_response();
    }

    if !authenticated {
        if let Some(tool) = mcp_mutation_tool_name(&body_bytes) {
            if is_mcp_mutation_tool(&tool) {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({ "error": "MCP authentication required" })),
                )
                    .into_response();
            }
        }
    }

    let req = Request::from_parts(parts, Body::from(body_bytes));
    with_mcp_authenticated(authenticated, next.run(req)).await
}
