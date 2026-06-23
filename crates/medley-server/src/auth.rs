use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    extract::FromRequestParts,
    http::{header, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

pub const ADMIN_COOKIE: &str = "medley_token";
const JWT_AUD: &str = "medley";
const JWT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

use crate::state::AppState;

const MCP_MUTATION_TOOLS: &[&str] = &["add_song", "update_song", "delete_song"];

#[derive(Clone)]
pub struct AdminAuth {
    token: String,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

#[derive(Clone)]
pub struct McpAuth {
    token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdminClaims {
    pub sub: String,
    pub role: String,
    pub aud: String,
    pub exp: u64,
}

pub struct AdminUser;

#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub token: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
}

impl AdminAuth {
    pub fn new(admin_token: String) -> Self {
        let encoding_key = EncodingKey::from_secret(admin_token.as_bytes());
        let decoding_key = DecodingKey::from_secret(admin_token.as_bytes());
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.set_audience(&[JWT_AUD]);
        Self {
            token: admin_token,
            encoding_key,
            decoding_key,
            validation,
        }
    }

    pub fn enabled(&self) -> bool {
        !self.token.is_empty()
    }

    pub fn verify_login(&self, token: &str) -> bool {
        if !self.enabled() {
            return false;
        }
        constant_time_eq(token.as_bytes(), self.token.as_bytes())
    }

    pub fn issue_jwt(&self) -> Result<String, jsonwebtoken::errors::Error> {
        let exp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_secs()
            + JWT_TTL.as_secs();
        let claims = AdminClaims {
            sub: "bootstrap".into(),
            role: "admin".into(),
            aud: JWT_AUD.into(),
            exp,
        };
        encode(&Header::default(), &claims, &self.encoding_key)
    }

    pub fn verify_jwt(&self, token: &str) -> Option<AdminClaims> {
        if !self.enabled() {
            return None;
        }
        decode::<AdminClaims>(token, &self.decoding_key, &self.validation)
            .ok()
            .map(|data| data.claims)
            .filter(|c| c.role == "admin")
    }

    pub fn token_from_parts(parts: &Parts) -> Option<String> {
        if let Some(cookie_header) = parts.headers.get(header::COOKIE) {
            let cookies = cookie_header.to_str().ok()?;
            for cookie in cookies.split(';') {
                let cookie = cookie.trim();
                if let Some(value) = cookie.strip_prefix(&format!("{ADMIN_COOKIE}=")) {
                    return Some(value.to_string());
                }
            }
        }
        bearer_from_headers(&parts.headers)
    }
}

impl McpAuth {
    pub fn new(api_token: String) -> Self {
        Self { token: api_token }
    }

    pub fn enabled(&self) -> bool {
        !self.token.is_empty()
    }

    pub fn verify_bearer(&self, token: &str) -> bool {
        if !self.enabled() {
            return false;
        }
        constant_time_eq(token.as_bytes(), self.token.as_bytes())
    }

    pub fn is_authenticated_headers(&self, headers: &axum::http::HeaderMap) -> bool {
        bearer_from_headers(headers)
            .is_some_and(|token| self.verify_bearer(&token))
    }
}

fn bearer_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    value.strip_prefix("Bearer ").map(str::trim).map(str::to_string)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && left.ct_eq(right).into()
}

pub fn admin_cookie_value(jwt: &str) -> String {
    format!("{ADMIN_COOKIE}={jwt}; HttpOnly; Path=/; SameSite=Lax; Max-Age={}", JWT_TTL.as_secs())
}

pub fn clear_admin_cookie() -> String {
    format!("{ADMIN_COOKIE}=; HttpOnly; Path=/; SameSite=Lax; Max-Age=0")
}

pub fn mcp_mutation_tool_name(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let method = value.get("method")?.as_str()?;
    if method != "tools/call" {
        return None;
    }
    let name = value
        .pointer("/params/name")
        .or_else(|| value.pointer("/params/tool"))
        .and_then(|v| v.as_str())?;
    Some(name.to_string())
}

pub fn is_mcp_mutation_tool(name: &str) -> bool {
    MCP_MUTATION_TOOLS.contains(&name)
}

pub struct AuthRejection {
    status: StatusCode,
    message: String,
}

impl AuthRejection {
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }
}

impl IntoResponse for AuthRejection {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

impl FromRequestParts<AppState> for AdminUser {
    type Rejection = AuthRejection;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let admin_auth = &state.admin_auth;

        if !admin_auth.enabled() {
            return Err(AuthRejection::service_unavailable(
                "ADMIN_TOKEN is not set",
            ));
        }

        let token = AdminAuth::token_from_parts(parts).ok_or_else(|| {
            AuthRejection::unauthorized("admin authentication required")
        })?;

        admin_auth
            .verify_jwt(&token)
            .ok_or_else(|| AuthRejection::unauthorized("invalid or expired admin token"))?;

        Ok(AdminUser)
    }
}

/// Read admin JWT from request without rejecting.
pub fn optional_admin_from_parts(parts: &Parts, admin_auth: &AdminAuth) -> bool {
    AdminAuth::token_from_parts(parts)
        .and_then(|token| admin_auth.verify_jwt(&token))
        .is_some()
}

pub fn optional_admin_from_request<B>(req: &axum::http::Request<B>, admin_auth: &AdminAuth) -> bool {
    if let Some(cookie_header) = req.headers().get(header::COOKIE).and_then(|v| v.to_str().ok()) {
        for cookie in cookie_header.split(';') {
            let cookie = cookie.trim();
            if let Some(value) = cookie.strip_prefix(&format!("{ADMIN_COOKIE}=")) {
                if admin_auth.verify_jwt(value).is_some() {
                    return true;
                }
            }
        }
    }
    bearer_from_headers(req.headers())
        .and_then(|token| admin_auth.verify_jwt(&token))
        .is_some()
}

tokio::task_local! {
    static MCP_AUTHENTICATED: bool;
}

pub async fn with_mcp_authenticated<F, R>(authenticated: bool, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    MCP_AUTHENTICATED.scope(authenticated, f).await
}

pub fn mcp_is_authenticated() -> bool {
    MCP_AUTHENTICATED.try_with(|v| *v).unwrap_or(false)
}

pub fn require_mcp_authenticated() -> Result<(), String> {
    if mcp_is_authenticated() {
        Ok(())
    } else {
        Err("MCP authentication required".into())
    }
}
