use std::sync::Arc;

use medley_core::service::search_service::SearchService;
use medley_core::service::song_service::SongService;
use medley_core::service::submission_service::SubmissionService;

use crate::auth::{AdminAuth, McpAuth};
use crate::rate_limit::RateLimiter;

#[derive(Clone)]
pub struct AppState {
    pub songs: Arc<SongService>,
    pub submissions: Arc<SubmissionService>,
    pub search: Arc<SearchService>,
    pub admin_auth: Arc<AdminAuth>,
    pub mcp_auth: Arc<McpAuth>,
    pub rate_limit: Arc<RateLimiter>,
    pub submission_rate_limit: Arc<RateLimiter>,
}

impl AppState {
    pub fn new(
        songs: Arc<SongService>,
        submissions: Arc<SubmissionService>,
        search: Arc<SearchService>,
        admin_auth: Arc<AdminAuth>,
        mcp_auth: Arc<McpAuth>,
        rate_limit: Arc<RateLimiter>,
        submission_rate_limit: Arc<RateLimiter>,
    ) -> Self {
        Self {
            songs,
            submissions,
            search,
            admin_auth,
            mcp_auth,
            rate_limit,
            submission_rate_limit,
        }
    }

    pub fn is_admin_from_cookie_header(&self, cookie_header: Option<&str>) -> bool {
        let Some(cookie_header) = cookie_header else {
            return false;
        };
        for cookie in cookie_header.split(';') {
            let cookie = cookie.trim();
            if let Some(value) = cookie.strip_prefix("medley_token=") {
                return self.admin_auth.verify_jwt(value).is_some();
            }
        }
        false
    }
}
