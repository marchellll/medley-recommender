use chrono::{DateTime, NaiveDateTime, Utc};
use uuid::{NoContext, Timestamp, Uuid};

use crate::domain::error::AppError;

pub fn new_song_id() -> String {
    Uuid::now_v7().to_string()
}

pub fn new_song_id_from_created_at(created_at: &str) -> String {
    let dt = parse_datetime(created_at).unwrap_or_else(|_| Utc::now());
    let ts = Timestamp::from_unix(
        NoContext,
        dt.timestamp() as u64,
        dt.timestamp_subsec_nanos(),
    );
    Uuid::new_v7(ts).to_string()
}

pub fn parse_song_id(id: &str) -> Result<Uuid, AppError> {
    Uuid::parse_str(id).map_err(|_| AppError::Validation(format!("invalid song id: {id}")))
}

pub fn is_legacy_song_id(id: &str) -> bool {
    id.starts_with("youtube/")
}

pub(crate) fn parse_datetime(s: &str) -> Result<DateTime<Utc>, AppError> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .or_else(|_| {
            NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
                .map(|n| n.and_utc())
        })
        .map_err(|e| AppError::Internal(format!("bad datetime: {e}")))
}
