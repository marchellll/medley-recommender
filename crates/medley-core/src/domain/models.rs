use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const VALID_KEYS: &[&str] = &[
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Song {
    pub song_id: String,
    pub title: String,
    pub youtube_url: String,
    pub lyrics: String,
    pub bpm: f64,
    pub key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongSummary {
    pub song_id: String,
    pub title: String,
    pub youtube_url: String,
    pub bpm: f64,
    pub key: String,
}

impl From<Song> for SongSummary {
    fn from(s: Song) -> Self {
        Self {
            song_id: s.song_id,
            title: s.title,
            youtube_url: s.youtube_url,
            bpm: s.bpm,
            key: s.key,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NewSong {
    pub title: String,
    pub youtube_url: String,
    pub lyrics: String,
    pub bpm: f64,
    pub key: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SongPatch {
    pub title: Option<String>,
    pub youtube_url: Option<String>,
    pub lyrics: Option<String>,
    pub bpm: Option<f64>,
    pub key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SongSubmission {
    pub submission_id: String,
    pub title: String,
    pub youtube_url: String,
    pub lyrics: String,
    pub bpm: f64,
    pub key: String,
    pub submitted_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SubmissionListQuery {
    pub limit: Option<u32>,
    pub last_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SongListQuery {
    pub q: Option<String>,
    pub key: Option<String>,
    pub bpm_min: Option<f64>,
    pub bpm_max: Option<f64>,
    pub limit: Option<u32>,
    pub last_id: Option<String>,
    pub last_rank: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub bpm_min: Option<f64>,
    pub bpm_max: Option<f64>,
    pub keys: Option<Vec<String>>,
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub song_id: String,
    pub title: String,
    pub bpm: f64,
    pub key: String,
    pub similarity_score: f64,
    pub youtube_url: String,
}
