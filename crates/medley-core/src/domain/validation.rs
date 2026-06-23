use crate::domain::error::AppError;
use crate::domain::models::{NewSong, Song, SongPatch, VALID_KEYS};

pub fn validate_key(key: &str) -> Result<(), AppError> {
    if VALID_KEYS.contains(&key) {
        Ok(())
    } else {
        Err(AppError::Validation(format!(
            "key must be one of: {}",
            VALID_KEYS.join(", ")
        )))
    }
}

pub fn validate_bpm(bpm: f64) -> Result<(), AppError> {
    if bpm > 0.0 && bpm <= 300.0 {
        Ok(())
    } else {
        Err(AppError::Validation("bpm must be > 0 and <= 300".into()))
    }
}

pub fn validate_new_song(song: &NewSong) -> Result<(), AppError> {
    if song.title.trim().is_empty() {
        return Err(AppError::Validation("title is required".into()));
    }
    if song.lyrics.trim().is_empty() {
        return Err(AppError::Validation("lyrics is required".into()));
    }
    validate_bpm(song.bpm)?;
    validate_key(&song.key)?;
    crate::domain::youtube::extract_video_id(&song.youtube_url)?;
    Ok(())
}

pub fn validate_patch(patch: &SongPatch) -> Result<(), AppError> {
    if let Some(title) = &patch.title {
        if title.trim().is_empty() {
            return Err(AppError::Validation("title cannot be empty".into()));
        }
    }
    if let Some(lyrics) = &patch.lyrics {
        if lyrics.trim().is_empty() {
            return Err(AppError::Validation("lyrics cannot be empty".into()));
        }
    }
    if let Some(bpm) = patch.bpm {
        validate_bpm(bpm)?;
    }
    if let Some(key) = &patch.key {
        validate_key(key)?;
    }
    if let Some(url) = &patch.youtube_url {
        crate::domain::youtube::extract_video_id(url)?;
    }
    Ok(())
}

pub fn song_readiness_issues(song: &Song) -> Vec<&'static str> {
    let mut issues = Vec::new();
    if song.lyrics.trim().is_empty() {
        issues.push("missing lyrics");
    }
    if song.bpm <= 0.0 {
        issues.push("missing or invalid bpm");
    }
    if song.key.trim().is_empty() {
        issues.push("missing key");
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::{NewSong, Song, SongPatch};
    use chrono::Utc;

    fn sample_new_song() -> NewSong {
        NewSong {
            title: "Test".into(),
            youtube_url: "https://www.youtube.com/watch?v=abc12345678".into(),
            lyrics: "hello".into(),
            bpm: 120.0,
            key: "G".into(),
        }
    }

    fn sample_song() -> Song {
        let now = Utc::now();
        Song {
            song_id: "019ade09-0000-7000-8000-000000000001".into(),
            title: "Test Song".into(),
            youtube_url: "https://www.youtube.com/watch?v=test1234567".into(),
            lyrics: "Line one\nLine two".into(),
            bpm: 120.0,
            key: "G".into(),
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn rejects_invalid_key() {
        let mut song = sample_new_song();
        song.key = "H".into();
        assert!(validate_new_song(&song).is_err());
    }

    #[test]
    fn accepts_valid_new_song() {
        assert!(validate_new_song(&sample_new_song()).is_ok());
    }

    #[test]
    fn rejects_empty_title() {
        let mut song = sample_new_song();
        song.title = "   ".into();
        assert!(validate_new_song(&song).is_err());
    }

    #[test]
    fn rejects_invalid_bpm() {
        let mut song = sample_new_song();
        song.bpm = 0.0;
        assert!(validate_new_song(&song).is_err());
    }

    #[test]
    fn patch_rejects_empty_lyrics() {
        let patch = SongPatch {
            lyrics: Some("  ".into()),
            ..Default::default()
        };
        assert!(validate_patch(&patch).is_err());
    }

    #[test]
    fn ready_song_has_no_issues() {
        assert!(song_readiness_issues(&sample_song()).is_empty());
    }

    #[test]
    fn readiness_flags_missing_lyrics_bpm_and_key() {
        let mut song = sample_song();
        song.lyrics = "  ".into();
        song.bpm = 0.0;
        song.key = String::new();
        let issues = song_readiness_issues(&song);
        assert!(issues.contains(&"missing lyrics"));
        assert!(issues.iter().any(|i| i.contains("bpm")));
        assert!(issues.contains(&"missing key"));
    }
}
