use crate::domain::error::AppError;

pub fn extract_video_id(youtube_url: &str) -> Result<String, AppError> {
    let patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ];
    for pattern in patterns {
        if let Some(caps) = regex_lite_match(pattern, youtube_url) {
            return Ok(caps);
        }
    }
    Err(AppError::Validation(format!(
        "Could not extract video ID from URL: {youtube_url}"
    )))
}

fn regex_lite_match(pattern: &str, text: &str) -> Option<String> {
    use regex::Regex;
    let re = Regex::new(pattern).ok()?;
    re.captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_watch_url() {
        let id = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extracts_youtu_be_url() {
        let id = extract_video_id("https://youtu.be/dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn rejects_invalid_url() {
        assert!(extract_video_id("https://example.com/not-youtube").is_err());
    }
}
