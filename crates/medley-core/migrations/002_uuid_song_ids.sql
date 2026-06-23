-- Enforce one catalog row per YouTube URL (UUID v7 is assigned in application migration).
CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_youtube_url ON songs(youtube_url);
