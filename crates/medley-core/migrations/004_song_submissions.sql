CREATE TABLE IF NOT EXISTS song_submissions (
    submission_id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    youtube_url TEXT NOT NULL UNIQUE,
    lyrics TEXT NOT NULL,
    bpm REAL NOT NULL,
    key TEXT NOT NULL,
    submitted_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_submissions_submitted_at ON song_submissions(submitted_at);
