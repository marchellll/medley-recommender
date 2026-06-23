-- Additive upgrade for existing Python-era medley.db.
-- CREATE TABLE IF NOT EXISTS keeps legacy rows/columns (e.g. embedding_file_path).
-- FTS rebuild/backfill happens in application code after migrate.

CREATE TABLE IF NOT EXISTS songs (
    song_id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    youtube_url TEXT NOT NULL,
    lyrics TEXT NOT NULL,
    bpm REAL NOT NULL,
    key TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS songs_fts USING fts5(
    title,
    lyrics,
    content='songs',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS songs_ai AFTER INSERT ON songs BEGIN
    INSERT INTO songs_fts(rowid, title, lyrics) VALUES (new.rowid, new.title, new.lyrics);
END;

CREATE TRIGGER IF NOT EXISTS songs_ad AFTER DELETE ON songs BEGIN
    INSERT INTO songs_fts(songs_fts, rowid, title, lyrics) VALUES ('delete', old.rowid, old.title, old.lyrics);
END;

CREATE TRIGGER IF NOT EXISTS songs_au AFTER UPDATE ON songs BEGIN
    INSERT INTO songs_fts(songs_fts, rowid, title, lyrics) VALUES ('delete', old.rowid, old.title, old.lyrics);
    INSERT INTO songs_fts(rowid, title, lyrics) VALUES (new.rowid, new.title, new.lyrics);
END;
