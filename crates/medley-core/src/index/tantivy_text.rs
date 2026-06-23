use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, FuzzyTermQuery, Occur, Query};
use tantivy::schema::{Field, Schema, Value, FAST, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term, TantivyDocument};
use tokio::task;

use super::TextSearchHit;
use super::TextSearchPage;
use crate::domain::error::AppError;
use crate::domain::models::{Song, SongListQuery};
use crate::domain::pagination::clamp_limit;

const WRITER_HEAP_BYTES: usize = 50_000_000;
const MAX_SEARCH_HITS: usize = 10_000;
/// Fuzzy edit distance only for longer tokens; short ones like "iman" would match "amin".
const MIN_FUZZY_TOKEN_LEN: usize = 5;

fn token_fuzzy_distance(token: &str) -> u8 {
    if token.len() >= MIN_FUZZY_TOKEN_LEN {
        1
    } else {
        0
    }
}

#[derive(Clone)]
struct SongFields {
    song_id: Field,
    title: Field,
    lyrics: Field,
    key: Field,
    bpm: Field,
}

pub struct TantivyTextIndex {
    fields: SongFields,
    writer: Arc<Mutex<IndexWriter>>,
    reader: Arc<RwLock<IndexReader>>,
}

impl TantivyTextIndex {
    pub fn open(path: &Path) -> Result<Self, AppError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AppError::Internal(e.to_string()))?;
        }

        let (index, fields) = if path.join("meta.json").exists() {
            let index = Index::open_in_dir(path).map_err(map_tantivy_err)?;
            let fields = song_fields(index.schema());
            (index, fields)
        } else {
            std::fs::create_dir_all(path).map_err(|e| AppError::Internal(e.to_string()))?;
            let (schema, fields) = build_schema();
            let index = Index::create_in_dir(path, schema).map_err(map_tantivy_err)?;
            (index, fields)
        };

        let writer = index
            .writer(WRITER_HEAP_BYTES)
            .map_err(map_tantivy_err)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(map_tantivy_err)?;

        Ok(Self {
            fields,
            writer: Arc::new(Mutex::new(writer)),
            reader: Arc::new(RwLock::new(reader)),
        })
    }

    fn build_search_query(fields: &SongFields, q: &str) -> Result<Box<dyn Query>, AppError> {
        let tokens: Vec<String> = q
            .split_whitespace()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect();

        if tokens.is_empty() {
            return Err(AppError::Validation("empty search query".into()));
        }

        let mut clauses = Vec::with_capacity(tokens.len());
        for token in tokens {
            let distance = token_fuzzy_distance(&token);
            let title_term = Term::from_field_text(fields.title, &token);
            let lyrics_term = Term::from_field_text(fields.lyrics, &token);
            let title_q =
                Box::new(FuzzyTermQuery::new_prefix(title_term, distance, true)) as Box<dyn Query>;
            let lyrics_q =
                Box::new(FuzzyTermQuery::new_prefix(lyrics_term, distance, true)) as Box<dyn Query>;
            clauses.push((
                Occur::Must,
                Box::new(BooleanQuery::union(vec![title_q, lyrics_q])) as Box<dyn Query>,
            ));
        }

        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    fn matches_filters(
        fields: &SongFields,
        doc: &TantivyDocument,
        query: &SongListQuery,
    ) -> Result<bool, AppError> {
        if let Some(key) = &query.key {
            let doc_key = doc
                .get_first(fields.key)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if doc_key != key {
                return Ok(false);
            }
        }
        if let Some(bpm_min) = query.bpm_min {
            let bpm = doc
                .get_first(fields.bpm)
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if bpm < bpm_min {
                return Ok(false);
            }
        }
        if let Some(bpm_max) = query.bpm_max {
            let bpm = doc
                .get_first(fields.bpm)
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if bpm > bpm_max {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn run_search(
    fields: &SongFields,
    reader: &Arc<RwLock<IndexReader>>,
    query: &SongListQuery,
) -> Result<TextSearchPage, AppError> {
        let q_text = query.q.as_ref().map(|s| s.trim()).unwrap_or("");
        if q_text.is_empty() {
            return Err(AppError::Validation("empty search query".into()));
        }

        let limit = clamp_limit(query.limit, 20, 100);

        if query.last_id.is_some() && query.last_rank.is_none() {
            return Err(AppError::InvalidCursor(
                "last_rank required when q is set".into(),
            ));
        }

    let search_query = TantivyTextIndex::build_search_query(fields, q_text)?;
    let reader = reader
        .read()
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let searcher = reader.searcher();
    let top_docs = searcher
        .search(
            &search_query,
            &TopDocs::with_limit(MAX_SEARCH_HITS).and_offset(0),
        )
        .map_err(map_tantivy_err)?;

    let mut hits: Vec<(String, f32)> = Vec::new();
    for (score, doc_address) in top_docs {
        let doc: TantivyDocument = searcher.doc(doc_address).map_err(map_tantivy_err)?;
        if !TantivyTextIndex::matches_filters(fields, &doc, query)? {
            continue;
        }
        let song_id = doc
            .get_first(fields.song_id)
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::Internal("missing song_id in text index".into()))?
            .to_string();
        hits.push((song_id, score));
    }

        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let total = hits.len() as i64;

        if let (Some(last_id), Some(last_rank)) = (&query.last_id, query.last_rank) {
            let last_rank = last_rank as f32;
            let cursor_ok = hits.iter().any(|(id, score)| {
                id == last_id && (*score - last_rank).abs() < f32::EPSILON
            });
            if !cursor_ok {
                return Err(AppError::InvalidCursor(
                    "last_id/last_rank not in current result set".into(),
                ));
            }
            hits.retain(|(id, score)| {
                *score < last_rank - f32::EPSILON
                    || ((*score - last_rank).abs() < f32::EPSILON && id.as_str() > last_id.as_str())
            });
        }

        let has_more = hits.len() > limit as usize;
        hits.truncate(limit as usize);

        let next_last_id = if has_more {
            hits.last().map(|(id, _)| id.clone())
        } else {
            None
        };
        let next_last_rank = if has_more {
            hits.last().map(|(_, score)| *score as f64)
        } else {
            None
        };

    Ok(TextSearchPage {
        hits: hits
            .into_iter()
            .map(|(song_id, score)| TextSearchHit { song_id, score })
            .collect(),
        limit,
        next_last_id,
        next_last_rank,
        has_more,
        total,
    })
}

fn build_schema() -> (Schema, SongFields) {
    let mut schema_builder = Schema::builder();
    let song_id = schema_builder.add_text_field("song_id", STRING | STORED | FAST);
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let lyrics = schema_builder.add_text_field("lyrics", TEXT | STORED);
    let key = schema_builder.add_text_field("key", STRING | STORED | FAST);
    let bpm = schema_builder.add_f64_field("bpm", STORED | FAST);
    let schema = schema_builder.build();
    let fields = SongFields {
        song_id,
        title,
        lyrics,
        key,
        bpm,
    };
    (schema, fields)
}

fn song_fields(schema: Schema) -> SongFields {
    SongFields {
        song_id: schema.get_field("song_id").expect("song_id field"),
        title: schema.get_field("title").expect("title field"),
        lyrics: schema.get_field("lyrics").expect("lyrics field"),
        key: schema.get_field("key").expect("key field"),
        bpm: schema.get_field("bpm").expect("bpm field"),
    }
}

fn map_tantivy_err(err: tantivy::TantivyError) -> AppError {
    AppError::Internal(err.to_string())
}

#[async_trait]
impl super::TextIndex for TantivyTextIndex {
    async fn upsert(&self, song: &Song) -> Result<(), AppError> {
        let fields = self.fields.clone();
        let writer = self.writer.clone();
        let reader = self.reader.clone();
        let song = song.clone();

        task::spawn_blocking(move || {
            let term = Term::from_field_text(fields.song_id, &song.song_id);
            let mut writer = writer.lock().map_err(|e| AppError::Internal(e.to_string()))?;
            writer.delete_term(term);
            let document = doc!(
                fields.song_id => song.song_id.clone(),
                fields.title => song.title.clone(),
                fields.lyrics => song.lyrics.clone(),
                fields.key => song.key.clone(),
                fields.bpm => song.bpm,
            );
            writer.add_document(document).map_err(map_tantivy_err)?;
            writer.commit().map_err(map_tantivy_err)?;
            let reader = reader.read().map_err(|e| AppError::Internal(e.to_string()))?;
            reader.reload().map_err(map_tantivy_err)
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn delete(&self, song_id: &str) -> Result<(), AppError> {
        let fields = self.fields.clone();
        let writer = self.writer.clone();
        let reader = self.reader.clone();
        let song_id = song_id.to_string();

        task::spawn_blocking(move || {
            let term = Term::from_field_text(fields.song_id, &song_id);
            let mut writer = writer.lock().map_err(|e| AppError::Internal(e.to_string()))?;
            writer.delete_term(term);
            writer.commit().map_err(map_tantivy_err)?;
            let reader = reader.read().map_err(|e| AppError::Internal(e.to_string()))?;
            reader.reload().map_err(map_tantivy_err)
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn search(&self, query: &SongListQuery) -> Result<TextSearchPage, AppError> {
        let fields = self.fields.clone();
        let reader = self.reader.clone();
        let query = query.clone();
        task::spawn_blocking(move || run_search(&fields, &reader, &query))
            .await
            .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn rebuild(&self, songs: &[Song]) -> Result<(), AppError> {
        let fields = self.fields.clone();
        let writer = self.writer.clone();
        let reader = self.reader.clone();
        let songs = songs.to_vec();

        task::spawn_blocking(move || {
            let mut writer = writer.lock().map_err(|e| AppError::Internal(e.to_string()))?;
            writer.delete_all_documents().map_err(map_tantivy_err)?;
            for song in songs {
                let document = doc!(
                    fields.song_id => song.song_id.clone(),
                    fields.title => song.title.clone(),
                    fields.lyrics => song.lyrics.clone(),
                    fields.key => song.key.clone(),
                    fields.bpm => song.bpm,
                );
                writer.add_document(document).map_err(map_tantivy_err)?;
            }
            writer.commit().map_err(map_tantivy_err)?;
            let reader = reader.read().map_err(|e| AppError::Internal(e.to_string()))?;
            reader.reload().map_err(map_tantivy_err)
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn doc_count(&self) -> Result<u64, AppError> {
        let reader = self.reader.clone();
        task::spawn_blocking(move || {
            let reader = reader.read().map_err(|e| AppError::Internal(e.to_string()))?;
            Ok(reader.searcher().num_docs())
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }
}
