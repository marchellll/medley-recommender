use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use qdrant_edge::external::{ordered_float::OrderedFloat, uuid::Uuid};
use qdrant_edge::{
    Condition, Distance, EdgeConfigBuilder, EdgeShard, EdgeVectorParamsBuilder, FieldCondition,
    Filter, NamedQuery, Payload, PointId, PointInsertOperations, PointOperations,
    PointStructPersisted, QueryEnum, Range, SearchRequest, UpdateOperation, VectorStructPersisted,
};
use tokio::task;

use super::{VectorHit, VectorIndex};
use crate::domain::error::AppError;

pub struct EdgeVectorIndex {
    shard: Arc<Mutex<EdgeShard>>,
}

impl EdgeVectorIndex {
    pub fn open(path: &Path, dimension: usize) -> Result<Self, AppError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| AppError::Internal(e.to_string()))?;
        }

        let config = EdgeConfigBuilder::new()
            .vector(
                "",
                EdgeVectorParamsBuilder::new(dimension, Distance::Dot).build(),
            )
            .build();

        let shard = if path.join("edge_config.json").exists() {
            EdgeShard::load(path, None).map_err(map_edge_err)?
        } else {
            std::fs::create_dir_all(path).map_err(|e| AppError::Internal(e.to_string()))?;
            EdgeShard::new(path, config).map_err(map_edge_err)?
        };

        Ok(Self {
            shard: Arc::new(Mutex::new(shard)),
        })
    }

    fn song_point_id(song_id: &str) -> Result<PointId, AppError> {
        let id = uuid::Uuid::parse_str(song_id)
            .map_err(|_| AppError::Internal(format!("invalid song uuid: {song_id}")))?;
        Ok(PointId::Uuid(Uuid::from_bytes(*id.as_bytes())))
    }

    fn payload_key(name: &str) -> qdrant_edge::JsonPath {
        name.parse().expect("valid payload key")
    }

    fn song_payload(song_id: &str, key: &str, bpm: f64) -> Payload {
        Payload(serde_json::Map::from_iter([
            ("song_id".into(), serde_json::json!(song_id)),
            ("key".into(), serde_json::json!(key)),
            ("bpm".into(), serde_json::json!(bpm)),
        ]))
    }

    fn build_filter(
        keys: Option<&[String]>,
        bpm_min: Option<f64>,
        bpm_max: Option<f64>,
    ) -> Option<Filter> {
        let mut must = Vec::new();

        if let Some(keys) = keys {
            if !keys.is_empty() {
                must.push(Condition::Field(FieldCondition::new_match(
                    Self::payload_key("key"),
                    keys.to_vec().into(),
                )));
            }
        }

        if bpm_min.is_some() || bpm_max.is_some() {
            must.push(Condition::Field(FieldCondition::new_range(
                Self::payload_key("bpm"),
                Range {
                    gte: bpm_min.map(OrderedFloat),
                    lte: bpm_max.map(OrderedFloat),
                    lt: None,
                    gt: None,
                },
            )));
        }

        if must.is_empty() {
            None
        } else {
            Some(Filter {
                must: Some(must),
                ..Default::default()
            })
        }
    }

    fn payload_song_id(payload: &Payload) -> Option<String> {
        payload
            .0
            .get("song_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    }
}

fn map_edge_err(err: qdrant_edge::OperationError) -> AppError {
    AppError::IndexUnavailable(err.to_string())
}

#[async_trait]
impl VectorIndex for EdgeVectorIndex {
    async fn upsert(
        &self,
        song_id: &str,
        vector: Vec<f32>,
        key: &str,
        bpm: f64,
    ) -> Result<(), AppError> {
        let point = PointStructPersisted {
            id: Self::song_point_id(song_id)?,
            vector: VectorStructPersisted::from(vector),
            payload: Some(Self::song_payload(song_id, key, bpm)),
        };

        let shard = self.shard.clone();
        task::spawn_blocking(move || {
            let shard = shard
                .lock()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            shard
                .update(UpdateOperation::PointOperation(
                    PointOperations::UpsertPoints(PointInsertOperations::PointsList(vec![point])),
                ))
                .map_err(map_edge_err)
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn delete(&self, song_id: &str) -> Result<(), AppError> {
        let id = Self::song_point_id(song_id)?;
        let shard = self.shard.clone();
        task::spawn_blocking(move || {
            let shard = shard
                .lock()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            shard
                .update(UpdateOperation::PointOperation(
                    PointOperations::DeletePoints { ids: vec![id] },
                ))
                .map_err(map_edge_err)
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn search(
        &self,
        vector: &[f32],
        limit: usize,
        keys: Option<&[String]>,
        bpm_min: Option<f64>,
        bpm_max: Option<f64>,
    ) -> Result<Vec<VectorHit>, AppError> {
        let query_vector = vector.to_vec();
        let filter = Self::build_filter(keys, bpm_min, bpm_max);
        let shard = self.shard.clone();

        task::spawn_blocking(move || {
            let shard = shard
                .lock()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            let request = SearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(query_vector)),
                filter,
                params: None,
                limit,
                offset: 0,
                with_payload: Some(true.into()),
                with_vector: None,
                score_threshold: None,
            };
            let points = shard.search(request).map_err(map_edge_err)?;
            Ok(points
                .into_iter()
                .filter_map(|point| {
                    let song_id = point.payload.as_ref().and_then(Self::payload_song_id)?;
                    Some(VectorHit {
                        song_id,
                        score: point.score,
                    })
                })
                .collect::<Vec<_>>())
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }

    async fn flush(&self) -> Result<(), AppError> {
        let shard = self.shard.clone();
        task::spawn_blocking(move || {
            let shard = shard
                .lock()
                .map_err(|e| AppError::Internal(e.to_string()))?;
            shard.optimize().map_err(map_edge_err)?;
            Ok(())
        })
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
    }
}
