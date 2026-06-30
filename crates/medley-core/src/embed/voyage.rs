use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{EmbeddingProvider, InputType};
use crate::domain::error::AppError;

pub struct VoyageClient {
    client: Client,
    api_key: String,
    model: String,
    dimension: usize,
    base_url: String,
}

impl VoyageClient {
    pub fn new(api_key: String, model: String, dimension: usize, base_url: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            dimension,
            base_url,
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[async_trait]
impl EmbeddingProvider for VoyageClient {
    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(
        &self,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if self.api_key.is_empty() {
            return Err(AppError::Embedding("VOYAGE_API_KEY is not set".into()));
        }
        let input_type_str = match input_type {
            InputType::Query => "query",
            InputType::Document => "document",
        };
        let req = EmbedRequest {
            input: texts,
            model: &self.model,
            input_type: input_type_str,
            output_dimension: self.dimension,
        };
        let resp = self
            .client
            .post(format!(
                "{}/v1/embeddings",
                self.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| AppError::Embedding(e.to_string()))?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Embedding(format!("Voyage API error: {body}")));
        }
        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| AppError::Embedding(e.to_string()))?;
        Ok(parsed
            .data
            .into_iter()
            .map(|d| l2_normalize(d.embedding))
            .collect())
    }
}
