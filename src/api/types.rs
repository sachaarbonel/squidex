use serde::{Deserialize, Serialize};

use crate::models::*;

/// Request to index a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    pub content: String,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<DocumentMetadata>,
    #[serde(default)]
    pub refresh: Option<String>, // "none" (default) | "wait_for"
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// Response after indexing a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    pub id: DocumentId,
    pub commit_index: u64,
}

/// Search request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequestApi {
    pub query: Option<String>,
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub mode: SearchModeApi,
    #[serde(default = "default_keyword_weight")]
    pub keyword_weight: f32,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub filters: Vec<Filter>,
    #[serde(default)]
    pub min_index_applied_index: Option<u64>,
    #[serde(default)]
    pub wait_for: Option<u64>, // optional wait_for milliseconds until index_applied_index >= min
}

fn default_keyword_weight() -> f32 {
    0.5
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchModeApi {
    Keyword,
    Vector,
    Hybrid,
}

impl Default for SearchModeApi {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: u64,
    pub total_hits: u64,
}

/// Cluster status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub leader_id: Option<u64>,
    pub node_id: u64,
    pub is_leader: bool,
    pub voters: Vec<u64>,
    pub learners: Vec<u64>,
    pub total_documents: u64,
    pub index_version: u64,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// API Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

impl ErrorResponse {
    pub fn new(error: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            message: message.into(),
        }
    }
}
