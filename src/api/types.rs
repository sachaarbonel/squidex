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
///
/// Uses Elasticsearch-compatible Query DSL. All queries go through the `query` field.
///
/// # Examples
///
/// Simple match query:
/// ```json
/// { "query": { "match": { "content": "rust programming" } } }
/// ```
///
/// Lucene-style query string:
/// ```json
/// { "query": { "query_string": { "query": "title:rust AND tags:tutorial" } } }
/// ```
///
/// Complex bool query:
/// ```json
/// {
///   "query": {
///     "bool": {
///       "must": [{ "match": { "content": "rust" } }],
///       "filter": [{ "range": { "year": { "gte": 2024 } } }]
///     }
///   }
/// }
/// ```
///
/// Hybrid search (text + vector):
/// ```json
/// {
///   "query": { "match": { "content": "rust" } },
///   "embedding": [0.1, 0.2, ...],
///   "keyword_weight": 0.7
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequestApi {
    /// Query DSL (required)
    pub query: serde_json::Value,
    /// Optional embedding for hybrid search
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    /// Weight for keyword vs vector scoring (0.0 = vector only, 1.0 = keyword only)
    #[serde(default = "default_keyword_weight")]
    pub keyword_weight: f32,
    /// Number of results to return
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum index version to wait for (consistency)
    #[serde(default)]
    pub min_index_applied_index: Option<u64>,
    /// Timeout in ms to wait for index version
    #[serde(default)]
    pub wait_for: Option<u64>,
}

fn default_keyword_weight() -> f32 {
    0.5
}

fn default_top_k() -> usize {
    10
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
