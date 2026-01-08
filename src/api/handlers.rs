use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;

use crate::api::types::{self, *};
use crate::consensus::LogEntry;
use crate::error::SquidexError;
use crate::models::*;

use super::router::AppState;

/// Error wrapper for API handlers
pub enum ApiError {
    Squidex(SquidexError),
    NotLeader,
    BadRequest(String),
}

impl From<SquidexError> for ApiError {
    fn from(e: SquidexError) -> Self {
        ApiError::Squidex(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            ApiError::Squidex(e) => {
                let error_type = match &e {
                    SquidexError::DocumentNotFound(_) => "document_not_found",
                    SquidexError::InvalidEmbeddingDimensions { .. } => "invalid_embedding",
                    SquidexError::Serialization(_) => "serialization_error",
                    SquidexError::IndexError(_) => "index_error",
                    SquidexError::IncompatibleSnapshot { .. } => "incompatible_snapshot",
                    SquidexError::Io(_) => "io_error",
                    SquidexError::Consensus(_) => "consensus_error",
                    SquidexError::NotLeader => "not_leader",
                    SquidexError::InvalidRequest(_) => "invalid_request",
                    SquidexError::SearchError(_) => "search_error",
                    SquidexError::Internal(_) => "internal_error",
                    SquidexError::VectorNotFound(_) => "vector_not_found",
                    SquidexError::VectorStoreNotTrained => "vector_store_not_trained",
                };
                (StatusCode::BAD_REQUEST, error_type, e.to_string())
            }
            ApiError::NotLeader => (
                StatusCode::SERVICE_UNAVAILABLE,
                "not_leader",
                "This node is not the leader. Please forward request to the leader.".to_string(),
            ),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "bad_request", msg),
        };

        let error_response = ErrorResponse::new(error_type, message);
        (status, Json(error_response)).into_response()
    }
}

/// Index a single document
pub async fn index_document(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Only leader can accept writes
    if !state.node.is_leader().await {
        return Err(ApiError::NotLeader);
    }

    let doc_id = state.state_machine.next_document_id();

    let doc = Document {
        id: doc_id,
        content: req.content,
        embedding: req.embedding,
        metadata: req.metadata.unwrap_or_default(),
        created_at: current_timestamp(),
        updated_at: current_timestamp(),
    };

    let entry = LogEntry::IndexDocument(doc);

    state
        .node
        .propose(entry)
        .await
        .map_err(|e| SquidexError::Consensus(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(IndexResponse { id: doc_id })))
}

/// Get a document by ID
pub async fn get_document(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<impl IntoResponse, ApiError> {
    match state.state_machine.get_document(id) {
        Some(doc) => Ok(Json(doc)),
        None => Err(ApiError::Squidex(SquidexError::DocumentNotFound(id))),
    }
}

/// Delete a document
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<impl IntoResponse, ApiError> {
    // Only leader can accept writes
    if !state.node.is_leader().await {
        return Err(ApiError::NotLeader);
    }

    let entry = LogEntry::DeleteDocument(id);

    state
        .node
        .propose(entry)
        .await
        .map_err(|e| SquidexError::Consensus(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Search documents
pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequestApi>,
) -> Result<impl IntoResponse, ApiError> {
    let start = std::time::Instant::now();

    let results = match req.mode {
        SearchModeApi::Keyword => {
            let query = req.query.ok_or_else(|| {
                ApiError::BadRequest("query required for keyword search".to_string())
            })?;
            state.state_machine.keyword_search(&query, req.top_k)
        }
        SearchModeApi::Vector => {
            let embedding = req.embedding.ok_or_else(|| {
                ApiError::BadRequest("embedding required for vector search".to_string())
            })?;
            state.state_machine.vector_search(&embedding, req.top_k)
        }
        SearchModeApi::Hybrid => {
            let query = req.query.ok_or_else(|| {
                ApiError::BadRequest("query required for hybrid search".to_string())
            })?;
            let embedding = req.embedding.ok_or_else(|| {
                ApiError::BadRequest("embedding required for hybrid search".to_string())
            })?;
            state
                .state_machine
                .hybrid_search(&query, &embedding, req.top_k, req.keyword_weight)
        }
    };

    let response = types::SearchResponse {
        results,
        took_ms: start.elapsed().as_millis() as u64,
        total_hits: state.state_machine.total_documents(),
    };

    Ok(Json(response))
}

/// Batch index documents
pub async fn batch_index(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<IndexRequest>>,
) -> Result<impl IntoResponse, ApiError> {
    // Only leader can accept writes
    if !state.node.is_leader().await {
        return Err(ApiError::NotLeader);
    }

    let documents: Vec<Document> = requests
        .into_iter()
        .map(|req| {
            let doc_id = state.state_machine.next_document_id();
            Document {
                id: doc_id,
                content: req.content,
                embedding: req.embedding,
                metadata: req.metadata.unwrap_or_default(),
                created_at: current_timestamp(),
                updated_at: current_timestamp(),
            }
        })
        .collect();

    let entry = LogEntry::BatchIndex(documents);

    state
        .node
        .propose(entry)
        .await
        .map_err(|e| SquidexError::Consensus(e.to_string()))?;

    Ok(StatusCode::CREATED)
}

/// Get cluster status
pub async fn cluster_status(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    let membership = state.node.membership().await;
    let is_leader = state.node.is_leader().await;
    let leader_id = state.node.leader_id().await;

    let status = ClusterStatus {
        leader_id,
        node_id: state.node.id(),
        is_leader,
        voters: membership.voters,
        learners: membership.learners,
        total_documents: state.state_machine.total_documents(),
        index_version: state.state_machine.index_version(),
    };

    Ok(Json(status))
}

/// Health check endpoint
pub async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: crate::VERSION.to_string(),
    })
}

/// Prometheus metrics endpoint
pub async fn metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = state.metrics.registry().gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();

    (
        axum::http::StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        buffer,
    )
}
