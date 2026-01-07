use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::metrics::SearchMetrics;
use crate::state_machine::SearchStateMachine;
use octopii::OctopiiNode;

use super::handlers::*;

/// Application state shared across all handlers
pub struct AppState {
    pub node: Arc<OctopiiNode>,
    pub state_machine: Arc<SearchStateMachine>,
    pub metrics: Arc<SearchMetrics>,
}

/// Create the HTTP router with all endpoints
pub fn create_router(state: AppState) -> Router {
    let state = Arc::new(state);

    Router::new()
        // Document operations
        .route("/v1/documents", post(index_document))
        .route("/v1/documents/:id", get(get_document))
        .route("/v1/documents/:id", delete(delete_document))
        // Search
        .route("/v1/search", post(search))
        // Batch operations
        .route("/v1/batch/index", post(batch_index))
        // Cluster management
        .route("/v1/cluster/status", get(cluster_status))
        // Health and metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
}
