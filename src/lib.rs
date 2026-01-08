pub mod api;
pub mod config;
pub mod consensus;
pub mod error;
pub mod metrics;
pub mod models;
pub mod segment;
pub mod state_machine;
pub mod tokenizer;
pub mod vector;
pub mod persistence;

pub use api::{create_router, AppState};
pub use config::{
    IndexSettings, NodeConfig, PerformanceProfile, ProductQuantizationConfig, TokenizerConfig,
};
pub use consensus::{LogEntry, RaftServiceImpl, SquidexNode};
pub use error::{Result, SquidexError};
pub use metrics::SearchMetrics;
pub use models::*;
pub use state_machine::SearchStateMachine;
pub use tokenizer::Tokenizer;
pub use vector::{Codebook, QuantizedVector, QuantizedVectorStore, VectorStoreSnapshot};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
