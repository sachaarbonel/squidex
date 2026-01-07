pub mod api;
pub mod config;
pub mod error;
pub mod metrics;
pub mod models;
pub mod state_machine;
pub mod tokenizer;

pub use api::{create_router, AppState};
pub use config::{IndexSettings, NodeConfig, PerformanceProfile, TokenizerConfig};
pub use error::{Result, SquidexError};
pub use metrics::SearchMetrics;
pub use models::*;
pub use state_machine::SearchStateMachine;
pub use tokenizer::Tokenizer;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
