//! Query DSL and execution engine
//!
//! This module provides a structured query language for Squidex, supporting:
//! - Boolean queries (AND, OR, NOT)
//! - Term queries (exact match)
//! - Match queries (full-text search)
//! - Range queries (numeric/date ranges)
//! - Filter queries (cached, non-scoring)
//!
//! # Example
//!
//! ```json
//! {
//!   "query": {
//!     "bool": {
//!       "must": [
//!         { "match": { "content": "rust programming" } }
//!       ],
//!       "filter": [
//!         { "range": { "created_at": { "gte": "2024-01-01" } } }
//!       ]
//!     }
//!   }
//! }
//! ```

pub mod ast;
pub mod context;
pub mod executor;
pub mod nodes;
pub mod parser;
pub mod planner;
pub mod types;

pub use ast::{QueryNode, QueryNodeRef};
pub use context::QueryContext;
pub use executor::QueryExecutor;
pub use nodes::{
    AllDocsQuery, BoolQuery, MatchQuery, RangeQuery, TermQuery, TermsQuery,
};
pub use parser::QueryParser;
pub use planner::QueryPlanner;
pub use types::*;
