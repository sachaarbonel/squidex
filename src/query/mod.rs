//! Query DSL and execution engine
//!
//! This module provides a structured query language for Squidex, supporting:
//! - Boolean queries (AND, OR, NOT)
//! - Term queries (exact match)
//! - Match queries (full-text search)
//! - Range queries (numeric/date ranges)
//! - Filter queries (cached, non-scoring)
//! - Phrase queries (exact phrase with proximity)
//! - Wildcard queries (* and ?)
//! - Fuzzy queries (edit distance matching)
//! - Prefix queries (term prefix matching)
//!
//! # JSON DSL Example
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
//!
//! # Query String Example
//!
//! ```rust
//! use squidex::query::query_string::QueryStringParser;
//!
//! let mut parser = QueryStringParser::new("title:rust AND (tags:tutorial OR tags:guide)").unwrap();
//! let query = parser.parse().unwrap();
//! ```

pub mod accessor;
pub mod ast;
pub mod context;
pub mod executor;
pub mod nodes;
pub mod parser;
pub mod planner;
pub mod query_string;
pub mod types;

pub use accessor::{IndexAccessor, PostingEntry, SegmentAccessor, TermStats};
pub use ast::{QueryNode, QueryNodeRef};
pub use context::QueryContext;
pub use executor::QueryExecutor;
pub use nodes::{
    AllDocsQuery, BoolQuery, FuzzyQuery, MatchQuery, PhraseQuery, PrefixQuery, RangeQuery,
    TermQuery, TermsQuery, WildcardQuery,
};
pub use parser::QueryParser;
pub use planner::QueryPlanner;
pub use query_string::QueryStringParser;
pub use types::*;
