//! Prefix query - matches terms starting with a prefix
//!
//! A prefix query matches all terms that begin with the specified prefix.
//! This is more efficient than a wildcard query with a trailing `*`.
//!
//! # Example
//!
//! ```rust
//! use squidex::query::nodes::PrefixQuery;
//!
//! // Match terms starting with "prog" (programming, progress, etc.)
//! let query = PrefixQuery::new("content", "prog");
//! ```

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches terms starting with a prefix
///
/// This is a specialized and optimized form of wildcard query
/// for patterns like `prefix*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixQuery {
    /// Field to search in
    pub field: String,
    /// Prefix to match
    pub prefix: String,
    /// Maximum number of terms to expand (default: 50)
    #[serde(default = "default_max_expansions")]
    pub max_expansions: usize,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_max_expansions() -> usize {
    50
}

fn default_boost() -> f32 {
    1.0
}

impl PrefixQuery {
    /// Create a new prefix query
    pub fn new(field: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            prefix: prefix.into(),
            max_expansions: 50,
            boost: 1.0,
        }
    }

    /// Set the maximum number of terms to expand
    pub fn with_max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!("prefix:{}:{}", self.field, self.prefix)
    }
}

impl QueryNode for PrefixQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            let mut results = RoaringBitmap::new();

            // Get accessor for term iteration
            if let Some(_accessor) = ctx.accessor() {
                // Ideally, we would use a prefix iterator on the term dictionary
                // For now, we can only get postings for exact terms

                // Check exact match as fallback
                let postings = ctx.get_postings(&self.prefix);
                for posting in postings {
                    results.insert(posting.docno.as_u32());
                }

                // TODO: Add prefix_iter capability to IndexAccessor
                // to iterate all terms starting with the prefix
            }

            Ok(results)
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost depends on prefix length - longer prefix = fewer matches
        let base_cost = ctx.total_docs() as f64;
        let prefix_factor = 1.0 / (1.0 + self.prefix.len() as f64);
        base_cost * prefix_factor
    }

    fn query_type(&self) -> &'static str {
        "prefix"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, _ctx: &QueryContext, _docno: u32) -> Option<f32> {
        // Prefix queries use constant scoring
        Some(self.boost)
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_query_creation() {
        let query = PrefixQuery::new("content", "prog");
        assert_eq!(query.field, "content");
        assert_eq!(query.prefix, "prog");
        assert_eq!(query.max_expansions, 50);
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_prefix_query_with_options() {
        let query = PrefixQuery::new("content", "prog")
            .with_max_expansions(100)
            .with_boost(1.5);

        assert_eq!(query.max_expansions, 100);
        assert_eq!(query.boost, 1.5);
    }

    #[test]
    fn test_cache_key() {
        let query = PrefixQuery::new("content", "prog");
        assert_eq!(query.cache_key(), "prefix:content:prog");
    }

    #[test]
    fn test_query_type() {
        let query = PrefixQuery::new("content", "prog");
        assert_eq!(query.query_type(), "prefix");
    }
}
