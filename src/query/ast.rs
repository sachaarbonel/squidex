//! Abstract Syntax Tree for query representation
//!
//! This module defines the core `QueryNode` trait that all query types implement,
//! providing a unified interface for query execution, optimization, and cost estimation.

use crate::Result;
use roaring::RoaringBitmap;
use std::fmt::Debug;
use std::sync::Arc;

use super::context::QueryContext;

/// Reference-counted query node for efficient tree sharing
pub type QueryNodeRef = Arc<dyn QueryNode>;

/// Core trait for all query nodes in the AST
///
/// Query nodes form a tree structure that represents the logical structure
/// of a search query. Each node can be executed against a `QueryContext`
/// to produce a set of matching document IDs.
pub trait QueryNode: Send + Sync + Debug {
    /// Execute the query and return matching document IDs as a bitmap
    ///
    /// The bitmap contains document numbers (dense IDs within a segment).
    /// The caller is responsible for mapping these back to external document IDs.
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap>;

    /// Estimate the execution cost of this query
    ///
    /// This is used by the query planner to reorder clauses for optimal execution.
    /// Lower costs should be executed first (e.g., highly selective filters).
    ///
    /// The cost is estimated based on:
    /// - Document frequency of terms
    /// - Selectivity of filters
    /// - Complexity of the query structure
    fn estimate_cost(&self, ctx: &QueryContext) -> f64;

    /// Get the query type name for debugging and logging
    fn query_type(&self) -> &'static str;

    /// Whether this query produces scores (vs just filtering)
    ///
    /// Scoring queries contribute to the relevance score of matching documents.
    /// Non-scoring queries (filters) only determine which documents match.
    fn is_scoring(&self) -> bool {
        true
    }

    /// Get the boost factor for this query
    fn boost(&self) -> f32 {
        1.0
    }

    /// Calculate the score contribution for a matching document
    ///
    /// This is called for each document that matches the query.
    /// Returns None if the document doesn't match or if this is a non-scoring query.
    fn score(&self, _ctx: &QueryContext, _docno: u32) -> Option<f32> {
        None
    }

    /// Clone this query node into a boxed trait object
    fn clone_box(&self) -> Box<dyn QueryNode>;
}

impl Clone for Box<dyn QueryNode> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// A query that matches all documents
#[derive(Clone, Debug)]
pub struct MatchAllQuery {
    pub boost: f32,
}

impl Default for MatchAllQuery {
    fn default() -> Self {
        Self { boost: 1.0 }
    }
}

impl QueryNode for MatchAllQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        // Return all document numbers
        let max_doc = ctx.total_docs();
        let mut bitmap = RoaringBitmap::new();
        for i in 0..max_doc {
            bitmap.insert(i as u32);
        }
        Ok(bitmap)
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost is proportional to total documents
        ctx.total_docs() as f64
    }

    fn query_type(&self) -> &'static str {
        "match_all"
    }

    fn is_scoring(&self) -> bool {
        false
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, _ctx: &QueryContext, _docno: u32) -> Option<f32> {
        Some(self.boost)
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(self.clone())
    }
}

/// A query that matches no documents
#[derive(Clone, Debug, Default)]
pub struct MatchNoneQuery;

impl QueryNode for MatchNoneQuery {
    fn execute(&self, _ctx: &QueryContext) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }

    fn estimate_cost(&self, _ctx: &QueryContext) -> f64 {
        0.0 // Free - matches nothing
    }

    fn query_type(&self) -> &'static str {
        "match_none"
    }

    fn is_scoring(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_all_query() {
        let query = MatchAllQuery::default();
        assert_eq!(query.query_type(), "match_all");
        assert!(!query.is_scoring());
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_match_none_query() {
        let query = MatchNoneQuery;
        assert_eq!(query.query_type(), "match_none");
        assert!(!query.is_scoring());
    }
}
