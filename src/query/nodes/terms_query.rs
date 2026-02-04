//! Terms query - matches documents containing any of the specified terms

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches documents containing any of the specified terms in a field
///
/// This is equivalent to a boolean OR of multiple term queries, but more efficient
/// as it can be processed in a single pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermsQuery {
    /// Field to search in
    pub field: String,
    /// Terms to match (document must contain at least one)
    pub terms: Vec<String>,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

impl TermsQuery {
    /// Create a new terms query
    pub fn new(field: impl Into<String>, terms: Vec<String>) -> Self {
        Self {
            field: field.into(),
            terms,
            boost: 1.0,
        }
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Add a term to the query
    pub fn add_term(mut self, term: impl Into<String>) -> Self {
        self.terms.push(term.into());
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        let mut sorted_terms = self.terms.clone();
        sorted_terms.sort();
        format!("terms:{}:{}", self.field, sorted_terms.join(","))
    }
}

impl QueryNode for TermsQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // Union of all term posting lists
            // Real implementation would:
            // 1. Look up each term in the term dictionary
            // 2. Union all posting lists
            // 3. Filter out tombstones
            Ok(RoaringBitmap::new())
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Sum of all term document frequencies
        // More terms = higher cost
        let total_freq: u64 = self
            .terms
            .iter()
            .map(|term| {
                let key = format!("term:{}:{}", self.field, term);
                ctx.doc_frequency(&key) as u64
            })
            .sum();

        if total_freq > 0 {
            total_freq as f64
        } else {
            // Assume 20% of docs match if unknown (more than single term)
            ctx.total_docs() as f64 * 0.2 * self.terms.len() as f64
        }
    }

    fn query_type(&self) -> &'static str {
        "terms"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use crate::TokenizerConfig;
    use std::sync::Arc;

    fn create_test_context() -> QueryContext {
        let tokenizer = Arc::new(Tokenizer::new(&TokenizerConfig::default()));
        QueryContext::builder()
            .total_docs(1000)
            .avg_doc_length(100.0)
            .tokenizer(tokenizer)
            .build()
    }

    #[test]
    fn test_terms_query_creation() {
        let query = TermsQuery::new("status", vec!["active".to_string(), "pending".to_string()]);
        assert_eq!(query.field, "status");
        assert_eq!(query.terms.len(), 2);
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_terms_query_builder() {
        let query = TermsQuery::new("tags", vec![])
            .add_term("rust")
            .add_term("programming")
            .with_boost(1.5);

        assert_eq!(query.terms.len(), 2);
        assert_eq!(query.boost, 1.5);
    }

    #[test]
    fn test_terms_query_cache_key() {
        let query = TermsQuery::new("tags", vec!["b".to_string(), "a".to_string()]);
        // Terms should be sorted in cache key
        assert_eq!(query.cache_key(), "terms:tags:a,b");
    }

    #[test]
    fn test_terms_query_execute() {
        let ctx = create_test_context();
        let query = TermsQuery::new("status", vec!["active".to_string()]);
        let result = query.execute(&ctx).unwrap();

        // Currently returns empty bitmap (placeholder)
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_terms_query_type() {
        let query = TermsQuery::new("status", vec![]);
        assert_eq!(query.query_type(), "terms");
        assert!(query.is_scoring());
    }
}
