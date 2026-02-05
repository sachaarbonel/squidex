//! Term query - exact match on a field

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches documents containing an exact term in a field
///
/// This is the most basic query type - it looks up the term in the inverted
/// index and returns the posting list as a bitmap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermQuery {
    /// Field to search in
    pub field: String,
    /// Exact term to match
    pub term: String,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

impl TermQuery {
    /// Create a new term query
    pub fn new(field: impl Into<String>, term: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            term: term.into(),
            boost: 1.0,
        }
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the cache key for this term
    pub fn cache_key(&self) -> String {
        format!("term:{}:{}", self.field, self.term)
    }
}

impl QueryNode for TermQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // Get postings from the accessor
            let postings = ctx.get_postings(&self.term);
            let mut bitmap = RoaringBitmap::new();
            for posting in postings {
                bitmap.insert(posting.docno.as_u32());
            }
            Ok(bitmap)
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost is estimated by document frequency
        // If we don't have stats, assume moderate selectivity
        let doc_freq = ctx.doc_frequency(&self.cache_key());
        if doc_freq > 0 {
            doc_freq as f64
        } else {
            // Assume 10% of docs match if unknown
            ctx.total_docs() as f64 * 0.1
        }
    }

    fn query_type(&self) -> &'static str {
        "term"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, ctx: &QueryContext, docno: u32) -> Option<f32> {
        // Get term statistics for IDF calculation
        let stats = ctx.get_term_stats(&self.term);
        if stats.doc_frequency == 0 {
            return None;
        }

        // Get postings to find term frequency for this doc
        let postings = ctx.get_postings(&self.term);
        let posting = postings
            .iter()
            .find(|p| p.docno.as_u32() == docno)?;

        // Full BM25: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * dl/avgdl))
        let idf = ctx.bm25_idf(stats.doc_frequency);
        let tf = posting.term_frequency as f32;
        let dl = posting.doc_length as f32;
        let avgdl = ctx.avg_doc_length();
        let k1 = 1.2f32;
        let b = 0.75f32;

        let norm = 1.0 - b + b * (dl / avgdl.max(1.0));
        Some(idf * (tf * (k1 + 1.0)) / (tf + k1 * norm) * self.boost)
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
    fn test_term_query_creation() {
        let query = TermQuery::new("title", "rust");
        assert_eq!(query.field, "title");
        assert_eq!(query.term, "rust");
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_term_query_with_boost() {
        let query = TermQuery::new("title", "rust").with_boost(2.5);
        assert_eq!(query.boost, 2.5);
    }

    #[test]
    fn test_term_query_cache_key() {
        let query = TermQuery::new("title", "rust");
        assert_eq!(query.cache_key(), "term:title:rust");
    }

    #[test]
    fn test_term_query_execute() {
        let ctx = create_test_context();
        let query = TermQuery::new("title", "rust");
        let result = query.execute(&ctx).unwrap();

        // Currently returns empty bitmap (placeholder)
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_term_query_type() {
        let query = TermQuery::new("title", "rust");
        assert_eq!(query.query_type(), "term");
        assert!(query.is_scoring());
    }
}
