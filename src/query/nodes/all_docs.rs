//! All documents query - matches every document in the index

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;

/// Query that matches all documents (excluding tombstones)
#[derive(Clone, Debug)]
pub struct AllDocsQuery {
    /// Boost factor for scoring
    pub boost: f32,
}

impl Default for AllDocsQuery {
    fn default() -> Self {
        Self { boost: 1.0 }
    }
}

impl AllDocsQuery {
    /// Create a new all docs query
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl QueryNode for AllDocsQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let total_docs = ctx.total_docs();
        let mut bitmap = RoaringBitmap::new();

        // Add all documents
        for docno in 0..total_docs as u32 {
            bitmap.insert(docno);
        }

        // Remove tombstoned documents
        bitmap -= ctx.tombstones();

        Ok(bitmap)
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost is proportional to total documents
        ctx.total_docs() as f64
    }

    fn query_type(&self) -> &'static str {
        "all_docs"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;
    use crate::TokenizerConfig;
    use std::sync::Arc;

    #[test]
    fn test_all_docs_query() {
        let tokenizer = Arc::new(Tokenizer::new(&TokenizerConfig::default()));
        let ctx = QueryContext::builder()
            .total_docs(100)
            .avg_doc_length(50.0)
            .tokenizer(tokenizer)
            .build();

        let query = AllDocsQuery::new();
        let result = query.execute(&ctx).unwrap();

        assert_eq!(result.len(), 100);
        assert_eq!(query.query_type(), "all_docs");
        assert!(!query.is_scoring());
    }

    #[test]
    fn test_all_docs_with_boost() {
        let query = AllDocsQuery::new().with_boost(2.0);
        assert_eq!(query.boost(), 2.0);
    }
}
