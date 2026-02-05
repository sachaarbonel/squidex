//! Match query - full-text search with analysis

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::query::types::MatchOperator;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that performs full-text search on a field
///
/// The input text is analyzed (tokenized, lowercased, stemmed) and the
/// resulting terms are searched using the specified operator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchQuery {
    /// Field to search in
    pub field: String,
    /// Text to search for (will be analyzed)
    pub text: String,
    /// How to combine terms (AND/OR)
    #[serde(default)]
    pub operator: MatchOperator,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
    /// Minimum number of terms that should match (for OR operator)
    #[serde(default)]
    pub minimum_should_match: Option<String>,
    /// Analyzer to use (if not specified, uses field's default analyzer)
    #[serde(default)]
    pub analyzer: Option<String>,
}

fn default_boost() -> f32 {
    1.0
}

impl MatchQuery {
    /// Create a new match query
    pub fn new(field: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            text: text.into(),
            operator: MatchOperator::default(),
            boost: 1.0,
            minimum_should_match: None,
            analyzer: None,
        }
    }

    /// Set the operator to AND (all terms must match)
    pub fn with_and_operator(mut self) -> Self {
        self.operator = MatchOperator::And;
        self
    }

    /// Set the operator to OR (at least one term must match)
    pub fn with_or_operator(mut self) -> Self {
        self.operator = MatchOperator::Or;
        self
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set minimum should match
    pub fn with_minimum_should_match(mut self, msm: impl Into<String>) -> Self {
        self.minimum_should_match = Some(msm.into());
        self
    }

    /// Set the analyzer
    pub fn with_analyzer(mut self, analyzer: impl Into<String>) -> Self {
        self.analyzer = Some(analyzer.into());
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!(
            "match:{}:{}:{:?}",
            self.field, self.text, self.operator
        )
    }

    /// Analyze the query text and return terms
    pub fn analyze(&self, ctx: &QueryContext) -> Vec<String> {
        ctx.tokenizer().tokenize(&self.text)
    }
}

impl QueryNode for MatchQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        // Analyze the query text into terms
        let terms = self.analyze(ctx);

        if terms.is_empty() {
            return Ok(RoaringBitmap::new());
        }

        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // Get posting bitmaps for each term
            let bitmaps: Vec<RoaringBitmap> = terms
                .iter()
                .map(|term| ctx.get_postings_bitmap(term))
                .collect();

            if bitmaps.is_empty() {
                return Ok(RoaringBitmap::new());
            }

            // Combine based on operator
            let result = match self.operator {
                MatchOperator::And => {
                    // Intersect all bitmaps
                    bitmaps
                        .into_iter()
                        .reduce(|a, b| a & b)
                        .unwrap_or_default()
                }
                MatchOperator::Or => {
                    // Union all bitmaps
                    bitmaps
                        .into_iter()
                        .reduce(|a, b| a | b)
                        .unwrap_or_default()
                }
            };

            Ok(result)
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        let terms = self.analyze(ctx);

        if terms.is_empty() {
            return 0.0;
        }

        // Estimate based on term frequencies
        let term_costs: Vec<f64> = terms
            .iter()
            .map(|term| {
                let key = format!("term:{}:{}", self.field, term);
                let freq = ctx.doc_frequency(&key);
                if freq > 0 {
                    freq as f64
                } else {
                    ctx.total_docs() as f64 * 0.1
                }
            })
            .collect();

        match self.operator {
            // AND: cost is minimum of all term costs (most selective)
            MatchOperator::And => term_costs.iter().cloned().fold(f64::MAX, f64::min),
            // OR: cost is sum of all term costs (union)
            MatchOperator::Or => term_costs.iter().sum(),
        }
    }

    fn query_type(&self) -> &'static str {
        "match"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, ctx: &QueryContext, docno: u32) -> Option<f32> {
        // Full BM25 scoring - aggregate scores from all matching terms
        let terms = self.analyze(ctx);
        let avgdl = ctx.avg_doc_length().max(1.0);
        let k1 = 1.2f32;
        let b = 0.75f32;

        let mut total_score = 0.0f32;

        for term in &terms {
            let stats = ctx.get_term_stats(term);
            if stats.doc_frequency == 0 {
                continue;
            }

            // Find this term's posting for docno
            let postings = ctx.get_postings(term);
            if let Some(posting) = postings.iter().find(|p| p.docno.as_u32() == docno) {
                let idf = ctx.bm25_idf(stats.doc_frequency);
                let tf = posting.term_frequency as f32;
                let dl = posting.doc_length as f32;
                let norm = 1.0 - b + b * (dl / avgdl);
                total_score += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
            }
        }

        if total_score > 0.0 {
            Some(total_score * self.boost)
        } else {
            None
        }
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
    fn test_match_query_creation() {
        let query = MatchQuery::new("content", "rust programming");
        assert_eq!(query.field, "content");
        assert_eq!(query.text, "rust programming");
        assert_eq!(query.operator, MatchOperator::Or);
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_match_query_builder() {
        let query = MatchQuery::new("content", "rust")
            .with_and_operator()
            .with_boost(2.0)
            .with_analyzer("english");

        assert_eq!(query.operator, MatchOperator::And);
        assert_eq!(query.boost, 2.0);
        assert_eq!(query.analyzer, Some("english".to_string()));
    }

    #[test]
    fn test_match_query_analyze() {
        let ctx = create_test_context();
        let query = MatchQuery::new("content", "Rust Programming");

        let terms = query.analyze(&ctx);
        // Should be lowercased and stemmed
        assert!(!terms.is_empty());
    }

    #[test]
    fn test_match_query_execute() {
        let ctx = create_test_context();
        let query = MatchQuery::new("content", "rust");
        let result = query.execute(&ctx).unwrap();

        // Currently returns empty bitmap (placeholder)
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_match_query_type() {
        let query = MatchQuery::new("content", "rust");
        assert_eq!(query.query_type(), "match");
        assert!(query.is_scoring());
    }

    #[test]
    fn test_match_query_cache_key() {
        let query = MatchQuery::new("content", "rust").with_and_operator();
        assert!(query.cache_key().contains("match:content:rust"));
    }
}
