//! Range query - matches documents with field values in a range

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::query::types::RangeBounds;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches documents with field values within a specified range
///
/// Works with numeric fields (Long, Double) and date fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeQuery {
    /// Field to search in
    pub field: String,
    /// Range bounds (gte, gt, lte, lt)
    #[serde(flatten)]
    pub bounds: RangeBounds,
}

impl RangeQuery {
    /// Create a new range query
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            bounds: RangeBounds::default(),
        }
    }

    /// Set the greater-than-or-equal bound
    pub fn gte(mut self, value: i64) -> Self {
        self.bounds.gte = Some(crate::query::types::RangeValue::Long(value));
        self
    }

    /// Set the greater-than bound
    pub fn gt(mut self, value: i64) -> Self {
        self.bounds.gt = Some(crate::query::types::RangeValue::Long(value));
        self
    }

    /// Set the less-than-or-equal bound
    pub fn lte(mut self, value: i64) -> Self {
        self.bounds.lte = Some(crate::query::types::RangeValue::Long(value));
        self
    }

    /// Set the less-than bound
    pub fn lt(mut self, value: i64) -> Self {
        self.bounds.lt = Some(crate::query::types::RangeValue::Long(value));
        self
    }

    /// Set the greater-than-or-equal bound (float)
    pub fn gte_f64(mut self, value: f64) -> Self {
        self.bounds.gte = Some(crate::query::types::RangeValue::Double(value));
        self
    }

    /// Set the greater-than bound (float)
    pub fn gt_f64(mut self, value: f64) -> Self {
        self.bounds.gt = Some(crate::query::types::RangeValue::Double(value));
        self
    }

    /// Set the less-than-or-equal bound (float)
    pub fn lte_f64(mut self, value: f64) -> Self {
        self.bounds.lte = Some(crate::query::types::RangeValue::Double(value));
        self
    }

    /// Set the less-than bound (float)
    pub fn lt_f64(mut self, value: f64) -> Self {
        self.bounds.lt = Some(crate::query::types::RangeValue::Double(value));
        self
    }

    /// Set the bounds from a RangeBounds struct
    pub fn with_bounds(mut self, bounds: RangeBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.bounds.boost = boost;
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!(
            "range:{}:gte={:?}:gt={:?}:lte={:?}:lt={:?}",
            self.field, self.bounds.gte, self.bounds.gt, self.bounds.lte, self.bounds.lt
        )
    }

    /// Check if this is a point query (gte == lte)
    pub fn is_point_query(&self) -> bool {
        match (&self.bounds.gte, &self.bounds.lte) {
            (Some(gte), Some(lte)) => gte == lte,
            _ => false,
        }
    }

    /// Check if this range is unbounded on the lower end
    pub fn is_unbounded_lower(&self) -> bool {
        self.bounds.gte.is_none() && self.bounds.gt.is_none()
    }

    /// Check if this range is unbounded on the upper end
    pub fn is_unbounded_upper(&self) -> bool {
        self.bounds.lte.is_none() && self.bounds.lt.is_none()
    }
}

impl QueryNode for RangeQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // Real implementation would:
            // 1. Read the doc values column for this field
            // 2. Scan the column and filter by range predicate
            // 3. Build bitmap of matching document numbers
            // 4. Filter out tombstones

            // For now, return empty bitmap (placeholder)
            Ok(RoaringBitmap::new())
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Range queries typically scan doc values
        // Cost depends on selectivity of the range

        // Estimate selectivity based on bounds
        let selectivity = if self.is_point_query() {
            0.01 // Point query is very selective
        } else if self.is_unbounded_lower() || self.is_unbounded_upper() {
            0.5 // Half-open range matches ~50%
        } else {
            0.25 // Bounded range matches ~25%
        };

        ctx.total_docs() as f64 * selectivity
    }

    fn query_type(&self) -> &'static str {
        "range"
    }

    fn is_scoring(&self) -> bool {
        // Range queries typically don't contribute to relevance
        false
    }

    fn boost(&self) -> f32 {
        self.bounds.boost
    }

    fn score(&self, _ctx: &QueryContext, _docno: u32) -> Option<f32> {
        // Range queries are typically used as filters and don't score
        Some(self.bounds.boost)
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
    fn test_range_query_creation() {
        let query = RangeQuery::new("price").gte(100).lte(500);
        assert_eq!(query.field, "price");
        assert!(query.bounds.gte.is_some());
        assert!(query.bounds.lte.is_some());
    }

    #[test]
    fn test_range_query_builder() {
        let query = RangeQuery::new("age")
            .gt(18)
            .lt(65)
            .with_boost(2.0);

        assert!(query.bounds.gt.is_some());
        assert!(query.bounds.lt.is_some());
        assert_eq!(query.bounds.boost, 2.0);
    }

    #[test]
    fn test_range_query_float() {
        let query = RangeQuery::new("score")
            .gte_f64(0.5)
            .lte_f64(1.0);

        assert!(query.bounds.gte.is_some());
        assert!(query.bounds.lte.is_some());
    }

    #[test]
    fn test_range_query_point() {
        use crate::query::types::RangeValue;

        let query = RangeQuery::new("year")
            .with_bounds(RangeBounds {
                gte: Some(RangeValue::Long(2024)),
                lte: Some(RangeValue::Long(2024)),
                ..Default::default()
            });

        assert!(query.is_point_query());
    }

    #[test]
    fn test_range_query_unbounded() {
        let query_lower = RangeQuery::new("price").lte(100);
        assert!(query_lower.is_unbounded_lower());
        assert!(!query_lower.is_unbounded_upper());

        let query_upper = RangeQuery::new("price").gte(100);
        assert!(!query_upper.is_unbounded_lower());
        assert!(query_upper.is_unbounded_upper());
    }

    #[test]
    fn test_range_query_execute() {
        let ctx = create_test_context();
        let query = RangeQuery::new("price").gte(100).lte(500);
        let result = query.execute(&ctx).unwrap();

        // Currently returns empty bitmap (placeholder)
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_range_query_type() {
        let query = RangeQuery::new("price");
        assert_eq!(query.query_type(), "range");
        assert!(!query.is_scoring()); // Range queries don't score
    }

    #[test]
    fn test_range_query_cache_key() {
        let query = RangeQuery::new("price").gte(100).lte(500);
        let cache_key = query.cache_key();
        assert!(cache_key.contains("range:price"));
    }
}
