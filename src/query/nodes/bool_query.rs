//! Boolean query - combines multiple clauses with AND, OR, NOT semantics

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::query::types::MinimumShouldMatch;
use crate::Result;
use roaring::RoaringBitmap;

/// Boolean query combining multiple clauses
///
/// The boolean query supports four types of clauses:
/// - `must`: All clauses must match (AND). Contributes to score.
/// - `should`: At least one clause should match (OR). Contributes to score.
/// - `must_not`: No clause must match (NOT). Does not contribute to score.
/// - `filter`: All clauses must match (AND). Does not contribute to score. Cached.
///
/// # Example
///
/// ```json
/// {
///   "bool": {
///     "must": [
///       { "match": { "content": "rust programming" } }
///     ],
///     "should": [
///       { "term": { "tags": "tutorial" } }
///     ],
///     "must_not": [
///       { "term": { "status": "draft" } }
///     ],
///     "filter": [
///       { "range": { "created_at": { "gte": "2024-01-01" } } }
///     ]
///   }
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct BoolQuery {
    /// Clauses that must match (AND, scoring)
    pub must: Vec<Box<dyn QueryNode>>,
    /// Clauses where at least one should match (OR, scoring)
    pub should: Vec<Box<dyn QueryNode>>,
    /// Clauses that must not match (NOT, no scoring)
    pub must_not: Vec<Box<dyn QueryNode>>,
    /// Clauses that must match (AND, no scoring, cached)
    pub filter: Vec<Box<dyn QueryNode>>,
    /// Minimum number of should clauses that must match
    pub minimum_should_match: MinimumShouldMatch,
    /// Boost factor for scoring
    pub boost: f32,
}

impl BoolQuery {
    /// Create a new empty boolean query
    pub fn new() -> Self {
        Self {
            must: Vec::new(),
            should: Vec::new(),
            must_not: Vec::new(),
            filter: Vec::new(),
            minimum_should_match: MinimumShouldMatch::default(),
            boost: 1.0,
        }
    }

    /// Add a must clause
    pub fn must(mut self, query: impl QueryNode + 'static) -> Self {
        self.must.push(Box::new(query));
        self
    }

    /// Add a should clause
    pub fn should(mut self, query: impl QueryNode + 'static) -> Self {
        self.should.push(Box::new(query));
        self
    }

    /// Add a must_not clause
    pub fn must_not(mut self, query: impl QueryNode + 'static) -> Self {
        self.must_not.push(Box::new(query));
        self
    }

    /// Add a filter clause
    pub fn filter(mut self, query: impl QueryNode + 'static) -> Self {
        self.filter.push(Box::new(query));
        self
    }

    /// Add a must clause (boxed)
    pub fn must_boxed(mut self, query: Box<dyn QueryNode>) -> Self {
        self.must.push(query);
        self
    }

    /// Add a should clause (boxed)
    pub fn should_boxed(mut self, query: Box<dyn QueryNode>) -> Self {
        self.should.push(query);
        self
    }

    /// Add a must_not clause (boxed)
    pub fn must_not_boxed(mut self, query: Box<dyn QueryNode>) -> Self {
        self.must_not.push(query);
        self
    }

    /// Add a filter clause (boxed)
    pub fn filter_boxed(mut self, query: Box<dyn QueryNode>) -> Self {
        self.filter.push(query);
        self
    }

    /// Set minimum should match
    pub fn with_minimum_should_match(mut self, msm: MinimumShouldMatch) -> Self {
        self.minimum_should_match = msm;
        self
    }

    /// Set boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Check if this is an empty query
    pub fn is_empty(&self) -> bool {
        self.must.is_empty()
            && self.should.is_empty()
            && self.must_not.is_empty()
            && self.filter.is_empty()
    }

    /// Get total number of clauses
    pub fn clause_count(&self) -> usize {
        self.must.len() + self.should.len() + self.must_not.len() + self.filter.len()
    }

    /// Reorder clauses by estimated cost (cheapest first)
    pub fn optimize_clause_order(&mut self, ctx: &QueryContext) {
        // Sort must clauses by cost (cheapest first for early termination)
        self.must.sort_by(|a, b| {
            a.estimate_cost(ctx)
                .partial_cmp(&b.estimate_cost(ctx))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort filter clauses by cost (cheapest first for early termination)
        self.filter.sort_by(|a, b| {
            a.estimate_cost(ctx)
                .partial_cmp(&b.estimate_cost(ctx))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Don't reorder should clauses as it might affect scoring order
    }
}

impl QueryNode for BoolQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        // Handle empty query
        if self.is_empty() {
            // Empty bool query matches all documents
            let total_docs = ctx.total_docs();
            let mut result = RoaringBitmap::new();
            for i in 0..total_docs as u32 {
                result.insert(i);
            }
            result -= ctx.tombstones();
            return Ok(result);
        }

        let mut result: Option<RoaringBitmap> = None;

        // Execute FILTER clauses first (most likely to be cached and selective)
        for query in &self.filter {
            let matches = query.execute(ctx)?;
            result = Some(match result {
                Some(r) => r & matches, // Intersection
                None => matches,
            });

            // Early exit if no matches
            if let Some(ref r) = result {
                if r.is_empty() {
                    return Ok(RoaringBitmap::new());
                }
            }
        }

        // Execute MUST clauses (all required, scoring)
        for query in &self.must {
            let matches = query.execute(ctx)?;
            result = Some(match result {
                Some(r) => r & matches, // Intersection
                None => matches,
            });

            // Early exit if no matches
            if let Some(ref r) = result {
                if r.is_empty() {
                    return Ok(RoaringBitmap::new());
                }
            }
        }

        // Execute SHOULD clauses
        if !self.should.is_empty() {
            let mut should_matches = RoaringBitmap::new();
            let mut match_counts: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();

            for query in &self.should {
                let matches = query.execute(ctx)?;
                for docno in matches.iter() {
                    *match_counts.entry(docno).or_insert(0) += 1;
                }
                should_matches |= matches; // Union
            }

            // Apply minimum_should_match
            let min_match = self.minimum_should_match.calculate(self.should.len());
            if min_match > 1 {
                // Filter to only docs matching >= min_match clauses
                let filtered: RoaringBitmap = match_counts
                    .iter()
                    .filter(|(_, &count)| count >= min_match)
                    .map(|(&docno, _)| docno)
                    .collect();
                should_matches = filtered;
            }

            // Combine with result
            if self.must.is_empty() && self.filter.is_empty() {
                // No must/filter clauses: should clauses determine matches
                result = Some(should_matches);
            } else {
                // Must/filter clauses present: should only boosts score
                // Docs must already match must/filter to be included
                // (the should matches are used for scoring, not filtering)
                // Keep the intersection
                if let Some(ref r) = result {
                    // Only keep docs that also have at least one should match
                    // Actually, in Elasticsearch, should doesn't filter when must/filter present
                    // It only affects scoring. So we keep the result as-is.
                    // The scoring will be applied separately.
                }
            }
        }

        // Execute MUST_NOT clauses (exclusion)
        for query in &self.must_not {
            let matches = query.execute(ctx)?;
            if let Some(ref mut r) = result {
                *r -= matches; // Difference
            }
        }

        // Remove tombstones from final result
        let mut final_result = result.unwrap_or_default();
        final_result -= ctx.tombstones();

        Ok(final_result)
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost is dominated by the most selective clause
        let must_cost: f64 = self
            .must
            .iter()
            .map(|q| q.estimate_cost(ctx))
            .fold(f64::MAX, f64::min);

        let filter_cost: f64 = self
            .filter
            .iter()
            .map(|q| q.estimate_cost(ctx))
            .fold(f64::MAX, f64::min);

        let should_cost: f64 = self.should.iter().map(|q| q.estimate_cost(ctx)).sum();

        let must_not_cost: f64 = self.must_not.iter().map(|q| q.estimate_cost(ctx)).sum();

        // Base cost is minimum of must/filter costs
        let base_cost = if must_cost < f64::MAX || filter_cost < f64::MAX {
            must_cost.min(filter_cost)
        } else if should_cost > 0.0 {
            should_cost
        } else {
            ctx.total_docs() as f64
        };

        // Add should and must_not costs (they process the base result)
        base_cost + should_cost * 0.1 + must_not_cost * 0.1
    }

    fn query_type(&self) -> &'static str {
        "bool"
    }

    fn is_scoring(&self) -> bool {
        // Bool query scores if any must or should clause scores
        self.must.iter().any(|q| q.is_scoring()) || self.should.iter().any(|q| q.is_scoring())
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, ctx: &QueryContext, docno: u32) -> Option<f32> {
        let mut total_score = 0.0f32;

        // Sum scores from must clauses
        for query in &self.must {
            if let Some(score) = query.score(ctx, docno) {
                total_score += score;
            }
        }

        // Sum scores from should clauses
        for query in &self.should {
            if let Some(score) = query.score(ctx, docno) {
                total_score += score;
            }
        }

        if total_score > 0.0 {
            Some(total_score * self.boost)
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(BoolQuery {
            must: self.must.iter().map(|q| q.clone_box()).collect(),
            should: self.should.iter().map(|q| q.clone_box()).collect(),
            must_not: self.must_not.iter().map(|q| q.clone_box()).collect(),
            filter: self.filter.iter().map(|q| q.clone_box()).collect(),
            minimum_should_match: self.minimum_should_match.clone(),
            boost: self.boost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::nodes::{MatchQuery, RangeQuery, TermQuery};
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
    fn test_bool_query_creation() {
        let query = BoolQuery::new()
            .must(MatchQuery::new("content", "rust"))
            .should(TermQuery::new("tags", "tutorial"))
            .must_not(TermQuery::new("status", "draft"))
            .filter(RangeQuery::new("year").gte(2024));

        assert_eq!(query.must.len(), 1);
        assert_eq!(query.should.len(), 1);
        assert_eq!(query.must_not.len(), 1);
        assert_eq!(query.filter.len(), 1);
        assert_eq!(query.clause_count(), 4);
    }

    #[test]
    fn test_bool_query_empty() {
        let ctx = create_test_context();
        let query = BoolQuery::new();
        let result = query.execute(&ctx).unwrap();

        // Empty bool matches all docs
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_bool_query_minimum_should_match() {
        let query = BoolQuery::new()
            .should(TermQuery::new("tags", "rust"))
            .should(TermQuery::new("tags", "programming"))
            .should(TermQuery::new("tags", "tutorial"))
            .with_minimum_should_match(MinimumShouldMatch::Count(2));

        assert_eq!(query.minimum_should_match.calculate(3), 2);
    }

    #[test]
    fn test_bool_query_type() {
        let query = BoolQuery::new();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_bool_query_scoring() {
        let scoring_query = BoolQuery::new().must(MatchQuery::new("content", "rust"));

        let non_scoring_query = BoolQuery::new().filter(RangeQuery::new("year").gte(2024));

        assert!(scoring_query.is_scoring());
        assert!(!non_scoring_query.is_scoring());
    }

    #[test]
    fn test_bool_query_optimize() {
        let ctx = create_test_context();
        let mut query = BoolQuery::new()
            .must(MatchQuery::new("content", "very common term"))
            .must(TermQuery::new("id", "specific_id")); // Should be more selective

        query.optimize_clause_order(&ctx);
        // After optimization, order might change based on estimated costs
        assert_eq!(query.must.len(), 2);
    }

    #[test]
    fn test_bool_query_clone() {
        let query = BoolQuery::new()
            .must(MatchQuery::new("content", "rust"))
            .with_boost(2.0);

        let cloned = query.clone_box();
        assert_eq!(cloned.query_type(), "bool");
        assert_eq!(cloned.boost(), 2.0);
    }
}
