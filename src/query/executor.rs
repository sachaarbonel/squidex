//! Query executor for running queries against the index
//!
//! The executor takes a query plan and executes it, producing search results.

use crate::models::SearchResult;
use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::query::planner::{QueryPlan, QueryPlanner};
use crate::query::types::QueryStats;
use crate::Result;
use roaring::RoaringBitmap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Query execution result
#[derive(Debug)]
pub struct QueryResult {
    /// Matching documents with scores
    pub hits: Vec<SearchResult>,
    /// Total number of matching documents
    pub total_hits: u64,
    /// Execution statistics
    pub stats: QueryStats,
}

/// Query executor for running queries
pub struct QueryExecutor;

impl QueryExecutor {
    /// Execute a query and return results
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute
    /// * `ctx` - Query execution context
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Query results including hits, total count, and statistics
    pub fn execute(
        query: Box<dyn QueryNode>,
        ctx: &QueryContext,
        top_k: usize,
    ) -> Result<QueryResult> {
        let start = Instant::now();

        // Create execution plan
        let plan = QueryPlanner::plan(query, ctx);

        // Execute the query
        let matches = plan.root.execute(ctx)?;
        let total_hits = matches.len() as u64;

        // Score and collect top-k results
        let hits = if plan.uses_scoring {
            Self::collect_top_k_scored(plan.root.as_ref(), ctx, &matches, top_k)
        } else {
            Self::collect_top_k_unscored(ctx, &matches, top_k)
        };

        let stats = QueryStats {
            docs_matched: total_hits,
            postings_read: 0, // Would be tracked during execution
            filter_cache_hits: 0,
            filter_cache_misses: 0,
            execution_time_us: start.elapsed().as_micros() as u64,
        };

        Ok(QueryResult {
            hits,
            total_hits,
            stats,
        })
    }

    /// Execute a query with a pre-built plan
    pub fn execute_plan(
        plan: &QueryPlan,
        ctx: &QueryContext,
        top_k: usize,
    ) -> Result<QueryResult> {
        let start = Instant::now();

        // Execute the query
        let matches = plan.root.execute(ctx)?;
        let total_hits = matches.len() as u64;

        // Score and collect top-k results
        let hits = if plan.uses_scoring {
            Self::collect_top_k_scored(plan.root.as_ref(), ctx, &matches, top_k)
        } else {
            Self::collect_top_k_unscored(ctx, &matches, top_k)
        };

        let stats = QueryStats {
            docs_matched: total_hits,
            postings_read: 0,
            filter_cache_hits: 0,
            filter_cache_misses: 0,
            execution_time_us: start.elapsed().as_micros() as u64,
        };

        Ok(QueryResult {
            hits,
            total_hits,
            stats,
        })
    }

    /// Collect top-k results with scoring
    fn collect_top_k_scored(
        query: &dyn QueryNode,
        ctx: &QueryContext,
        matches: &RoaringBitmap,
        top_k: usize,
    ) -> Vec<SearchResult> {
        if matches.is_empty() {
            return Vec::new();
        }

        // Use a min-heap to collect top-k highest scores
        // The heap contains (score, docno) where lower scores are at the top
        let mut heap: BinaryHeap<Reverse<(OrderedFloat, u32)>> =
            BinaryHeap::with_capacity(top_k + 1);

        for docno in matches.iter() {
            let score = query.score(ctx, docno).unwrap_or(0.0);

            if heap.len() < top_k {
                heap.push(Reverse((OrderedFloat(score), docno)));
            } else if let Some(&Reverse((OrderedFloat(min_score), _))) = heap.peek() {
                if score > min_score {
                    heap.pop();
                    heap.push(Reverse((OrderedFloat(score), docno)));
                }
            }
        }

        // Extract results in descending score order
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|Reverse((OrderedFloat(score), docno))| {
                let doc_id = ctx.docno_to_doc_id(crate::segment::DocNo(docno)).unwrap_or(docno as u64);
                SearchResult::new(doc_id, score)
            })
            .collect();

        // Sort by descending score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Collect top-k results without scoring (e.g., filter-only queries)
    fn collect_top_k_unscored(
        ctx: &QueryContext,
        matches: &RoaringBitmap,
        top_k: usize,
    ) -> Vec<SearchResult> {
        matches
            .iter()
            .take(top_k)
            .map(|docno| {
                let doc_id = ctx.docno_to_doc_id(crate::segment::DocNo(docno)).unwrap_or(docno as u64);
                SearchResult::new(doc_id, 1.0) // Uniform score for non-scoring queries
            })
            .collect()
    }
}

/// Wrapper for f32 that implements Ord for use in BinaryHeap
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::nodes::{AllDocsQuery, BoolQuery, MatchQuery, RangeQuery};
    use crate::tokenizer::Tokenizer;
    use crate::TokenizerConfig;
    use std::sync::Arc;

    fn create_test_context() -> QueryContext {
        let tokenizer = Arc::new(Tokenizer::new(&TokenizerConfig::default()));
        QueryContext::builder()
            .total_docs(100)
            .avg_doc_length(50.0)
            .tokenizer(tokenizer)
            .build()
    }

    #[test]
    fn test_execute_all_docs() {
        let ctx = create_test_context();
        let query: Box<dyn QueryNode> = Box::new(AllDocsQuery::new());

        let result = QueryExecutor::execute(query, &ctx, 10).unwrap();

        // AllDocsQuery should match all 100 documents
        assert_eq!(result.total_hits, 100);
        assert_eq!(result.hits.len(), 10); // Only top 10 returned
    }

    #[test]
    fn test_execute_empty_result() {
        let ctx = create_test_context();
        // Match query returns empty bitmap in placeholder implementation
        let query: Box<dyn QueryNode> = Box::new(MatchQuery::new("content", "nonexistent"));

        let result = QueryExecutor::execute(query, &ctx, 10).unwrap();

        assert_eq!(result.total_hits, 0);
        assert!(result.hits.is_empty());
    }

    #[test]
    fn test_execute_bool_query() {
        let ctx = create_test_context();
        let query: Box<dyn QueryNode> = Box::new(
            BoolQuery::new()
                .filter(AllDocsQuery::new())
                .must(MatchQuery::new("content", "rust")),
        );

        let result = QueryExecutor::execute(query, &ctx, 10).unwrap();

        // Match query returns empty, so intersection with AllDocs is empty
        assert_eq!(result.total_hits, 0);
    }

    #[test]
    fn test_execute_with_stats() {
        let ctx = create_test_context();
        let query: Box<dyn QueryNode> = Box::new(AllDocsQuery::new());

        let result = QueryExecutor::execute(query, &ctx, 10).unwrap();

        // Stats should be populated
        assert!(result.stats.execution_time_us > 0);
        assert_eq!(result.stats.docs_matched, 100);
    }

    #[test]
    fn test_ordered_float() {
        let a = OrderedFloat(1.0);
        let b = OrderedFloat(2.0);
        let c = OrderedFloat(1.0);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
    }
}
