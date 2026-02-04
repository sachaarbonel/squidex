//! Query planner for optimizing query execution
//!
//! The query planner analyzes the query AST and produces an optimized
//! execution plan.

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;

/// Query execution plan
#[derive(Debug)]
pub struct QueryPlan {
    /// Root query node (potentially optimized)
    pub root: Box<dyn QueryNode>,
    /// Estimated total cost
    pub estimated_cost: f64,
    /// Whether the query uses scoring
    pub uses_scoring: bool,
    /// Optimization hints applied
    pub optimizations: Vec<String>,
}

/// Query planner for optimizing query execution
pub struct QueryPlanner;

impl QueryPlanner {
    /// Create an optimized execution plan for a query
    pub fn plan(query: Box<dyn QueryNode>, ctx: &QueryContext) -> QueryPlan {
        let mut optimizations = Vec::new();

        // Apply optimizations
        let optimized = Self::optimize(query, ctx, &mut optimizations);

        // Calculate estimated cost
        let estimated_cost = optimized.estimate_cost(ctx);
        let uses_scoring = optimized.is_scoring();

        QueryPlan {
            root: optimized,
            estimated_cost,
            uses_scoring,
            optimizations,
        }
    }

    /// Apply optimizations to a query
    fn optimize(
        query: Box<dyn QueryNode>,
        ctx: &QueryContext,
        optimizations: &mut Vec<String>,
    ) -> Box<dyn QueryNode> {
        let query_type = query.query_type();

        match query_type {
            "bool" => Self::optimize_bool(query, ctx, optimizations),
            _ => query, // No optimization for other query types yet
        }
    }

    /// Optimize a boolean query
    fn optimize_bool(
        query: Box<dyn QueryNode>,
        ctx: &QueryContext,
        optimizations: &mut Vec<String>,
    ) -> Box<dyn QueryNode> {
        // Clone the query to modify it
        let mut bool_query = query.clone_box();

        // The clone_box returns a Box<dyn QueryNode>, but we need to check
        // if it's actually a BoolQuery. Since we can't downcast easily with
        // the current trait design, we'll apply optimizations that work
        // generically.

        // For now, just note that we analyzed the bool query
        optimizations.push("analyzed_bool_query".to_string());

        // Reorder clauses by cost would happen here if we could downcast
        // to BoolQuery. For now, the BoolQuery itself has an optimize method.

        bool_query
    }

    /// Analyze query for potential optimizations
    pub fn analyze(query: &dyn QueryNode, ctx: &QueryContext) -> QueryAnalysis {
        let estimated_cost = query.estimate_cost(ctx);
        let uses_scoring = query.is_scoring();
        let query_type = query.query_type().to_string();

        let suggestions = Self::suggest_optimizations(query, ctx, estimated_cost);

        QueryAnalysis {
            query_type,
            estimated_cost,
            uses_scoring,
            suggestions,
        }
    }

    /// Suggest optimizations for a query
    fn suggest_optimizations(
        query: &dyn QueryNode,
        ctx: &QueryContext,
        estimated_cost: f64,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // High cost query warning
        let total_docs = ctx.total_docs() as f64;
        if estimated_cost > total_docs * 0.5 {
            suggestions.push(
                "Query estimated to scan >50% of documents. Consider adding more selective filters."
                    .to_string(),
            );
        }

        // Non-scoring query could use filter context
        if !query.is_scoring() && query.query_type() != "range" {
            suggestions.push(
                "Non-scoring query could benefit from filter context for caching.".to_string(),
            );
        }

        suggestions
    }
}

/// Analysis result for a query
#[derive(Debug)]
pub struct QueryAnalysis {
    /// Type of the root query node
    pub query_type: String,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Whether the query uses scoring
    pub uses_scoring: bool,
    /// Optimization suggestions
    pub suggestions: Vec<String>,
}

impl QueryAnalysis {
    /// Check if the query is likely to be expensive
    pub fn is_expensive(&self, total_docs: usize) -> bool {
        self.estimated_cost > total_docs as f64 * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::nodes::{BoolQuery, MatchQuery, RangeQuery, TermQuery};
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
    fn test_plan_simple_query() {
        let ctx = create_test_context();
        let query: Box<dyn QueryNode> = Box::new(MatchQuery::new("content", "rust"));

        let plan = QueryPlanner::plan(query, &ctx);

        assert_eq!(plan.root.query_type(), "match");
        assert!(plan.uses_scoring);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_plan_bool_query() {
        let ctx = create_test_context();
        let query: Box<dyn QueryNode> = Box::new(
            BoolQuery::new()
                .must(MatchQuery::new("content", "rust"))
                .filter(RangeQuery::new("year").gte(2024)),
        );

        let plan = QueryPlanner::plan(query, &ctx);

        assert_eq!(plan.root.query_type(), "bool");
        assert!(!plan.optimizations.is_empty());
    }

    #[test]
    fn test_analyze_query() {
        let ctx = create_test_context();
        let query = TermQuery::new("status", "published");

        let analysis = QueryPlanner::analyze(&query, &ctx);

        assert_eq!(analysis.query_type, "term");
        assert!(analysis.uses_scoring);
    }

    #[test]
    fn test_analyze_expensive_query() {
        let ctx = QueryContext::builder()
            .total_docs(100)
            .avg_doc_length(100.0)
            .tokenizer(Arc::new(Tokenizer::new(&TokenizerConfig::default())))
            .build();

        // A match query with no doc frequency info is assumed to match 10% of docs
        let query = MatchQuery::new("content", "common term");
        let analysis = QueryPlanner::analyze(&query, &ctx);

        // 10 estimated docs out of 100 = not expensive
        assert!(!analysis.is_expensive(100));
    }
}
