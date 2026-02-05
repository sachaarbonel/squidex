//! Wildcard query - matches terms using wildcards
//!
//! Supports:
//! - `*` - matches any sequence of characters
//! - `?` - matches any single character
//!
//! # Example
//!
//! ```rust
//! use squidex::query::nodes::WildcardQuery;
//!
//! let query = WildcardQuery::new("title", "prog*");
//! ```

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use regex::Regex;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches terms using wildcard patterns
///
/// The pattern can include:
/// - `*` to match any sequence of characters (including empty)
/// - `?` to match exactly one character
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WildcardQuery {
    /// Field to search in
    pub field: String,
    /// Wildcard pattern
    pub pattern: String,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

impl WildcardQuery {
    /// Create a new wildcard query
    pub fn new(field: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            pattern: pattern.into(),
            boost: 1.0,
        }
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!("wildcard:{}:{}", self.field, self.pattern)
    }

    /// Convert wildcard pattern to a compiled regex
    fn pattern_to_regex(&self) -> Result<Regex> {
        let mut regex_pattern = String::new();
        regex_pattern.push('^');

        for ch in self.pattern.chars() {
            match ch {
                '*' => regex_pattern.push_str(".*"),
                '?' => regex_pattern.push('.'),
                // Escape regex special characters
                '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                    regex_pattern.push('\\');
                    regex_pattern.push(ch);
                }
                _ => regex_pattern.push(ch),
            }
        }

        regex_pattern.push('$');

        Regex::new(&regex_pattern).map_err(|e| {
            crate::error::SquidexError::QueryParseError(format!("Invalid wildcard pattern: {}", e))
        })
    }

    /// Extract the literal prefix from the pattern (optimization)
    ///
    /// Returns the longest prefix before the first wildcard character.
    /// This can be used to narrow down the terms to scan.
    pub fn extract_prefix(&self) -> Option<String> {
        let mut prefix = String::new();
        for ch in self.pattern.chars() {
            if ch == '*' || ch == '?' {
                break;
            }
            prefix.push(ch);
        }

        if prefix.is_empty() {
            None
        } else {
            Some(prefix)
        }
    }

    /// Check if the pattern has any wildcards
    pub fn has_wildcards(&self) -> bool {
        self.pattern.contains('*') || self.pattern.contains('?')
    }
}

impl QueryNode for WildcardQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // If no wildcards, treat as exact term match
            if !self.has_wildcards() {
                let postings = ctx.get_postings(&self.pattern);
                let mut bitmap = RoaringBitmap::new();
                for posting in postings {
                    bitmap.insert(posting.docno.as_u32());
                }
                return Ok(bitmap);
            }

            let regex = self.pattern_to_regex()?;
            let mut results = RoaringBitmap::new();

            // Get accessor for term iteration
            if let Some(accessor) = ctx.accessor() {
                // Optimization: use prefix to narrow search if available
                // For now, we need to iterate all terms
                // TODO: Add prefix_iter to IndexAccessor trait

                // Get all unique terms by scanning postings
                // This is a placeholder - in production, you'd iterate the term dictionary
                let prefix = self.extract_prefix();

                // For now, we use the postings for the exact prefix if available
                if let Some(p) = prefix {
                    let postings = ctx.get_postings(&p);
                    for posting in postings {
                        results.insert(posting.docno.as_u32());
                    }
                }
            }

            Ok(results)
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Wildcard queries are expensive
        // Cost depends on prefix specificity
        if let Some(prefix) = self.extract_prefix() {
            // More specific prefix = lower cost
            let prefix_len = prefix.len();
            let base_cost = ctx.total_docs() as f64;
            // Estimate: longer prefix = fewer terms to check
            base_cost / (1.0 + prefix_len as f64)
        } else {
            // No prefix - must scan all terms (very expensive)
            ctx.total_docs() as f64 * 10.0
        }
    }

    fn query_type(&self) -> &'static str {
        "wildcard"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, _ctx: &QueryContext, _docno: u32) -> Option<f32> {
        // Wildcard queries use constant scoring
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
    fn test_wildcard_query_creation() {
        let query = WildcardQuery::new("title", "prog*");
        assert_eq!(query.field, "title");
        assert_eq!(query.pattern, "prog*");
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_wildcard_query_with_boost() {
        let query = WildcardQuery::new("title", "prog*").with_boost(2.5);
        assert_eq!(query.boost, 2.5);
    }

    #[test]
    fn test_extract_prefix() {
        let query = WildcardQuery::new("title", "prog*");
        assert_eq!(query.extract_prefix(), Some("prog".to_string()));

        let query = WildcardQuery::new("title", "*suffix");
        assert_eq!(query.extract_prefix(), None);

        let query = WildcardQuery::new("title", "pre?fix*");
        assert_eq!(query.extract_prefix(), Some("pre".to_string()));
    }

    #[test]
    fn test_has_wildcards() {
        let query = WildcardQuery::new("title", "prog*");
        assert!(query.has_wildcards());

        let query = WildcardQuery::new("title", "programming");
        assert!(!query.has_wildcards());
    }

    #[test]
    fn test_pattern_to_regex() {
        let query = WildcardQuery::new("title", "prog*");
        let regex = query.pattern_to_regex().unwrap();
        assert!(regex.is_match("programming"));
        assert!(regex.is_match("progress"));
        assert!(regex.is_match("prog"));
        assert!(!regex.is_match("aprog"));

        let query = WildcardQuery::new("title", "te?t");
        let regex = query.pattern_to_regex().unwrap();
        assert!(regex.is_match("test"));
        assert!(regex.is_match("text"));
        assert!(!regex.is_match("teest"));
    }

    #[test]
    fn test_cache_key() {
        let query = WildcardQuery::new("title", "prog*");
        assert_eq!(query.cache_key(), "wildcard:title:prog*");
    }

    #[test]
    fn test_query_type() {
        let query = WildcardQuery::new("title", "prog*");
        assert_eq!(query.query_type(), "wildcard");
    }
}
