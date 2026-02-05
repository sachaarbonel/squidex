//! Fuzzy query - matches terms within an edit distance
//!
//! Uses Levenshtein distance to find terms that are similar to the query term.
//!
//! # Example
//!
//! ```rust
//! use squidex::query::nodes::FuzzyQuery;
//!
//! // Find terms within edit distance 2 of "roust" (matches "rust")
//! let query = FuzzyQuery::new("content", "roust").with_fuzziness(2);
//! ```

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches terms within an edit distance of the query term
///
/// The edit distance is calculated using Levenshtein distance, counting:
/// - Insertions
/// - Deletions
/// - Substitutions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuzzyQuery {
    /// Field to search in
    pub field: String,
    /// Term to match approximately
    pub term: String,
    /// Maximum edit distance (default: 2)
    #[serde(default = "default_fuzziness")]
    pub fuzziness: u32,
    /// Number of initial characters that must match exactly (default: 0)
    #[serde(default)]
    pub prefix_length: usize,
    /// Maximum number of terms to consider (default: 50)
    #[serde(default = "default_max_expansions")]
    pub max_expansions: usize,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_fuzziness() -> u32 {
    2
}

fn default_max_expansions() -> usize {
    50
}

fn default_boost() -> f32 {
    1.0
}

impl FuzzyQuery {
    /// Create a new fuzzy query with default fuzziness of 2
    pub fn new(field: impl Into<String>, term: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            term: term.into(),
            fuzziness: 2,
            prefix_length: 0,
            max_expansions: 50,
            boost: 1.0,
        }
    }

    /// Set the maximum edit distance
    pub fn with_fuzziness(mut self, fuzziness: u32) -> Self {
        self.fuzziness = fuzziness;
        self
    }

    /// Set the number of initial characters that must match exactly
    pub fn with_prefix_length(mut self, prefix_length: usize) -> Self {
        self.prefix_length = prefix_length;
        self
    }

    /// Set the maximum number of terms to consider
    pub fn with_max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!(
            "fuzzy:{}:{}:{}:{}",
            self.field, self.term, self.fuzziness, self.prefix_length
        )
    }

    /// Get the prefix that must match exactly
    pub fn required_prefix(&self) -> &str {
        let end = self.prefix_length.min(self.term.len());
        &self.term[..end]
    }
}

impl QueryNode for FuzzyQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            let mut results = RoaringBitmap::new();

            // If fuzziness is 0, this is just an exact match
            if self.fuzziness == 0 {
                let postings = ctx.get_postings(&self.term);
                for posting in postings {
                    results.insert(posting.docno.as_u32());
                }
                return Ok(results);
            }

            // Get accessor for term iteration
            if let Some(_accessor) = ctx.accessor() {
                // Optimization: if prefix_length > 0, only check terms with matching prefix
                let prefix = self.required_prefix();

                // For now, we check the exact term and common variations
                // In production, this would iterate the term dictionary

                // Check exact match
                let postings = ctx.get_postings(&self.term);
                for posting in postings {
                    results.insert(posting.docno.as_u32());
                }

                // TODO: Iterate term dictionary and filter by Levenshtein distance
                // This requires adding term iteration capability to IndexAccessor
            }

            Ok(results)
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Fuzzy queries are expensive - must scan many terms
        let base_cost = ctx.total_docs() as f64;

        // Higher fuzziness = more terms to check
        let fuzz_factor = 1.0 + self.fuzziness as f64;

        // Longer prefix = fewer terms to check
        let prefix_factor = if self.prefix_length > 0 {
            1.0 / (1.0 + self.prefix_length as f64)
        } else {
            1.0
        };

        base_cost * fuzz_factor * prefix_factor * 0.5
    }

    fn query_type(&self) -> &'static str {
        "fuzzy"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, ctx: &QueryContext, docno: u32) -> Option<f32> {
        // Score based on edit distance (closer = higher score)
        // For now, use constant scoring
        let postings = ctx.get_postings(&self.term);
        let has_match = postings.iter().any(|p| p.docno.as_u32() == docno);

        if has_match {
            Some(self.boost)
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn QueryNode> {
        Box::new(self.clone())
    }
}

/// Calculate Levenshtein distance between two strings
///
/// Uses dynamic programming with O(m*n) time and O(min(m,n)) space.
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    // Early termination for empty strings
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Use smaller string for columns to minimize space
    let (shorter, longer, short_len, long_len) = if len1 <= len2 {
        (&s1_chars, &s2_chars, len1, len2)
    } else {
        (&s2_chars, &s1_chars, len2, len1)
    };

    // Two rows for DP
    let mut prev_row: Vec<usize> = (0..=short_len).collect();
    let mut curr_row = vec![0; short_len + 1];

    for i in 1..=long_len {
        curr_row[0] = i;

        for j in 1..=short_len {
            let cost = if longer[i - 1] == shorter[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = std::cmp::min(
                std::cmp::min(
                    prev_row[j] + 1,     // deletion
                    curr_row[j - 1] + 1, // insertion
                ),
                prev_row[j - 1] + cost, // substitution
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[short_len]
}

/// Calculate Damerau-Levenshtein distance (includes transpositions)
///
/// This allows adjacent character swaps as a single operation.
pub fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    // Full matrix for Damerau-Levenshtein
    let mut matrix = vec![vec![0usize; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = std::cmp::min(
                std::cmp::min(
                    matrix[i - 1][j] + 1,     // deletion
                    matrix[i][j - 1] + 1,     // insertion
                ),
                matrix[i - 1][j - 1] + cost, // substitution
            );

            // Transposition
            if i > 1
                && j > 1
                && s1_chars[i - 1] == s2_chars[j - 2]
                && s1_chars[i - 2] == s2_chars[j - 1]
            {
                matrix[i][j] = std::cmp::min(matrix[i][j], matrix[i - 2][j - 2] + cost);
            }
        }
    }

    matrix[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_query_creation() {
        let query = FuzzyQuery::new("content", "rust");
        assert_eq!(query.field, "content");
        assert_eq!(query.term, "rust");
        assert_eq!(query.fuzziness, 2);
        assert_eq!(query.prefix_length, 0);
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_fuzzy_query_with_options() {
        let query = FuzzyQuery::new("content", "rust")
            .with_fuzziness(1)
            .with_prefix_length(2)
            .with_boost(1.5);

        assert_eq!(query.fuzziness, 1);
        assert_eq!(query.prefix_length, 2);
        assert_eq!(query.boost, 1.5);
    }

    #[test]
    fn test_levenshtein_distance() {
        // Same strings
        assert_eq!(levenshtein_distance("rust", "rust"), 0);

        // Single character operations
        assert_eq!(levenshtein_distance("rust", "just"), 1); // substitution
        assert_eq!(levenshtein_distance("rust", "rusts"), 1); // insertion
        assert_eq!(levenshtein_distance("rusts", "rust"), 1); // deletion

        // Multiple operations
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);

        // Empty strings
        assert_eq!(levenshtein_distance("", "test"), 4);
        assert_eq!(levenshtein_distance("test", ""), 4);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_damerau_levenshtein_distance() {
        // Same strings
        assert_eq!(damerau_levenshtein_distance("rust", "rust"), 0);

        // Transposition (should be 1 in Damerau, 2 in standard Levenshtein)
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);

        // Standard operations still work
        assert_eq!(damerau_levenshtein_distance("rust", "just"), 1);
        assert_eq!(damerau_levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_required_prefix() {
        let query = FuzzyQuery::new("content", "programming").with_prefix_length(4);
        assert_eq!(query.required_prefix(), "prog");

        let query = FuzzyQuery::new("content", "hi").with_prefix_length(10);
        assert_eq!(query.required_prefix(), "hi");
    }

    #[test]
    fn test_cache_key() {
        let query = FuzzyQuery::new("content", "rust").with_fuzziness(1);
        assert_eq!(query.cache_key(), "fuzzy:content:rust:1:0");
    }

    #[test]
    fn test_query_type() {
        let query = FuzzyQuery::new("content", "rust");
        assert_eq!(query.query_type(), "fuzzy");
    }
}
