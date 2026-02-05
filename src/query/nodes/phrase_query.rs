//! Phrase query - matches exact phrases with optional proximity/slop
//!
//! A phrase query matches documents containing the exact sequence of terms,
//! optionally allowing for a number of intervening terms (slop).
//!
//! # Example
//!
//! ```rust
//! use squidex::query::nodes::PhraseQuery;
//!
//! // Exact phrase match
//! let query = PhraseQuery::new("content", "rust programming");
//!
//! // Phrase with slop (allows 2 terms between)
//! let query = PhraseQuery::new("content", "rust programming").with_slop(2);
//! ```

use crate::query::ast::QueryNode;
use crate::query::context::QueryContext;
use crate::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Query that matches an exact phrase of terms
///
/// The phrase is tokenized and all terms must appear in the document
/// in the specified order. The `slop` parameter allows for flexibility
/// in term positions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhraseQuery {
    /// Field to search in
    pub field: String,
    /// The phrase to match (will be tokenized)
    pub phrase: String,
    /// Maximum number of positions between terms (default: 0 for exact phrase)
    #[serde(default)]
    pub slop: u32,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

impl PhraseQuery {
    /// Create a new phrase query with exact matching (slop=0)
    pub fn new(field: impl Into<String>, phrase: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            phrase: phrase.into(),
            slop: 0,
            boost: 1.0,
        }
    }

    /// Set the slop (maximum positions between terms)
    ///
    /// - slop=0: exact phrase match (terms must be adjacent)
    /// - slop=1: one term can appear between phrase terms
    /// - slop=2: two terms can appear between phrase terms, etc.
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }

    /// Set the boost factor
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the cache key for this query
    pub fn cache_key(&self) -> String {
        format!("phrase:{}:{}:{}", self.field, self.phrase, self.slop)
    }
}

impl QueryNode for PhraseQuery {
    fn execute(&self, ctx: &QueryContext) -> Result<RoaringBitmap> {
        let cache_key = self.cache_key();
        ctx.get_or_cache_filter(&cache_key, || {
            // Tokenize the phrase
            let tokenizer = ctx.tokenizer();
            let terms = tokenizer.tokenize(&self.phrase);

            if terms.is_empty() {
                return Ok(RoaringBitmap::new());
            }

            // Single term - just do a term lookup
            if terms.len() == 1 {
                let postings = ctx.get_postings(&terms[0]);
                let mut bitmap = RoaringBitmap::new();
                for posting in postings {
                    bitmap.insert(posting.docno.as_u32());
                }
                return Ok(bitmap);
            }

            // Multiple terms - find intersection first
            // Then verify positions (when position data is available)
            let mut result: Option<RoaringBitmap> = None;

            for term in &terms {
                let postings = ctx.get_postings(term);
                let mut term_bitmap = RoaringBitmap::new();
                for posting in postings {
                    term_bitmap.insert(posting.docno.as_u32());
                }

                result = Some(match result {
                    Some(r) => r & term_bitmap,
                    None => term_bitmap,
                });

                // Early termination if no matches
                if result.as_ref().map(|r| r.is_empty()).unwrap_or(false) {
                    return Ok(RoaringBitmap::new());
                }
            }

            // TODO: When position data is available, verify phrase positions
            // For now, we return documents containing all terms
            // This is an approximation that may include false positives

            Ok(result.unwrap_or_default())
        })
    }

    fn estimate_cost(&self, ctx: &QueryContext) -> f64 {
        // Cost is based on the rarest term in the phrase
        let tokenizer = ctx.tokenizer();
        let terms = tokenizer.tokenize(&self.phrase);

        if terms.is_empty() {
            return 0.0;
        }

        // Find minimum document frequency among all terms
        let min_df = terms
            .iter()
            .map(|t| ctx.doc_frequency(t))
            .filter(|&df| df > 0)
            .min()
            .unwrap_or(0);

        // Phrase queries have additional cost for position checking
        let position_check_cost = if self.slop == 0 { 2.0 } else { 3.0 + self.slop as f64 };

        (min_df as f64) * position_check_cost
    }

    fn query_type(&self) -> &'static str {
        "phrase"
    }

    fn is_scoring(&self) -> bool {
        true
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn score(&self, ctx: &QueryContext, docno: u32) -> Option<f32> {
        // Score based on term frequencies
        let tokenizer = ctx.tokenizer();
        let terms = tokenizer.tokenize(&self.phrase);

        if terms.is_empty() {
            return None;
        }

        // Calculate score as sum of BM25 scores for each term
        let mut total_score = 0.0;

        for term in &terms {
            let stats = ctx.get_term_stats(term);
            if stats.doc_frequency == 0 {
                continue;
            }

            let postings = ctx.get_postings(term);
            if let Some(posting) = postings.iter().find(|p| p.docno.as_u32() == docno) {
                let idf = ctx.bm25_idf(stats.doc_frequency);
                let tf = posting.term_frequency as f32;
                let dl = posting.doc_length as f32;
                let avgdl = ctx.avg_doc_length();
                let k1 = 1.2f32;
                let b = 0.75f32;

                let norm = 1.0 - b + b * (dl / avgdl.max(1.0));
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

/// Helper function to check if positions form a valid phrase
///
/// Returns true if the positions in `term_positions` can form a phrase
/// with the given slop tolerance.
#[allow(dead_code)]
fn positions_form_phrase(term_positions: &[Vec<u32>], slop: u32) -> bool {
    if term_positions.is_empty() {
        return true;
    }

    // Try each starting position from the first term
    for &start_pos in &term_positions[0] {
        if check_phrase_from(start_pos, &term_positions[1..], slop) {
            return true;
        }
    }

    false
}

/// Recursively check if remaining terms can form a phrase from given position
#[allow(dead_code)]
fn check_phrase_from(current_pos: u32, remaining: &[Vec<u32>], slop: u32) -> bool {
    if remaining.is_empty() {
        return true;
    }

    let expected_pos = current_pos + 1;
    let max_pos = expected_pos + slop;

    // Find a valid position for the next term
    for &pos in &remaining[0] {
        if pos >= expected_pos && pos <= max_pos {
            if check_phrase_from(pos, &remaining[1..], slop) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_query_creation() {
        let query = PhraseQuery::new("content", "rust programming");
        assert_eq!(query.field, "content");
        assert_eq!(query.phrase, "rust programming");
        assert_eq!(query.slop, 0);
        assert_eq!(query.boost, 1.0);
    }

    #[test]
    fn test_phrase_query_with_slop() {
        let query = PhraseQuery::new("content", "rust programming").with_slop(2);
        assert_eq!(query.slop, 2);
    }

    #[test]
    fn test_phrase_query_with_boost() {
        let query = PhraseQuery::new("content", "rust programming").with_boost(2.5);
        assert_eq!(query.boost, 2.5);
    }

    #[test]
    fn test_cache_key() {
        let query = PhraseQuery::new("content", "rust programming").with_slop(2);
        assert_eq!(query.cache_key(), "phrase:content:rust programming:2");
    }

    #[test]
    fn test_query_type() {
        let query = PhraseQuery::new("content", "rust programming");
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_positions_form_phrase_exact() {
        // "hello world" at positions 0, 1
        let positions = vec![vec![0, 5], vec![1, 8]];
        assert!(positions_form_phrase(&positions, 0));
    }

    #[test]
    fn test_positions_form_phrase_with_slop() {
        // "hello world" where "hello" is at 0 and "world" is at 2 (one word between)
        let positions = vec![vec![0], vec![2]];
        assert!(!positions_form_phrase(&positions, 0));
        assert!(positions_form_phrase(&positions, 1));
    }

    #[test]
    fn test_positions_no_match() {
        // Terms too far apart
        let positions = vec![vec![0], vec![10]];
        assert!(!positions_form_phrase(&positions, 2));
    }
}
