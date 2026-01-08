use serde::{Deserialize, Serialize};

use super::document::{DocumentId, Embedding};
use super::filter::Filter;

/// Search result with relevance score
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: DocumentId,
    pub score: f32,
    pub highlights: Vec<String>,
}

/// Search request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: Option<String>,        // Keyword query
    pub embedding: Option<Embedding>, // Vector query
    pub filters: Vec<Filter>,         // Metadata filters
    pub top_k: usize,                 // Number of results
    pub search_mode: SearchMode,      // Keyword, Vector, or Hybrid
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: None,
            embedding: None,
            filters: Vec::new(),
            top_k: 10,
            search_mode: SearchMode::Hybrid {
                keyword_weight: 0.5,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SearchMode {
    Keyword,
    Vector,
    Hybrid { keyword_weight: f32 },
}

/// Similarity metrics for vector search
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Search response with timing information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub took_ms: u64,
    pub total_hits: u64,
}

impl SearchResult {
    pub fn new(doc_id: DocumentId, score: f32) -> Self {
        Self {
            doc_id,
            score,
            highlights: Vec::new(),
        }
    }

    pub fn with_highlights(mut self, highlights: Vec<String>) -> Self {
        self.highlights = highlights;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_request_default() {
        let req = SearchRequest::default();
        assert_eq!(req.top_k, 10);
        assert!(matches!(req.search_mode, SearchMode::Hybrid { .. }));
    }

    #[test]
    fn test_search_result_builder() {
        let result = SearchResult::new(42, 0.95).with_highlights(vec!["highlight".to_string()]);
        assert_eq!(result.doc_id, 42);
        assert_eq!(result.score, 0.95);
        assert_eq!(result.highlights.len(), 1);
    }
}
