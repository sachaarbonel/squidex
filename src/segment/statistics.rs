//! Segment statistics for BM25+ scoring
//!
//! per segment store `doc_len` (bitpacked), `avgdl`, and `doc_count`.
//! Mutable buffers maintain their own stats; segment merges recompute stats on write.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{DocNo, DocumentId};

/// BM25+ parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bm25Params {
    /// Term frequency saturation parameter
    pub k1: f32,
    /// Length normalization parameter
    pub b: f32,
    /// BM25+ delta parameter (avoids zero scores for high-frequency terms)
    pub delta: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            delta: 1.0,
        }
    }
}

/// Statistics for a single segment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentStatistics {
    /// Total number of documents in this segment
    pub doc_count: u32,
    /// Sum of all document lengths (for computing avgdl)
    pub total_doc_length: u64,
    /// Document lengths indexed by docno (bitpacked in file format)
    doc_lengths: Vec<u32>,
    /// Cached average document length
    avgdl: f64,
}

impl SegmentStatistics {
    pub fn new() -> Self {
        Self {
            doc_count: 0,
            total_doc_length: 0,
            doc_lengths: Vec::new(),
            avgdl: 0.0,
        }
    }

    /// Create statistics with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            doc_count: 0,
            total_doc_length: 0,
            doc_lengths: Vec::with_capacity(capacity),
            avgdl: 0.0,
        }
    }

    /// Add a document with the given length
    pub fn add_document(&mut self, doc_len: u32) -> DocNo {
        let docno = DocNo::new(self.doc_count);
        self.doc_lengths.push(doc_len);
        self.total_doc_length += doc_len as u64;
        self.doc_count += 1;
        self.update_avgdl();
        docno
    }

    /// Get document length for a docno
    pub fn get_doc_length(&self, docno: DocNo) -> Option<u32> {
        self.doc_lengths.get(docno.as_usize()).copied()
    }

    /// Get average document length
    pub fn avgdl(&self) -> f64 {
        self.avgdl
    }

    /// Update cached average document length
    fn update_avgdl(&mut self) {
        if self.doc_count > 0 {
            self.avgdl = self.total_doc_length as f64 / self.doc_count as f64;
        } else {
            self.avgdl = 0.0;
        }
    }

    /// Compute BM25 score for a term occurrence
    pub fn bm25_score(
        &self,
        tf: f32,
        df: u32,
        total_docs: u32,
        doc_len: u32,
        params: &Bm25Params,
    ) -> f32 {
        let avgdl = self.avgdl as f32;
        if avgdl == 0.0 || total_docs == 0 {
            return 0.0;
        }

        // IDF with Robertson-Sparck-Jones formula
        let n = total_docs as f32;
        let df = df as f32;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // Length normalization
        let doc_len = doc_len as f32;
        let norm = 1.0 - params.b + params.b * (doc_len / avgdl);

        // BM25+ formula (with delta to avoid zero scores)
        let tf_component = (tf * (params.k1 + 1.0)) / (tf + params.k1 * norm);
        idf * (tf_component + params.delta)
    }

    /// Compute IDF for a term
    pub fn idf(&self, df: u32, total_docs: u32) -> f32 {
        let n = total_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Merge statistics from another segment
    pub fn merge(&mut self, other: &SegmentStatistics) {
        self.doc_lengths.extend(other.doc_lengths.iter().copied());
        self.doc_count += other.doc_count;
        self.total_doc_length += other.total_doc_length;
        self.update_avgdl();
    }

    /// Get all document lengths (for serialization)
    pub fn doc_lengths(&self) -> &[u32] {
        &self.doc_lengths
    }

    /// Create from serialized document lengths
    pub fn from_doc_lengths(doc_lengths: Vec<u32>) -> Self {
        let doc_count = doc_lengths.len() as u32;
        let total_doc_length: u64 = doc_lengths.iter().map(|&l| l as u64).sum();
        let avgdl = if doc_count > 0 {
            total_doc_length as f64 / doc_count as f64
        } else {
            0.0
        };

        Self {
            doc_count,
            total_doc_length,
            doc_lengths,
            avgdl,
        }
    }
}

impl Default for SegmentStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated statistics across all segments for query planning
#[derive(Clone, Debug)]
pub struct IndexStatistics {
    /// Total documents across all segments
    pub total_docs: u32,
    /// Global average document length
    pub global_avgdl: f64,
    /// Per-term document frequencies (aggregated)
    term_dfs: HashMap<String, u32>,
}

impl IndexStatistics {
    pub fn new() -> Self {
        Self {
            total_docs: 0,
            global_avgdl: 0.0,
            term_dfs: HashMap::new(),
        }
    }

    /// Aggregate statistics from multiple segments
    pub fn aggregate(segments: &[SegmentStatistics]) -> Self {
        let total_docs: u32 = segments.iter().map(|s| s.doc_count).sum();
        let total_length: u64 = segments.iter().map(|s| s.total_doc_length).sum();
        let global_avgdl = if total_docs > 0 {
            total_length as f64 / total_docs as f64
        } else {
            0.0
        };

        Self {
            total_docs,
            global_avgdl,
            term_dfs: HashMap::new(),
        }
    }

    /// Add term document frequency
    pub fn add_term_df(&mut self, term: &str, df: u32) {
        *self.term_dfs.entry(term.to_string()).or_insert(0) += df;
    }

    /// Get term document frequency
    pub fn get_term_df(&self, term: &str) -> u32 {
        self.term_dfs.get(term).copied().unwrap_or(0)
    }

    /// Compute IDF for query planning
    pub fn idf(&self, term: &str) -> f32 {
        let df = self.get_term_df(term) as f32;
        let n = self.total_docs as f32;
        if n == 0.0 {
            return 0.0;
        }
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
}

impl Default for IndexStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_statistics() {
        let mut stats = SegmentStatistics::new();

        // Add documents with varying lengths
        let docno1 = stats.add_document(100);
        let docno2 = stats.add_document(200);
        let docno3 = stats.add_document(150);

        assert_eq!(stats.doc_count, 3);
        assert_eq!(stats.total_doc_length, 450);
        assert!((stats.avgdl() - 150.0).abs() < 0.001);

        assert_eq!(stats.get_doc_length(docno1), Some(100));
        assert_eq!(stats.get_doc_length(docno2), Some(200));
        assert_eq!(stats.get_doc_length(docno3), Some(150));
    }

    #[test]
    fn test_bm25_score() {
        let mut stats = SegmentStatistics::new();
        for _ in 0..100 {
            stats.add_document(100);
        }

        let params = Bm25Params::default();

        // Score for a term with TF=5 in a doc with length 100
        let score = stats.bm25_score(5.0, 10, 100, 100, &params);
        assert!(score > 0.0);

        // Higher TF should give higher score
        let score_low_tf = stats.bm25_score(1.0, 10, 100, 100, &params);
        let score_high_tf = stats.bm25_score(5.0, 10, 100, 100, &params);
        assert!(score_high_tf > score_low_tf);

        // Rarer terms (lower DF) should have higher scores
        let score_common = stats.bm25_score(5.0, 50, 100, 100, &params);
        let score_rare = stats.bm25_score(5.0, 5, 100, 100, &params);
        assert!(score_rare > score_common);
    }

    #[test]
    fn test_merge_statistics() {
        let mut stats1 = SegmentStatistics::new();
        stats1.add_document(100);
        stats1.add_document(200);

        let mut stats2 = SegmentStatistics::new();
        stats2.add_document(150);
        stats2.add_document(250);

        stats1.merge(&stats2);

        assert_eq!(stats1.doc_count, 4);
        assert_eq!(stats1.total_doc_length, 700);
        assert!((stats1.avgdl() - 175.0).abs() < 0.001);
    }

    #[test]
    fn test_index_statistics() {
        let mut stats1 = SegmentStatistics::new();
        for _ in 0..50 {
            stats1.add_document(100);
        }

        let mut stats2 = SegmentStatistics::new();
        for _ in 0..50 {
            stats2.add_document(150);
        }

        let index_stats = IndexStatistics::aggregate(&[stats1, stats2]);
        assert_eq!(index_stats.total_docs, 100);
        assert!((index_stats.global_avgdl - 125.0).abs() < 0.001);
    }
}
