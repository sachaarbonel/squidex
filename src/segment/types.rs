//! Core types for the segment-based index

use serde::{Deserialize, Serialize};
use std::fmt;

/// Segment identifier (monotonically increasing per shard)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SegmentId(pub u64);

impl SegmentId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for SegmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "segment_{}", self.0)
    }
}

/// Dense document number within a segment (0..max_doc)
/// This is used internally for efficient posting list storage
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DocNo(pub u32);

impl DocNo {
    pub const MAX: DocNo = DocNo(u32::MAX);

    pub fn new(n: u32) -> Self {
        Self(n)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Term identifier (hash of the term string for fast lookup)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TermId(pub u64);

impl TermId {
    pub fn from_term(term: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        term.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// External document ID (UUIDv7 stored as u128)
/// For simplicity, we use u64 to match the existing DocumentId type
pub type DocumentId = u64;

/// Document version for optimistic locking
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Version(pub u64);

impl Version {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
}

/// A single posting entry within a posting list
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Posting {
    /// Dense document number within the segment
    pub docno: DocNo,
    /// Term frequency in this document
    pub term_frequency: u32,
    /// Position offsets (for phrase queries, if enabled)
    pub positions: Vec<u32>,
}

impl Posting {
    pub fn new(docno: DocNo, term_frequency: u32) -> Self {
        Self {
            docno,
            term_frequency,
            positions: Vec::new(),
        }
    }

    pub fn with_positions(docno: DocNo, term_frequency: u32, positions: Vec<u32>) -> Self {
        Self {
            docno,
            term_frequency,
            positions,
        }
    }
}

/// Block of postings (fixed size for SIMD-friendly processing)
/// 128 or 256 docs per block
pub const BLOCK_SIZE: usize = 128;

/// A block of postings with skip data and impact metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostingBlock {
    /// Document numbers (delta-encoded when serialized)
    pub docnos: Vec<DocNo>,
    /// Term frequencies
    pub term_frequencies: Vec<u32>,
    /// Maximum document number in this block (for skip data)
    pub max_docno: DocNo,
    /// Maximum term frequency in this block (for WAND/MaxScore)
    pub max_tf: u32,
    /// Precomputed max score contribution for this block (optional)
    pub max_score: Option<f32>,
}

impl PostingBlock {
    pub fn new() -> Self {
        Self {
            docnos: Vec::with_capacity(BLOCK_SIZE),
            term_frequencies: Vec::with_capacity(BLOCK_SIZE),
            max_docno: DocNo(0),
            max_tf: 0,
            max_score: None,
        }
    }

    pub fn is_full(&self) -> bool {
        self.docnos.len() >= BLOCK_SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.docnos.is_empty()
    }

    pub fn len(&self) -> usize {
        self.docnos.len()
    }

    pub fn push(&mut self, posting: Posting) {
        if posting.docno > self.max_docno {
            self.max_docno = posting.docno;
        }
        if posting.term_frequency > self.max_tf {
            self.max_tf = posting.term_frequency;
        }
        self.docnos.push(posting.docno);
        self.term_frequencies.push(posting.term_frequency);
    }

    /// Compute max score for WAND/MaxScore optimization
    pub fn compute_max_score(&mut self, idf: f32, k1: f32, b: f32, avgdl: f32) {
        // Assume shortest doc length for max score (most pessimistic normalization)
        // This gives an upper bound on the possible score
        let min_doc_len = 1.0; // Conservative estimate
        let norm = 1.0 - b + b * (min_doc_len / avgdl);
        let tf = self.max_tf as f32;
        let score = idf * (tf * (k1 + 1.0)) / (tf + k1 * norm);
        self.max_score = Some(score);
    }
}

impl Default for PostingBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Posting list metadata stored in the term dictionary
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostingListMeta {
    /// Offset in the postings file
    pub offset: u64,
    /// Length in bytes
    pub length: u64,
    /// Document frequency (number of documents containing this term)
    pub doc_frequency: u32,
    /// Total term frequency across all documents
    pub total_term_frequency: u64,
}

/// Document value row for columnar storage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocValueRow {
    /// Numeric field values
    pub numerics: Vec<Option<i64>>,
    /// Boolean field values
    pub booleans: Vec<Option<bool>>,
    /// Keyword field values (stored as string for simplicity)
    pub keywords: Vec<Option<String>>,
}

impl DocValueRow {
    pub fn new() -> Self {
        Self {
            numerics: Vec::new(),
            booleans: Vec::new(),
            keywords: Vec::new(),
        }
    }
}

impl Default for DocValueRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry in the document number mapping
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DocNoEntry {
    /// External document ID
    pub doc_id: DocumentId,
    /// Document version
    pub version: Version,
}

impl DocNoEntry {
    pub fn new(doc_id: DocumentId, version: Version) -> Self {
        Self { doc_id, version }
    }
}

/// Raft index for tracking visibility
pub type RaftIndex = u64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_id() {
        let id = SegmentId::new(42);
        assert_eq!(id.0, 42);
        assert_eq!(id.next().0, 43);
        assert_eq!(format!("{}", id), "segment_42");
    }

    #[test]
    fn test_docno() {
        let docno = DocNo::new(100);
        assert_eq!(docno.as_u32(), 100);
        assert_eq!(docno.as_usize(), 100);
    }

    #[test]
    fn test_term_id() {
        let term_id1 = TermId::from_term("hello");
        let term_id2 = TermId::from_term("hello");
        let term_id3 = TermId::from_term("world");

        assert_eq!(term_id1, term_id2);
        assert_ne!(term_id1, term_id3);
    }

    #[test]
    fn test_posting_block() {
        let mut block = PostingBlock::new();
        assert!(block.is_empty());
        assert!(!block.is_full());

        block.push(Posting::new(DocNo(1), 5));
        block.push(Posting::new(DocNo(10), 3));

        assert_eq!(block.len(), 2);
        assert_eq!(block.max_docno, DocNo(10));
        assert_eq!(block.max_tf, 5);
    }
}
