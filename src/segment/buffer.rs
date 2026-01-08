//! Mutable buffer for in-memory writes
//!
//! mutable buffers are sharded by `hash(doc_id)` to avoid a global lock.
//! Flush merges shard-local builders into a segment.

use std::collections::HashMap;

use super::statistics::{Bm25Params, SegmentStatistics};
use super::types::{
    DocNo, DocNoEntry, DocValueRow, DocumentId, Posting, RaftIndex, TermId, Version,
};

/// Configuration for buffer flush triggers
#[derive(Clone, Debug)]
pub struct BufferConfig {
    /// Flush when buffer size exceeds this (bytes)
    pub max_bytes: usize,
    /// Flush when document count exceeds this
    pub max_docs: usize,
    /// Flush interval in seconds
    pub flush_interval_secs: u64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            // 256MB
            max_bytes: 256 * 1024 * 1024,
            // 50k docs
            max_docs: 50_000,
            // 2s
            flush_interval_secs: 2,
        }
    }
}

/// In-memory mutable buffer for recent writes
///
/// This holds indexed data before it is flushed to an immutable segment.
#[derive(Debug)]
pub struct MutableBuffer {
    /// Term to postings mapping
    terms: HashMap<String, Vec<Posting>>,
    /// DocNo to DocValueRow mapping
    docvalues: HashMap<DocNo, DocValueRow>,
    /// DocNo to external document ID mapping
    docno_map: Vec<DocNoEntry>,
    /// Delete bitset (docnos that are deleted)
    deleted: Vec<bool>,
    /// Segment statistics
    stats: SegmentStatistics,
    /// Approximate size in bytes
    size_bytes: usize,
    /// Minimum Raft index covered
    min_raft_index: Option<RaftIndex>,
    /// Maximum Raft index covered
    max_raft_index: Option<RaftIndex>,
    /// Document ID to DocNo lookup (for updates/deletes)
    doc_id_to_docno: HashMap<DocumentId, DocNo>,
}

impl MutableBuffer {
    /// Create a new empty mutable buffer
    pub fn new() -> Self {
        Self {
            terms: HashMap::new(),
            docvalues: HashMap::new(),
            docno_map: Vec::new(),
            deleted: Vec::new(),
            stats: SegmentStatistics::new(),
            size_bytes: 0,
            min_raft_index: None,
            max_raft_index: None,
            doc_id_to_docno: HashMap::new(),
        }
    }

    /// Create a new buffer with pre-allocated capacity
    pub fn with_capacity(doc_capacity: usize, term_capacity: usize) -> Self {
        Self {
            terms: HashMap::with_capacity(term_capacity),
            docvalues: HashMap::with_capacity(doc_capacity),
            docno_map: Vec::with_capacity(doc_capacity),
            deleted: Vec::with_capacity(doc_capacity),
            stats: SegmentStatistics::with_capacity(doc_capacity),
            size_bytes: 0,
            min_raft_index: None,
            max_raft_index: None,
            doc_id_to_docno: HashMap::with_capacity(doc_capacity),
        }
    }

    /// Index a document into the buffer
    ///
    /// Returns the assigned DocNo for this document.
    pub fn index_document(
        &mut self,
        doc_id: DocumentId,
        version: Version,
        term_frequencies: HashMap<String, u32>,
        doc_len: u32,
        docvalues: Option<DocValueRow>,
        raft_index: RaftIndex,
    ) -> DocNo {
        // Allocate a new docno
        let docno = self.stats.add_document(doc_len);

        // Store document ID mapping
        let entry = DocNoEntry::new(doc_id, version);
        self.docno_map.push(entry);
        self.deleted.push(false);
        self.doc_id_to_docno.insert(doc_id, docno);

        // Index terms
        for (term, tf) in term_frequencies {
            let posting = Posting::new(docno, tf);
            self.size_bytes += std::mem::size_of::<Posting>() + term.len();
            self.terms
                .entry(term)
                .or_insert_with(Vec::new)
                .push(posting);
        }

        // Store docvalues
        if let Some(dv) = docvalues {
            self.size_bytes += std::mem::size_of::<DocValueRow>();
            self.docvalues.insert(docno, dv);
        }

        // Update Raft index range
        self.update_raft_index(raft_index);

        docno
    }

    /// Index a document with positions for phrase queries
    pub fn index_document_with_positions(
        &mut self,
        doc_id: DocumentId,
        version: Version,
        term_positions: HashMap<String, Vec<u32>>,
        doc_len: u32,
        docvalues: Option<DocValueRow>,
        raft_index: RaftIndex,
    ) -> DocNo {
        // Allocate a new docno
        let docno = self.stats.add_document(doc_len);

        // Store document ID mapping
        let entry = DocNoEntry::new(doc_id, version);
        self.docno_map.push(entry);
        self.deleted.push(false);
        self.doc_id_to_docno.insert(doc_id, docno);

        // Index terms with positions
        for (term, positions) in term_positions {
            let tf = positions.len() as u32;
            let posting = Posting::with_positions(docno, tf, positions);
            self.size_bytes +=
                std::mem::size_of::<Posting>() + term.len() + posting.positions.len() * 4;
            self.terms
                .entry(term)
                .or_insert_with(Vec::new)
                .push(posting);
        }

        // Store docvalues
        if let Some(dv) = docvalues {
            self.size_bytes += std::mem::size_of::<DocValueRow>();
            self.docvalues.insert(docno, dv);
        }

        // Update Raft index range
        self.update_raft_index(raft_index);

        docno
    }

    /// Mark a document as deleted
    pub fn delete_document(&mut self, doc_id: DocumentId, raft_index: RaftIndex) -> bool {
        if let Some(&docno) = self.doc_id_to_docno.get(&doc_id) {
            if let Some(deleted) = self.deleted.get_mut(docno.as_usize()) {
                *deleted = true;
                self.update_raft_index(raft_index);
                return true;
            }
        }
        false
    }

    /// Check if a document exists and is not deleted
    pub fn contains_document(&self, doc_id: DocumentId) -> bool {
        if let Some(&docno) = self.doc_id_to_docno.get(&doc_id) {
            if let Some(&deleted) = self.deleted.get(docno.as_usize()) {
                return !deleted;
            }
        }
        false
    }

    /// Get postings for a term
    pub fn get_postings(&self, term: &str) -> Option<&Vec<Posting>> {
        self.terms.get(term)
    }

    /// Get document frequency for a term
    pub fn doc_frequency(&self, term: &str) -> u32 {
        self.terms
            .get(term)
            .map(|postings| {
                postings
                    .iter()
                    .filter(|p| !self.is_deleted(p.docno))
                    .count() as u32
            })
            .unwrap_or(0)
    }

    /// Check if a docno is deleted
    pub fn is_deleted(&self, docno: DocNo) -> bool {
        self.deleted.get(docno.as_usize()).copied().unwrap_or(false)
    }

    /// Get document entry for a docno
    pub fn get_doc_entry(&self, docno: DocNo) -> Option<&DocNoEntry> {
        self.docno_map.get(docno.as_usize())
    }

    /// Get docvalues for a docno
    pub fn get_docvalues(&self, docno: DocNo) -> Option<&DocValueRow> {
        self.docvalues.get(&docno)
    }

    /// Get the segment statistics
    pub fn stats(&self) -> &SegmentStatistics {
        &self.stats
    }

    /// Get mutable reference to segment statistics
    pub fn stats_mut(&mut self) -> &mut SegmentStatistics {
        &mut self.stats
    }

    /// Get document count
    pub fn doc_count(&self) -> u32 {
        self.stats.doc_count
    }

    /// Get live document count (excluding deleted)
    pub fn live_doc_count(&self) -> u32 {
        self.deleted.iter().filter(|&&d| !d).count() as u32
    }

    /// Get approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Check if buffer should be flushed
    pub fn should_flush(&self, config: &BufferConfig) -> bool {
        self.size_bytes >= config.max_bytes || self.doc_count() as usize >= config.max_docs
    }

    /// Get all terms in the buffer
    pub fn terms(&self) -> impl Iterator<Item = &String> {
        self.terms.keys()
    }

    /// Get all postings in the buffer (for segment writing)
    pub fn all_postings(&self) -> &HashMap<String, Vec<Posting>> {
        &self.terms
    }

    /// Get all docno mappings
    pub fn docno_map(&self) -> &[DocNoEntry] {
        &self.docno_map
    }

    /// Get all deleted flags
    pub fn deleted_flags(&self) -> &[bool] {
        &self.deleted
    }

    /// Get all docvalues
    pub fn all_docvalues(&self) -> &HashMap<DocNo, DocValueRow> {
        &self.docvalues
    }

    /// Get the Raft index range covered by this buffer
    pub fn raft_index_range(&self) -> (Option<RaftIndex>, Option<RaftIndex>) {
        (self.min_raft_index, self.max_raft_index)
    }

    /// Clear the buffer (after flush)
    pub fn clear(&mut self) {
        self.terms.clear();
        self.docvalues.clear();
        self.docno_map.clear();
        self.deleted.clear();
        self.stats = SegmentStatistics::new();
        self.size_bytes = 0;
        self.min_raft_index = None;
        self.max_raft_index = None;
        self.doc_id_to_docno.clear();
    }

    /// Perform BM25 search over the buffer
    pub fn search(
        &self,
        query_terms: &[String],
        params: &Bm25Params,
        total_docs: u32,
    ) -> Vec<(DocNo, f32)> {
        let mut scores: HashMap<DocNo, f32> = HashMap::new();

        for term in query_terms {
            if let Some(postings) = self.terms.get(term) {
                let df = self.doc_frequency(term);
                let idf = self.stats.idf(df, total_docs);

                for posting in postings {
                    if self.is_deleted(posting.docno) {
                        continue;
                    }

                    if let Some(doc_len) = self.stats.get_doc_length(posting.docno) {
                        let score = self.stats.bm25_score(
                            posting.term_frequency as f32,
                            df,
                            total_docs,
                            doc_len,
                            params,
                        );
                        *scores.entry(posting.docno).or_insert(0.0) += score;
                    }
                }
            }
        }

        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn update_raft_index(&mut self, raft_index: RaftIndex) {
        match self.min_raft_index {
            None => self.min_raft_index = Some(raft_index),
            Some(min) if raft_index < min => self.min_raft_index = Some(raft_index),
            _ => {}
        }
        match self.max_raft_index {
            None => self.max_raft_index = Some(raft_index),
            Some(max) if raft_index > max => self.max_raft_index = Some(raft_index),
            _ => {}
        }
    }
}

impl Default for MutableBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_document() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs = HashMap::new();
        term_freqs.insert("hello".to_string(), 2);
        term_freqs.insert("world".to_string(), 1);

        let docno = buffer.index_document(1, Version::new(1), term_freqs, 100, None, 1);

        assert_eq!(docno, DocNo::new(0));
        assert_eq!(buffer.doc_count(), 1);
        assert_eq!(buffer.doc_frequency("hello"), 1);
        assert_eq!(buffer.doc_frequency("world"), 1);
        assert!(buffer.contains_document(1));
    }

    #[test]
    fn test_delete_document() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs = HashMap::new();
        term_freqs.insert("hello".to_string(), 2);

        buffer.index_document(1, Version::new(1), term_freqs, 100, None, 1);
        assert!(buffer.contains_document(1));
        assert_eq!(buffer.doc_frequency("hello"), 1);

        buffer.delete_document(1, 2);
        assert!(!buffer.contains_document(1));
        assert_eq!(buffer.doc_frequency("hello"), 0);
    }

    #[test]
    fn test_search() {
        let mut buffer = MutableBuffer::new();

        // Index two documents
        let mut term_freqs1 = HashMap::new();
        term_freqs1.insert("rust".to_string(), 5);
        term_freqs1.insert("programming".to_string(), 3);
        buffer.index_document(1, Version::new(1), term_freqs1, 100, None, 1);

        let mut term_freqs2 = HashMap::new();
        term_freqs2.insert("rust".to_string(), 2);
        term_freqs2.insert("language".to_string(), 4);
        buffer.index_document(2, Version::new(1), term_freqs2, 150, None, 2);

        // Search for "rust"
        let results = buffer.search(&["rust".to_string()], &Bm25Params::default(), 2);

        assert_eq!(results.len(), 2);
        // Doc 1 should rank higher (higher TF for "rust")
        assert_eq!(results[0].0, DocNo::new(0));
    }

    #[test]
    fn test_should_flush() {
        let mut buffer = MutableBuffer::new();
        let config = BufferConfig {
            max_bytes: 1000,
            max_docs: 10,
            flush_interval_secs: 2,
        };

        assert!(!buffer.should_flush(&config));

        // Add enough documents to trigger flush
        for i in 0..10 {
            let mut term_freqs = HashMap::new();
            term_freqs.insert(format!("term{}", i), 1);
            buffer.index_document(i as u64, Version::new(1), term_freqs, 100, None, i as u64);
        }

        assert!(buffer.should_flush(&config));
    }

    #[test]
    fn test_raft_index_range() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs = HashMap::new();
        term_freqs.insert("test".to_string(), 1);

        buffer.index_document(1, Version::new(1), term_freqs.clone(), 100, None, 10);
        buffer.index_document(2, Version::new(1), term_freqs.clone(), 100, None, 5);
        buffer.index_document(3, Version::new(1), term_freqs.clone(), 100, None, 15);

        let (min, max) = buffer.raft_index_range();
        assert_eq!(min, Some(5));
        assert_eq!(max, Some(15));
    }
}
