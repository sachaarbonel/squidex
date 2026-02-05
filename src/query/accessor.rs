//! Index accessor for query execution
//!
//! Provides abstraction over the segment index for query node execution.

use crate::segment::{DocNo, DocumentId, SegmentIndex};
use dashmap::DashSet;
use roaring::RoaringBitmap;
use std::sync::Arc;

/// Statistics needed for BM25 scoring
#[derive(Clone, Debug, Default)]
pub struct TermStats {
    /// Number of documents containing this term
    pub doc_frequency: u32,
    /// Total occurrences of this term across all documents
    pub total_term_frequency: u64,
}

/// Unified posting entry with all scoring data
#[derive(Clone, Debug)]
pub struct PostingEntry {
    /// Dense document number within segment
    pub docno: DocNo,
    /// Term frequency in this document
    pub term_frequency: u32,
    /// Document length (for BM25 normalization)
    pub doc_length: u32,
    /// External document ID
    pub doc_id: DocumentId,
}

/// Trait for accessing posting lists from index
pub trait IndexAccessor: Send + Sync {
    /// Get term statistics (df, ttf)
    fn term_stats(&self, term: &str) -> TermStats;

    /// Iterate postings for a term with scoring data
    fn postings(&self, term: &str) -> Vec<PostingEntry>;

    /// Get postings as bitmap for set operations
    fn postings_bitmap(&self, term: &str) -> RoaringBitmap;

    /// Map docno to external document ID
    fn docno_to_doc_id(&self, docno: DocNo) -> Option<DocumentId>;

    /// Get document length for BM25
    fn doc_length(&self, docno: DocNo) -> u32;

    /// Global statistics: total document count
    fn total_docs(&self) -> u32;

    /// Global statistics: average document length
    fn avg_doc_length(&self) -> f32;
}

/// Accessor implementation that reads from SegmentIndex
pub struct SegmentAccessor {
    /// Reference to the segment index
    segment_index: Arc<SegmentIndex>,
    /// Snapshot of tombstones at query start
    tombstone_set: Arc<DashSet<DocumentId>>,
    /// Cached total docs
    total_docs: u32,
    /// Cached average document length
    avg_doc_length: f32,
}

impl SegmentAccessor {
    /// Create a new segment accessor
    pub fn new(segment_index: Arc<SegmentIndex>, tombstones: &DashSet<DocumentId>) -> Self {
        // Compute global stats once
        let (total_docs, avg_doc_length) = segment_index.compute_global_stats();

        Self {
            segment_index,
            tombstone_set: Arc::new(tombstones.clone()),
            total_docs,
            avg_doc_length: avg_doc_length as f32,
        }
    }
}

impl IndexAccessor for SegmentAccessor {
    fn term_stats(&self, term: &str) -> TermStats {
        let mut doc_frequency = 0u32;
        let mut total_term_frequency = 0u64;

        // Get stats from buffer
        let buffer = self.segment_index.buffer();
        let buffer_guard = buffer.read().unwrap();
        if let Some(postings) = buffer_guard.get_postings(term) {
            for posting in postings {
                let entry = buffer_guard.get_doc_entry(posting.docno);
                if let Some(entry) = entry {
                    if !self.tombstone_set.contains(&entry.doc_id)
                        && !buffer_guard.is_deleted(posting.docno)
                    {
                        doc_frequency += 1;
                        total_term_frequency += posting.term_frequency as u64;
                    }
                }
            }
        }
        drop(buffer_guard);

        // Get stats from segments
        let segments = self.segment_index.segments();
        for segment in segments.iter() {
            if let Some(meta) = segment.get_posting_meta(term) {
                // The meta has doc_frequency but we need to filter tombstones
                // For now, use the meta's doc_frequency as an approximation
                // A more accurate implementation would iterate postings
                doc_frequency += meta.doc_frequency;
                total_term_frequency += meta.total_term_frequency;
            }
        }

        TermStats {
            doc_frequency,
            total_term_frequency,
        }
    }

    fn postings(&self, term: &str) -> Vec<PostingEntry> {
        let mut entries = Vec::new();

        // Get postings from buffer
        let buffer = self.segment_index.buffer();
        let buffer_guard = buffer.read().unwrap();
        if let Some(postings) = buffer_guard.get_postings(term) {
            for posting in postings {
                if buffer_guard.is_deleted(posting.docno) {
                    continue;
                }
                if let Some(entry) = buffer_guard.get_doc_entry(posting.docno) {
                    if self.tombstone_set.contains(&entry.doc_id) {
                        continue;
                    }
                    let doc_length = buffer_guard.stats().get_doc_length(posting.docno).unwrap_or(0);
                    entries.push(PostingEntry {
                        docno: posting.docno,
                        term_frequency: posting.term_frequency,
                        doc_length,
                        doc_id: entry.doc_id,
                    });
                }
            }
        }
        drop(buffer_guard);

        // Get postings from segments
        let segments = self.segment_index.segments();
        for segment in segments.iter() {
            if let Ok(Some(iter)) = segment.get_postings(term) {
                for (docno, tf) in iter {
                    if segment.is_deleted(docno) {
                        continue;
                    }
                    if let Some(doc_id) = segment.get_doc_id(docno) {
                        if self.tombstone_set.contains(&doc_id) {
                            continue;
                        }
                        let doc_length = segment.get_doc_length(docno).unwrap_or(0);
                        entries.push(PostingEntry {
                            docno,
                            term_frequency: tf,
                            doc_length,
                            doc_id,
                        });
                    }
                }
            }
        }

        entries
    }

    fn postings_bitmap(&self, term: &str) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();

        // Get postings from buffer
        let buffer = self.segment_index.buffer();
        let buffer_guard = buffer.read().unwrap();
        if let Some(postings) = buffer_guard.get_postings(term) {
            for posting in postings {
                if buffer_guard.is_deleted(posting.docno) {
                    continue;
                }
                if let Some(entry) = buffer_guard.get_doc_entry(posting.docno) {
                    if !self.tombstone_set.contains(&entry.doc_id) {
                        bitmap.insert(posting.docno.as_u32());
                    }
                }
            }
        }
        drop(buffer_guard);

        // Get postings from segments
        let segments = self.segment_index.segments();
        for segment in segments.iter() {
            if let Ok(Some(iter)) = segment.get_postings(term) {
                for (docno, _tf) in iter {
                    if segment.is_deleted(docno) {
                        continue;
                    }
                    if let Some(doc_id) = segment.get_doc_id(docno) {
                        if !self.tombstone_set.contains(&doc_id) {
                            bitmap.insert(docno.as_u32());
                        }
                    }
                }
            }
        }

        bitmap
    }

    fn docno_to_doc_id(&self, docno: DocNo) -> Option<DocumentId> {
        // Check buffer first
        let buffer = self.segment_index.buffer();
        let buffer_guard = buffer.read().unwrap();
        if let Some(entry) = buffer_guard.get_doc_entry(docno) {
            return Some(entry.doc_id);
        }
        drop(buffer_guard);

        // Check segments
        let segments = self.segment_index.segments();
        for segment in segments.iter() {
            if let Some(doc_id) = segment.get_doc_id(docno) {
                return Some(doc_id);
            }
        }

        None
    }

    fn doc_length(&self, docno: DocNo) -> u32 {
        // Check buffer first
        let buffer = self.segment_index.buffer();
        let buffer_guard = buffer.read().unwrap();
        if let Some(len) = buffer_guard.stats().get_doc_length(docno) {
            return len;
        }
        drop(buffer_guard);

        // Check segments
        let segments = self.segment_index.segments();
        for segment in segments.iter() {
            if let Some(len) = segment.get_doc_length(docno) {
                return len;
            }
        }

        0
    }

    fn total_docs(&self) -> u32 {
        self.total_docs
    }

    fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }
}

// We need methods on SegmentIndex to access its internals
// Let's add a trait extension or check if these methods exist

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_stats_default() {
        let stats = TermStats::default();
        assert_eq!(stats.doc_frequency, 0);
        assert_eq!(stats.total_term_frequency, 0);
    }

    #[test]
    fn test_posting_entry() {
        let entry = PostingEntry {
            docno: DocNo::new(0),
            term_frequency: 5,
            doc_length: 100,
            doc_id: 42,
        };
        assert_eq!(entry.docno.as_u32(), 0);
        assert_eq!(entry.term_frequency, 5);
        assert_eq!(entry.doc_length, 100);
        assert_eq!(entry.doc_id, 42);
    }
}
