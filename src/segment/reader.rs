//! Immutable segment reader backed by mmapped files
//!
//! reads operate over `(mutable_buffer + segments[])` and merge top-k.
//! Each segment reader provides access to postings, term dictionary, docvalues,
//! and segment statistics.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::docno_map::DocNoMap;
use super::docvalues::DocValuesReader;
use super::postings::{PostingIterator, PostingsReader};
use super::statistics::{Bm25Params, SegmentStatistics};
use super::term_dict::TermDictionary;
use super::types::{DocNo, DocumentId, PostingListMeta, RaftIndex, SegmentId, Version};

/// Metadata for a segment stored in the manifest
#[derive(Clone, Debug)]
pub struct SegmentMeta {
    /// Unique segment identifier
    pub id: SegmentId,
    /// Minimum Raft index covered by this segment
    pub min_raft_index: RaftIndex,
    /// Maximum Raft index covered by this segment
    pub max_raft_index: RaftIndex,
    /// Number of documents in the segment
    pub doc_count: u32,
    /// Number of live (non-deleted) documents
    pub live_doc_count: u32,
    /// Size in bytes (all segment files combined)
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: u64,
}

impl SegmentMeta {
    pub fn delete_ratio(&self) -> f64 {
        if self.doc_count == 0 {
            0.0
        } else {
            1.0 - (self.live_doc_count as f64 / self.doc_count as f64)
        }
    }
}

/// Immutable segment reader backed by in-memory or mmapped data
pub struct SegmentReader {
    /// Segment metadata
    meta: SegmentMeta,
    /// Term dictionary (FST-based)
    terms: Arc<TermDictionary>,
    /// Postings reader
    postings: Arc<PostingsReader>,
    /// DocValues reader
    docvalues: Arc<DocValuesReader>,
    /// Segment statistics
    stats: Arc<SegmentStatistics>,
    /// DocNo mapping
    docno_map: Arc<DocNoMap>,
}

impl SegmentReader {
    /// Create a segment reader from in-memory data
    pub fn from_memory(
        meta: SegmentMeta,
        terms: TermDictionary,
        postings: PostingsReader,
        docvalues: DocValuesReader,
        stats: SegmentStatistics,
        docno_map: DocNoMap,
    ) -> Self {
        Self {
            meta,
            terms: Arc::new(terms),
            postings: Arc::new(postings),
            docvalues: Arc::new(docvalues),
            stats: Arc::new(stats),
            docno_map: Arc::new(docno_map),
        }
    }

    /// Get segment metadata
    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    /// Get segment ID
    pub fn id(&self) -> SegmentId {
        self.meta.id
    }

    /// Get the term dictionary
    pub fn terms(&self) -> &TermDictionary {
        &self.terms
    }

    /// Get postings metadata for a term
    pub fn get_posting_meta(&self, term: &str) -> Option<&PostingListMeta> {
        self.terms.get(term)
    }

    /// Get a posting iterator for a term
    pub fn get_postings(&self, term: &str) -> io::Result<Option<PostingIterator>> {
        if let Some(meta) = self.terms.get(term) {
            Ok(Some(self.postings.get_postings(meta)?))
        } else {
            Ok(None)
        }
    }

    /// Get document frequency for a term
    pub fn doc_frequency(&self, term: &str) -> u32 {
        self.terms.get(term).map(|m| m.doc_frequency).unwrap_or(0)
    }

    /// Check if a docno is deleted
    pub fn is_deleted(&self, docno: DocNo) -> bool {
        self.docno_map.is_deleted(docno)
    }

    /// Check if a docno is live
    pub fn is_live(&self, docno: DocNo) -> bool {
        self.docno_map.is_live(docno)
    }

    /// Get the external document ID for a docno
    pub fn get_doc_id(&self, docno: DocNo) -> Option<DocumentId> {
        self.docno_map.get_doc_id(docno)
    }

    /// Get the version for a docno
    pub fn get_version(&self, docno: DocNo) -> Option<Version> {
        self.docno_map.get_version(docno)
    }

    /// Get document length for a docno
    pub fn get_doc_length(&self, docno: DocNo) -> Option<u32> {
        self.stats.get_doc_length(docno)
    }

    /// Get segment statistics
    pub fn stats(&self) -> &SegmentStatistics {
        &self.stats
    }

    /// Get docvalues reader
    pub fn docvalues(&self) -> &DocValuesReader {
        &self.docvalues
    }

    /// Get the docno map
    pub fn docno_map(&self) -> &DocNoMap {
        &self.docno_map
    }

    /// Compute BM25 score for a posting
    pub fn bm25_score(
        &self,
        term_frequency: u32,
        doc_frequency: u32,
        total_docs: u32,
        docno: DocNo,
        params: &Bm25Params,
    ) -> f32 {
        if let Some(doc_len) = self.stats.get_doc_length(docno) {
            self.stats.bm25_score(
                term_frequency as f32,
                doc_frequency,
                total_docs,
                doc_len,
                params,
            )
        } else {
            0.0
        }
    }

    /// Get the number of documents (including deleted)
    pub fn doc_count(&self) -> u32 {
        self.meta.doc_count
    }

    /// Get the number of live documents
    pub fn live_doc_count(&self) -> u32 {
        self.meta.live_doc_count
    }

    /// Get the delete ratio
    pub fn delete_ratio(&self) -> f64 {
        self.meta.delete_ratio()
    }

    /// Get the number of unique terms
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// Builder for creating segment readers from serialized data
pub struct SegmentReaderBuilder {
    meta: Option<SegmentMeta>,
    terms_data: Option<(Vec<u8>, Vec<PostingListMeta>)>,
    postings_data: Option<Vec<u8>>,
    docvalues: Option<DocValuesReader>,
    stats: Option<SegmentStatistics>,
    docno_map: Option<DocNoMap>,
}

impl SegmentReaderBuilder {
    pub fn new() -> Self {
        Self {
            meta: None,
            terms_data: None,
            postings_data: None,
            docvalues: None,
            stats: None,
            docno_map: None,
        }
    }

    pub fn with_meta(mut self, meta: SegmentMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    pub fn with_terms(mut self, fst_data: Vec<u8>, metadata: Vec<PostingListMeta>) -> Self {
        self.terms_data = Some((fst_data, metadata));
        self
    }

    pub fn with_postings(mut self, data: Vec<u8>) -> Self {
        self.postings_data = Some(data);
        self
    }

    pub fn with_docvalues(mut self, docvalues: DocValuesReader) -> Self {
        self.docvalues = Some(docvalues);
        self
    }

    pub fn with_stats(mut self, stats: SegmentStatistics) -> Self {
        self.stats = Some(stats);
        self
    }

    pub fn with_docno_map(mut self, docno_map: DocNoMap) -> Self {
        self.docno_map = Some(docno_map);
        self
    }

    pub fn build(self) -> io::Result<SegmentReader> {
        let meta = self
            .meta
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing segment meta"))?;

        let (fst_data, term_meta) = self
            .terms_data
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing terms data"))?;

        let terms = TermDictionary::new(fst_data, term_meta)?;

        let postings_data = self
            .postings_data
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing postings data"))?;

        let postings = PostingsReader::new(postings_data);

        let docvalues = self.docvalues.unwrap_or_default();
        let stats = self.stats.unwrap_or_default();
        let docno_map = self.docno_map.unwrap_or_default();

        Ok(SegmentReader::from_memory(
            meta, terms, postings, docvalues, stats, docno_map,
        ))
    }
}

impl Default for SegmentReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{
        MutableBuffer, PostingsWriter, TermDictionaryBuilder, Version,
    };
    use std::collections::HashMap;

    fn create_test_segment() -> SegmentReader {
        // Create postings
        let mut postings_writer = PostingsWriter::new();
        let mut term_builder = TermDictionaryBuilder::new();

        // Add postings for "hello"
        postings_writer.start_posting_list();
        postings_writer.add_posting(super::super::types::Posting::new(DocNo(0), 2));
        postings_writer.add_posting(super::super::types::Posting::new(DocNo(2), 1));
        let hello_meta = postings_writer.finish_posting_list(2, 3);
        term_builder.add("hello".to_string(), hello_meta);

        // Add postings for "world"
        postings_writer.start_posting_list();
        postings_writer.add_posting(super::super::types::Posting::new(DocNo(1), 3));
        postings_writer.add_posting(super::super::types::Posting::new(DocNo(2), 2));
        let world_meta = postings_writer.finish_posting_list(2, 5);
        term_builder.add("world".to_string(), world_meta);

        let term_dict = term_builder.build().unwrap();

        // Create docno map
        let mut docno_map = DocNoMap::new();
        docno_map.add(100, Version::new(1));
        docno_map.add(200, Version::new(1));
        docno_map.add(300, Version::new(1));

        // Create stats
        let mut stats = SegmentStatistics::new();
        stats.add_document(50); // doc 0
        stats.add_document(75); // doc 1
        stats.add_document(100); // doc 2

        let meta = SegmentMeta {
            id: SegmentId::new(1),
            min_raft_index: 1,
            max_raft_index: 3,
            doc_count: 3,
            live_doc_count: 3,
            size_bytes: 1000,
            created_at: 0,
        };

        SegmentReader::from_memory(
            meta,
            term_dict,
            PostingsReader::new(postings_writer.into_data()),
            DocValuesReader::new(),
            stats,
            docno_map,
        )
    }

    #[test]
    fn test_segment_reader_basic() {
        let reader = create_test_segment();

        assert_eq!(reader.id(), SegmentId::new(1));
        assert_eq!(reader.doc_count(), 3);
        assert_eq!(reader.live_doc_count(), 3);
        assert_eq!(reader.term_count(), 2);

        // Check term lookups
        assert!(reader.get_posting_meta("hello").is_some());
        assert!(reader.get_posting_meta("world").is_some());
        assert!(reader.get_posting_meta("foo").is_none());

        // Check doc frequency
        assert_eq!(reader.doc_frequency("hello"), 2);
        assert_eq!(reader.doc_frequency("world"), 2);
        assert_eq!(reader.doc_frequency("foo"), 0);
    }

    #[test]
    fn test_segment_reader_docno_map() {
        let reader = create_test_segment();

        assert_eq!(reader.get_doc_id(DocNo(0)), Some(100));
        assert_eq!(reader.get_doc_id(DocNo(1)), Some(200));
        assert_eq!(reader.get_doc_id(DocNo(2)), Some(300));

        assert!(reader.is_live(DocNo(0)));
        assert!(!reader.is_deleted(DocNo(0)));
    }

    #[test]
    fn test_segment_reader_bm25() {
        let reader = create_test_segment();
        let params = Bm25Params::default();

        // Score for term with TF=2 in doc 0
        let score = reader.bm25_score(2, 2, 3, DocNo(0), &params);
        assert!(score > 0.0);

        // Score with higher TF should be higher
        let score_low = reader.bm25_score(1, 2, 3, DocNo(0), &params);
        let score_high = reader.bm25_score(3, 2, 3, DocNo(0), &params);
        assert!(score_high > score_low);
    }

    #[test]
    fn test_segment_reader_postings() {
        let reader = create_test_segment();

        let iter = reader.get_postings("hello").unwrap().unwrap();
        let postings: Vec<_> = iter.collect();

        assert_eq!(postings.len(), 2);
        assert_eq!(postings[0].0, DocNo(0));
        assert_eq!(postings[0].1, 2); // TF
        assert_eq!(postings[1].0, DocNo(2));
        assert_eq!(postings[1].1, 1); // TF
    }
}
