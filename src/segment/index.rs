//! Segment index combining mutable buffer + immutable segments
//!
//! SegmentIndex = mutable buffer + immutable segments
//! Provides BM25+ scoring across buffer + segments

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io;
use std::sync::{Arc, RwLock};

use super::buffer::{BufferConfig, MutableBuffer};
use super::manifest::{ManifestHolder, SegmentManifest};
use super::reader::{SegmentMeta, SegmentReader};
use super::statistics::{Bm25Params, IndexStatistics, SegmentStatistics};
use super::store::SegmentStore;
use super::types::{DocNo, DocumentId, RaftIndex, SegmentId, Version};
use super::writer::SegmentWriter;
use arc_swap::ArcSwap;

/// Search result from the segment index
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// External document ID
    pub doc_id: DocumentId,
    /// BM25+ score
    pub score: f32,
    /// Source segment (None if from buffer)
    pub segment_id: Option<SegmentId>,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior (we want top-k highest scores)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Configuration for the segment index
#[derive(Clone, Debug)]
pub struct SegmentIndexConfig {
    /// Buffer configuration
    pub buffer: BufferConfig,
    /// BM25 parameters
    pub bm25: Bm25Params,
    /// Maximum number of segments before forcing merge
    pub max_segments: usize,
}

impl Default for SegmentIndexConfig {
    fn default() -> Self {
        Self {
            buffer: BufferConfig::default(),
            bm25: Bm25Params::default(),
            max_segments: 10,
        }
    }
}

/// The main segment-based index
///
/// Combines a mutable buffer for recent writes with immutable segments
/// for efficient BM25+ search.
pub struct SegmentIndex {
    /// Mutable buffer for recent writes
    buffer: RwLock<MutableBuffer>,
    /// Immutable segment readers
    segments: ArcSwap<Vec<Arc<SegmentReader>>>,
    /// Segment manifest
    manifest: ManifestHolder,
    /// Index configuration
    config: SegmentIndexConfig,
    /// Optional on-disk store
    store: Option<SegmentStore>,
}

impl SegmentIndex {
    /// Create a new segment index
    pub fn new(config: SegmentIndexConfig) -> Self {
        Self {
            buffer: RwLock::new(MutableBuffer::new()),
            segments: ArcSwap::from_pointee(Vec::new()),
            manifest: ManifestHolder::default(),
            config,
            store: None,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(SegmentIndexConfig::default())
    }

    /// Open with a persistent store directory
    pub fn open_with_store(
        config: SegmentIndexConfig,
        dir: std::path::PathBuf,
    ) -> io::Result<Self> {
        let store = SegmentStore::new(dir)?;
        let manifest = store
            .load_manifest()
            .unwrap_or_else(|_| SegmentManifest::new());
        let mut index = Self {
            buffer: RwLock::new(MutableBuffer::new()),
            segments: ArcSwap::from_pointee(Vec::new()),
            manifest: ManifestHolder::new(manifest),
            config,
            store: Some(store),
        };
        // Load segments
        if let Some(store) = &index.store {
            let manifest_snapshot = index.manifest.snapshot();
            let mut segs = Vec::new();
            for entry in manifest_snapshot.segments.iter() {
                if let Ok(reader) = store.read_segment(entry.meta.clone()) {
                    segs.push(reader);
                }
            }
            index.segments.store(Arc::new(segs));
        }
        Ok(index)
    }

    /// Index a document
    pub fn index_document(
        &self,
        doc_id: DocumentId,
        version: Version,
        term_frequencies: HashMap<String, u32>,
        doc_len: u32,
        raft_index: RaftIndex,
    ) -> io::Result<DocNo> {
        let mut buffer = self.buffer.write().unwrap();
        let docno =
            buffer.index_document(doc_id, version, term_frequencies, doc_len, None, raft_index);

        // Check if we need to flush
        if buffer.should_flush(&self.config.buffer) {
            drop(buffer);
            self.flush()?;
        }

        Ok(docno)
    }

    /// Index a document with term positions (for phrase queries)
    pub fn index_document_with_positions(
        &self,
        doc_id: DocumentId,
        version: Version,
        term_positions: HashMap<String, Vec<u32>>,
        doc_len: u32,
        raft_index: RaftIndex,
    ) -> io::Result<DocNo> {
        let mut buffer = self.buffer.write().unwrap();
        let docno = buffer.index_document_with_positions(
            doc_id,
            version,
            term_positions,
            doc_len,
            None,
            raft_index,
        );

        // Check if we need to flush
        if buffer.should_flush(&self.config.buffer) {
            drop(buffer);
            self.flush()?;
        }

        Ok(docno)
    }

    /// Delete a document
    pub fn delete_document(&self, doc_id: DocumentId, raft_index: RaftIndex) -> bool {
        let mut buffer = self.buffer.write().unwrap();
        buffer.delete_document(doc_id, raft_index)
        // Note: for documents in segments, we'd need to update the segment's delete bitset
    }

    /// Flush the buffer to a new segment
    pub fn flush(&self) -> io::Result<Option<Arc<SegmentReader>>> {
        let mut buffer = self.buffer.write().unwrap();

        if buffer.doc_count() == 0 {
            return Ok(None);
        }

        // Allocate segment ID
        let segment_id = self.manifest.snapshot().next_segment_id;
        self.manifest.update(|m| {
            m.allocate_segment_id();
        });

        // Write segment
        let writer = SegmentWriter::new(segment_id);
        let result = writer.write_from_buffer(&buffer)?;
        let checksum = result.checksum();

        // Persist segment if store is configured
        if let Some(store) = &self.store {
            store.write_segment(&result)?;
        }

        // Add to segments
        let reader = Arc::new(result.reader);
        {
            let mut segments = self.segments.load().as_ref().clone();
            segments.push(reader.clone());
            self.segments.store(Arc::new(segments));
        }

        // Update manifest
        self.manifest.update(|m| {
            m.add_segment_with_checksum(reader.meta().clone(), checksum);
            if let Some(max_raft) = buffer.raft_index_range().1 {
                m.update_index_applied(max_raft);
            }
        });

        // Persist manifest if store configured
        if let Some(store) = &self.store {
            let snapshot = self.manifest.snapshot();
            let _ = store.save_manifest(&snapshot);
        }

        // Clear buffer
        buffer.clear();

        Ok(Some(reader))
    }

    /// Perform BM25+ keyword search
    pub fn keyword_search(
        &self,
        query_terms: &[String],
        top_k: usize,
    ) -> io::Result<Vec<SearchResult>> {
        self.keyword_search_with_tombstones(query_terms, top_k, |_| false)
    }

    /// Perform BM25+ keyword search with a tombstone predicate
    pub fn keyword_search_with_tombstones<F>(
        &self,
        query_terms: &[String],
        top_k: usize,
        is_tombstoned: F,
    ) -> io::Result<Vec<SearchResult>>
    where
        F: Fn(DocumentId) -> bool,
    {
        if query_terms.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Calculate global statistics
        let (total_docs, global_avgdl) = self.compute_global_stats();

        if total_docs == 0 {
            return Ok(Vec::new());
        }

        // Collect term document frequencies across all sources
        let mut term_dfs: HashMap<String, u32> = HashMap::new();

        // From buffer
        {
            let buffer = self.buffer.read().unwrap();
            for term in query_terms {
                let df = buffer.doc_frequency(term);
                *term_dfs.entry(term.clone()).or_insert(0) += df;
            }
        }

        // From segments
        {
            let segments = self.segments.load();
            for segment in segments.iter() {
                for term in query_terms {
                    let df = segment.doc_frequency(term);
                    *term_dfs.entry(term.clone()).or_insert(0) += df;
                }
            }
        }

        // Score documents using a min-heap for top-k
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::new();

        // Score buffer documents
        {
            let buffer = self.buffer.read().unwrap();
            let buffer_scores = buffer.search(query_terms, &self.config.bm25, total_docs);

            for (docno, score) in buffer_scores {
                if let Some(entry) = buffer.get_doc_entry(docno) {
                    if is_tombstoned(entry.doc_id) {
                        continue;
                    }
                    let result = SearchResult {
                        doc_id: entry.doc_id,
                        score,
                        segment_id: None,
                    };
                    self.add_to_heap(&mut heap, result, top_k);
                }
            }
        }

        // Score segment documents
        {
            let segments = self.segments.load();
            for segment in segments.iter() {
                let segment_scores =
                    self.score_segment(segment, query_terms, &term_dfs, total_docs)?;

                for (docno, score) in segment_scores {
                    if let Some(doc_id) = segment.get_doc_id(docno) {
                        if is_tombstoned(doc_id) {
                            continue;
                        }
                        let result = SearchResult {
                            doc_id,
                            score,
                            segment_id: Some(segment.id()),
                        };
                        self.add_to_heap(&mut heap, result, top_k);
                    }
                }
            }
        }

        // Convert heap to sorted results (already ordered highest score first due to reversed Ord)
        let results: Vec<_> = heap.into_sorted_vec();
        Ok(results)
    }

    /// Score documents in a segment
    fn score_segment(
        &self,
        segment: &SegmentReader,
        query_terms: &[String],
        term_dfs: &HashMap<String, u32>,
        total_docs: u32,
    ) -> io::Result<Vec<(DocNo, f32)>> {
        let mut scores: HashMap<DocNo, f32> = HashMap::new();

        for term in query_terms {
            let df = term_dfs.get(term).copied().unwrap_or(0);
            if df == 0 {
                continue;
            }

            if let Some(mut iter) = segment.get_postings(term)? {
                while let Some((docno, tf)) = iter.next() {
                    if segment.is_deleted(docno) {
                        continue;
                    }

                    let score = segment.bm25_score(tf, df, total_docs, docno, &self.config.bm25);
                    *scores.entry(docno).or_insert(0.0) += score;
                }
            }
        }

        Ok(scores.into_iter().collect())
    }

    /// Add result to top-k heap
    fn add_to_heap(&self, heap: &mut BinaryHeap<SearchResult>, result: SearchResult, top_k: usize) {
        if heap.len() < top_k {
            heap.push(result);
        } else if let Some(min) = heap.peek() {
            // heap is min-heap due to reversed Ord
            if result.score > min.score {
                heap.pop();
                heap.push(result);
            }
        }
    }

    /// Compute global statistics across buffer and segments
    pub fn compute_global_stats(&self) -> (u32, f64) {
        let mut total_docs = 0u32;
        let mut total_length = 0u64;

        // Buffer stats
        {
            let buffer = self.buffer.read().unwrap();
            total_docs += buffer.live_doc_count();
            total_length += buffer.stats().total_doc_length;
        }

        // Segment stats
        {
            let segments = self.segments.load();
            for segment in segments.iter() {
                total_docs += segment.live_doc_count();
                total_length += segment.stats().total_doc_length;
            }
        }

        let avgdl = if total_docs > 0 {
            total_length as f64 / total_docs as f64
        } else {
            0.0
        };

        (total_docs, avgdl)
    }

    /// Get total document count (including deleted)
    pub fn total_doc_count(&self) -> u32 {
        let buffer_count = self.buffer.read().unwrap().doc_count();
        let segment_count: u32 = self.segments.load().iter().map(|s| s.doc_count()).sum();
        buffer_count + segment_count
    }

    /// Get live document count (excluding deleted)
    pub fn live_doc_count(&self) -> u32 {
        let buffer_count = self.buffer.read().unwrap().live_doc_count();
        let segment_count: u32 = self
            .segments
            .load()
            .iter()
            .map(|s| s.live_doc_count())
            .sum();
        buffer_count + segment_count
    }

    /// Get segment count
    pub fn segment_count(&self) -> usize {
        self.segments.load().len()
    }

    /// Get buffer document count
    pub fn buffer_doc_count(&self) -> u32 {
        self.buffer.read().unwrap().doc_count()
    }

    /// Get the manifest
    pub fn manifest(&self) -> &ManifestHolder {
        &self.manifest
    }

    /// Get access to the buffer
    pub fn buffer(&self) -> &RwLock<MutableBuffer> {
        &self.buffer
    }

    /// Get access to the segments
    pub fn segments(&self) -> Arc<Vec<Arc<SegmentReader>>> {
        self.segments.load_full()
    }

    /// Check if document exists
    pub fn contains_document(&self, doc_id: DocumentId) -> bool {
        // Check buffer
        if self.buffer.read().unwrap().contains_document(doc_id) {
            return true;
        }

        // Check segments
        let segments = self.segments.load();
        for segment in segments.iter() {
            // Search through the docno map for this document
            for (docno, entry) in segment.docno_map().live_docs() {
                if entry.doc_id == doc_id {
                    return true;
                }
            }
        }

        false
    }

    /// Get segments that could be merged
    pub fn get_merge_candidates(&self) -> Vec<Arc<SegmentReader>> {
        let segments = self.segments.load();

        // Simple policy: merge all segments if we have too many
        if segments.len() > self.config.max_segments {
            segments.as_ref().clone()
        } else {
            Vec::new()
        }
    }

    /// Apply a merge result.
    pub fn apply_merge(
        &self,
        merged_result: super::writer::SegmentWriteResult,
        merged_ids: &[SegmentId],
    ) -> io::Result<()> {
        let checksum = merged_result.checksum();

        if let Some(store) = &self.store {
            store.write_segment(&merged_result)?;
        }

        let merged_segment = Arc::new(merged_result.reader);
        let mut segments = self.segments.load().as_ref().clone();

        // Remove old segments
        segments.retain(|s| !merged_ids.contains(&s.id()));

        // Add new segment
        segments.push(merged_segment.clone());
        self.segments.store(Arc::new(segments));

        // Update manifest
        self.manifest.update(|m| {
            for id in merged_ids {
                m.remove_segment(*id);
            }
            m.add_segment_with_checksum(merged_segment.meta().clone(), checksum);
        });

        if let Some(store) = &self.store {
            let snapshot = self.manifest.snapshot();
            let _ = store.save_manifest(&snapshot);
        }

        Ok(())
    }
}

impl Default for SegmentIndex {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_and_search() {
        let index = SegmentIndex::default_config();

        // Index documents
        let mut tf1 = HashMap::new();
        tf1.insert("rust".to_string(), 5);
        tf1.insert("programming".to_string(), 3);
        index
            .index_document(1, Version::new(1), tf1, 100, 1)
            .unwrap();

        let mut tf2 = HashMap::new();
        tf2.insert("rust".to_string(), 2);
        tf2.insert("language".to_string(), 4);
        index
            .index_document(2, Version::new(1), tf2, 150, 2)
            .unwrap();

        let mut tf3 = HashMap::new();
        tf3.insert("programming".to_string(), 1);
        tf3.insert("language".to_string(), 2);
        index
            .index_document(3, Version::new(1), tf3, 80, 3)
            .unwrap();

        // Search
        let results = index.keyword_search(&["rust".to_string()], 10).unwrap();

        assert_eq!(results.len(), 2);
        // Doc 1 should rank higher (higher TF for "rust")
        assert_eq!(results[0].doc_id, 1);
        assert_eq!(results[1].doc_id, 2);
    }

    #[test]
    fn test_multi_term_search() {
        let index = SegmentIndex::default_config();

        let mut tf1 = HashMap::new();
        tf1.insert("rust".to_string(), 3);
        tf1.insert("programming".to_string(), 2);
        index
            .index_document(1, Version::new(1), tf1, 100, 1)
            .unwrap();

        let mut tf2 = HashMap::new();
        tf2.insert("rust".to_string(), 1);
        tf2.insert("programming".to_string(), 5);
        index
            .index_document(2, Version::new(1), tf2, 100, 2)
            .unwrap();

        let results = index
            .keyword_search(&["rust".to_string(), "programming".to_string()], 10)
            .unwrap();

        assert_eq!(results.len(), 2);
        // Both documents match both terms
        assert!(results.iter().any(|r| r.doc_id == 1));
        assert!(results.iter().any(|r| r.doc_id == 2));
    }

    #[test]
    fn test_flush_to_segment() {
        let config = SegmentIndexConfig {
            buffer: BufferConfig {
                max_docs: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let index = SegmentIndex::new(config);

        // Index documents (should trigger flush after 2)
        let mut tf1 = HashMap::new();
        tf1.insert("test".to_string(), 1);
        index
            .index_document(1, Version::new(1), tf1, 50, 1)
            .unwrap();

        assert_eq!(index.segment_count(), 0);
        assert_eq!(index.buffer_doc_count(), 1);

        let mut tf2 = HashMap::new();
        tf2.insert("test".to_string(), 2);
        index
            .index_document(2, Version::new(1), tf2, 75, 2)
            .unwrap();

        // Should have flushed
        assert_eq!(index.segment_count(), 1);
        assert_eq!(index.buffer_doc_count(), 0);

        // Search should still work across segment
        let results = index.keyword_search(&["test".to_string()], 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete_document() {
        let index = SegmentIndex::default_config();

        let mut tf1 = HashMap::new();
        tf1.insert("hello".to_string(), 1);
        index
            .index_document(1, Version::new(1), tf1, 50, 1)
            .unwrap();

        let mut tf2 = HashMap::new();
        tf2.insert("hello".to_string(), 1);
        index
            .index_document(2, Version::new(1), tf2, 50, 2)
            .unwrap();

        // Both should be searchable
        let results = index.keyword_search(&["hello".to_string()], 10).unwrap();
        assert_eq!(results.len(), 2);

        // Delete one
        index.delete_document(1, 3);

        // Only one should be searchable now
        let results = index.keyword_search(&["hello".to_string()], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
    }

    #[test]
    fn test_keyword_search_with_tombstones() {
        let index = SegmentIndex::default_config();

        let mut tf1 = HashMap::new();
        tf1.insert("hello".to_string(), 1);
        index
            .index_document(1, Version::new(1), tf1, 50, 1)
            .unwrap();

        let mut tf2 = HashMap::new();
        tf2.insert("hello".to_string(), 1);
        index
            .index_document(2, Version::new(1), tf2, 50, 2)
            .unwrap();

        let results = index
            .keyword_search_with_tombstones(&["hello".to_string()], 10, |doc_id| doc_id == 1)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
    }

    #[test]
    fn test_document_count() {
        let index = SegmentIndex::default_config();

        assert_eq!(index.total_doc_count(), 0);
        assert_eq!(index.live_doc_count(), 0);

        let mut tf1 = HashMap::new();
        tf1.insert("test".to_string(), 1);
        index
            .index_document(1, Version::new(1), tf1, 50, 1)
            .unwrap();

        assert_eq!(index.total_doc_count(), 1);
        assert_eq!(index.live_doc_count(), 1);

        index.delete_document(1, 2);

        assert_eq!(index.total_doc_count(), 1);
        assert_eq!(index.live_doc_count(), 0);
    }
}
