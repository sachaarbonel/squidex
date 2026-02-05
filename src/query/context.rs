//! Query execution context
//!
//! The `QueryContext` provides access to index data and caching during query execution.

use crate::query::accessor::{IndexAccessor, PostingEntry, TermStats};
use crate::segment::{DocNo, DocumentId};
use crate::tokenizer::Tokenizer;
use crate::Result;
use parking_lot::RwLock;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// Filter cache for reusing expensive filter computations
pub type FilterCache = Arc<RwLock<HashMap<String, RoaringBitmap>>>;

/// Query execution context providing access to index data
///
/// This struct is passed to query nodes during execution, giving them
/// access to the index, document store, and various caches.
pub struct QueryContext {
    /// Total number of documents in the index
    total_docs: usize,

    /// Average document length (in tokens)
    avg_doc_length: f32,

    /// Document ID to DocNo mapping (external ID -> dense segment ID)
    doc_id_to_docno: Arc<HashMap<DocumentId, DocNo>>,

    /// DocNo to Document ID mapping (dense segment ID -> external ID)
    docno_to_doc_id: Arc<Vec<DocumentId>>,

    /// Document lengths (number of tokens per document)
    doc_lengths: Arc<Vec<u32>>,

    /// Term document frequencies cache
    term_doc_frequencies: Arc<RwLock<HashMap<String, u32>>>,

    /// Filter result cache (keyed by canonical filter representation)
    filter_cache: FilterCache,

    /// Tokenizer for text processing
    tokenizer: Arc<Tokenizer>,

    /// Tombstone bitmap (deleted documents)
    tombstones: Arc<RoaringBitmap>,

    /// Per-field term frequencies for BM25
    /// Key: "field:term", Value: (doc_frequency, total_term_frequency)
    field_term_stats: Arc<RwLock<HashMap<String, (u32, u64)>>>,

    /// Index accessor for real posting list lookups
    accessor: Option<Arc<dyn IndexAccessor>>,
}

impl QueryContext {
    /// Create a new query context
    pub fn new(
        total_docs: usize,
        avg_doc_length: f32,
        tokenizer: Arc<Tokenizer>,
    ) -> Self {
        Self {
            total_docs,
            avg_doc_length,
            doc_id_to_docno: Arc::new(HashMap::new()),
            docno_to_doc_id: Arc::new(Vec::new()),
            doc_lengths: Arc::new(Vec::new()),
            term_doc_frequencies: Arc::new(RwLock::new(HashMap::new())),
            filter_cache: Arc::new(RwLock::new(HashMap::new())),
            tokenizer,
            tombstones: Arc::new(RoaringBitmap::new()),
            field_term_stats: Arc::new(RwLock::new(HashMap::new())),
            accessor: None,
        }
    }

    /// Create a context builder
    pub fn builder() -> QueryContextBuilder {
        QueryContextBuilder::default()
    }

    /// Get total number of documents
    pub fn total_docs(&self) -> usize {
        self.total_docs
    }

    /// Get average document length
    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    /// Get document length for a specific document
    pub fn doc_length(&self, docno: DocNo) -> u32 {
        self.doc_lengths
            .get(docno.as_usize())
            .copied()
            .unwrap_or(0)
    }

    /// Get document frequency for a term (number of docs containing the term)
    pub fn doc_frequency(&self, term: &str) -> u32 {
        self.term_doc_frequencies
            .read()
            .get(term)
            .copied()
            .unwrap_or(0)
    }

    /// Get field-specific term statistics
    pub fn field_term_stats(&self, field: &str, term: &str) -> Option<(u32, u64)> {
        let key = format!("{}:{}", field, term);
        self.field_term_stats.read().get(&key).copied()
    }

    /// Set term document frequency
    pub fn set_doc_frequency(&self, term: &str, freq: u32) {
        self.term_doc_frequencies
            .write()
            .insert(term.to_string(), freq);
    }

    /// Set field-specific term statistics
    pub fn set_field_term_stats(&self, field: &str, term: &str, doc_freq: u32, total_freq: u64) {
        let key = format!("{}:{}", field, term);
        self.field_term_stats
            .write()
            .insert(key, (doc_freq, total_freq));
    }

    /// Get or compute a cached filter result
    pub fn get_or_cache_filter<F>(
        &self,
        cache_key: &str,
        compute: F,
    ) -> Result<RoaringBitmap>
    where
        F: FnOnce() -> Result<RoaringBitmap>,
    {
        // Check cache first
        if let Some(cached) = self.filter_cache.read().get(cache_key) {
            return Ok(cached.clone());
        }

        // Compute and cache
        let result = compute()?;
        self.filter_cache
            .write()
            .insert(cache_key.to_string(), result.clone());
        Ok(result)
    }

    /// Clear the filter cache
    pub fn clear_filter_cache(&self) {
        self.filter_cache.write().clear();
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Check if a document is tombstoned (deleted)
    pub fn is_tombstoned(&self, docno: DocNo) -> bool {
        self.tombstones.contains(docno.as_u32())
    }

    /// Get the tombstone bitmap
    pub fn tombstones(&self) -> &RoaringBitmap {
        &self.tombstones
    }

    /// Map external document ID to internal DocNo
    pub fn doc_id_to_docno(&self, doc_id: DocumentId) -> Option<DocNo> {
        self.doc_id_to_docno.get(&doc_id).copied()
    }

    /// Map internal DocNo to external document ID
    pub fn docno_to_doc_id(&self, docno: DocNo) -> Option<DocumentId> {
        // First try the direct mapping
        if let Some(&doc_id) = self.docno_to_doc_id.get(docno.as_usize()) {
            return Some(doc_id);
        }
        // Fall back to accessor if available
        if let Some(accessor) = &self.accessor {
            return accessor.docno_to_doc_id(docno);
        }
        None
    }

    /// Calculate BM25 IDF for a term
    ///
    /// IDF = log(1 + (N - n + 0.5) / (n + 0.5))
    /// where N = total docs, n = doc frequency
    pub fn bm25_idf(&self, doc_frequency: u32) -> f32 {
        let n = self.total_docs as f32;
        let df = doc_frequency as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Calculate BM25 score for a term in a document
    ///
    /// Score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (dl / avgdl)))
    pub fn bm25_score(
        &self,
        term_frequency: u32,
        doc_frequency: u32,
        doc_length: u32,
    ) -> f32 {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let idf = self.bm25_idf(doc_frequency);
        let tf = term_frequency as f32;
        let dl = doc_length as f32;
        let avgdl = self.avg_doc_length;

        let norm = 1.0 - B + B * (dl / avgdl);
        idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
    }

    /// Get the index accessor
    pub fn accessor(&self) -> Option<&dyn IndexAccessor> {
        self.accessor.as_ref().map(|a| a.as_ref())
    }

    /// Get postings for a term using the accessor
    pub fn get_postings(&self, term: &str) -> Vec<PostingEntry> {
        self.accessor
            .as_ref()
            .map(|a| a.postings(term))
            .unwrap_or_default()
    }

    /// Get postings as a bitmap for a term using the accessor
    pub fn get_postings_bitmap(&self, term: &str) -> RoaringBitmap {
        self.accessor
            .as_ref()
            .map(|a| a.postings_bitmap(term))
            .unwrap_or_default()
    }

    /// Get term statistics using the accessor
    pub fn get_term_stats(&self, term: &str) -> TermStats {
        self.accessor
            .as_ref()
            .map(|a| a.term_stats(term))
            .unwrap_or_default()
    }
}

/// Builder for QueryContext
#[derive(Default)]
pub struct QueryContextBuilder {
    total_docs: usize,
    avg_doc_length: f32,
    doc_id_to_docno: HashMap<DocumentId, DocNo>,
    docno_to_doc_id: Vec<DocumentId>,
    doc_lengths: Vec<u32>,
    tokenizer: Option<Arc<Tokenizer>>,
    tombstones: RoaringBitmap,
    accessor: Option<Arc<dyn IndexAccessor>>,
}

impl QueryContextBuilder {
    /// Set total number of documents
    pub fn total_docs(mut self, total: usize) -> Self {
        self.total_docs = total;
        self
    }

    /// Set average document length
    pub fn avg_doc_length(mut self, avg: f32) -> Self {
        self.avg_doc_length = avg;
        self
    }

    /// Set document ID to DocNo mapping
    pub fn doc_id_mapping(mut self, mapping: HashMap<DocumentId, DocNo>) -> Self {
        self.doc_id_to_docno = mapping;
        self
    }

    /// Set DocNo to document ID mapping
    pub fn docno_mapping(mut self, mapping: Vec<DocumentId>) -> Self {
        self.docno_to_doc_id = mapping;
        self
    }

    /// Set document lengths
    pub fn doc_lengths(mut self, lengths: Vec<u32>) -> Self {
        self.doc_lengths = lengths;
        self
    }

    /// Set the tokenizer
    pub fn tokenizer(mut self, tokenizer: Arc<Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set the tombstone bitmap
    pub fn tombstones(mut self, tombstones: RoaringBitmap) -> Self {
        self.tombstones = tombstones;
        self
    }

    /// Set the index accessor
    pub fn accessor(mut self, accessor: Arc<dyn IndexAccessor>) -> Self {
        self.accessor = Some(accessor);
        self
    }

    /// Build the QueryContext
    pub fn build(self) -> QueryContext {
        let tokenizer = self
            .tokenizer
            .unwrap_or_else(|| Arc::new(Tokenizer::new(&crate::TokenizerConfig::default())));

        QueryContext {
            total_docs: self.total_docs,
            avg_doc_length: self.avg_doc_length,
            doc_id_to_docno: Arc::new(self.doc_id_to_docno),
            docno_to_doc_id: Arc::new(self.docno_to_doc_id),
            doc_lengths: Arc::new(self.doc_lengths),
            term_doc_frequencies: Arc::new(RwLock::new(HashMap::new())),
            filter_cache: Arc::new(RwLock::new(HashMap::new())),
            tokenizer,
            tombstones: Arc::new(self.tombstones),
            field_term_stats: Arc::new(RwLock::new(HashMap::new())),
            accessor: self.accessor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TokenizerConfig;

    fn create_test_context() -> QueryContext {
        let tokenizer = Arc::new(Tokenizer::new(&TokenizerConfig::default()));
        QueryContext::builder()
            .total_docs(1000)
            .avg_doc_length(100.0)
            .tokenizer(tokenizer)
            .build()
    }

    #[test]
    fn test_context_creation() {
        let ctx = create_test_context();
        assert_eq!(ctx.total_docs(), 1000);
        assert_eq!(ctx.avg_doc_length(), 100.0);
    }

    #[test]
    fn test_doc_frequency() {
        let ctx = create_test_context();
        assert_eq!(ctx.doc_frequency("unknown"), 0);

        ctx.set_doc_frequency("test", 50);
        assert_eq!(ctx.doc_frequency("test"), 50);
    }

    #[test]
    fn test_bm25_idf() {
        let ctx = create_test_context();

        // Very rare term (df=1) should have high IDF
        let idf_rare = ctx.bm25_idf(1);

        // Very common term (df=999) should have low IDF
        let idf_common = ctx.bm25_idf(999);

        assert!(idf_rare > idf_common);
    }

    #[test]
    fn test_filter_cache() {
        let ctx = create_test_context();

        // First call should compute
        let result1 = ctx
            .get_or_cache_filter("test_filter", || {
                let mut bitmap = RoaringBitmap::new();
                bitmap.insert(1);
                bitmap.insert(2);
                Ok(bitmap)
            })
            .unwrap();

        assert_eq!(result1.len(), 2);

        // Second call should use cache (this closure should not be called)
        let result2 = ctx
            .get_or_cache_filter("test_filter", || {
                panic!("This should not be called - cache should be used");
            })
            .unwrap();

        assert_eq!(result2.len(), 2);
    }
}
