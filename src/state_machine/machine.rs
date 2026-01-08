use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::config::IndexSettings;
use crate::error::{Result, SquidexError};
use crate::models::*;
use crate::state_machine::scoring::*;
use crate::state_machine::snapshot::{SearchSnapshot, SNAPSHOT_VERSION};
use crate::tokenizer::Tokenizer;
use crate::vector::HnswIndex;

/// Production-ready distributed search state machine
pub struct SearchStateMachine {
    // Document storage
    documents: RwLock<HashMap<DocumentId, Document>>,

    // Inverted index for keyword search
    inverted_index: RwLock<HashMap<String, PostingList>>,

    // Vector storage for similarity search (HNSW + Product Quantization)
    hnsw_index: RwLock<HnswIndex>,

    // Metadata indices for filtering
    tag_index: RwLock<HashMap<String, HashSet<DocumentId>>>,
    source_index: RwLock<HashMap<String, HashSet<DocumentId>>>,
    date_index: RwLock<BTreeMap<u64, HashSet<DocumentId>>>,

    // Counters and state
    next_doc_id: AtomicU64,
    total_documents: AtomicU64,
    index_version: AtomicU64,

    // Configuration
    settings: RwLock<IndexSettings>,

    // Tokenizer instance
    tokenizer: Tokenizer,

    // Flag to track if PQ training is needed
    pq_training_pending: AtomicBool,
}

impl SearchStateMachine {
    pub fn new(settings: IndexSettings) -> Self {
        let tokenizer = Tokenizer::new(&settings.tokenizer_config);

        // Create the HNSW index with PQ configuration
        let hnsw_index = HnswIndex::new(
            settings.vector_dimensions,
            settings.pq_config.num_subspaces,
            Default::default(),
        );

        Self {
            documents: RwLock::new(HashMap::new()),
            inverted_index: RwLock::new(HashMap::new()),
            hnsw_index: RwLock::new(hnsw_index),
            tag_index: RwLock::new(HashMap::new()),
            source_index: RwLock::new(HashMap::new()),
            date_index: RwLock::new(BTreeMap::new()),
            next_doc_id: AtomicU64::new(1),
            total_documents: AtomicU64::new(0),
            index_version: AtomicU64::new(0),
            settings: RwLock::new(settings),
            tokenizer,
            pq_training_pending: AtomicBool::new(false),
        }
    }

    /// Index a single document
    pub fn index_document(&self, doc: Document) -> Result<DocumentId> {
        let doc_id = doc.id;

        // Validate embedding dimensions
        let settings = self.settings.read();
        if doc.embedding.len() != settings.vector_dimensions {
            return Err(SquidexError::InvalidEmbeddingDimensions {
                expected: settings.vector_dimensions,
                actual: doc.embedding.len(),
            });
        }
        let min_training_vectors = settings.pq_config.min_training_vectors;
        let pq_enabled = settings.pq_config.enabled;
        drop(settings);

        // Tokenize content for inverted index
        let term_frequencies = self.tokenizer.compute_term_frequencies(&doc.content);

        // Update inverted index
        {
            let mut idx = self.inverted_index.write();
            for (term, freq) in &term_frequencies {
                let posting = idx
                    .entry(term.clone())
                    .or_insert_with(|| PostingList::new(term.clone()));
                posting.add_document(doc_id, *freq);
            }
        }

        // Update HNSW index (with auto-training check)
        {
            let mut index = self.hnsw_index.write();
            index.insert(doc_id, &doc.embedding)?;

            // Check if we should trigger auto-training
            if pq_enabled && index.should_auto_train(min_training_vectors) {
                self.pq_training_pending.store(true, Ordering::SeqCst);
            }
        }

        // Update metadata indices
        self.update_metadata_indices(&doc);

        // Store document
        self.documents.write().insert(doc_id, doc);
        self.total_documents.fetch_add(1, Ordering::SeqCst);
        self.index_version.fetch_add(1, Ordering::SeqCst);

        // Trigger auto-training if pending (do this outside the lock)
        if self.pq_training_pending.swap(false, Ordering::SeqCst) {
            self.try_auto_train();
        }

        Ok(doc_id)
    }

    /// Attempt to auto-train the PQ codebooks if conditions are met
    fn try_auto_train(&self) {
        let mut index = self.hnsw_index.write();
        if let Err(e) = index.auto_train() {
            // Log the error but don't fail the operation
            tracing::warn!("PQ auto-training failed: {}", e);
        }
    }

    /// Delete a document
    pub fn delete_document(&self, doc_id: DocumentId) -> Result<()> {
        // Remove from document store
        let doc = self
            .documents
            .write()
            .remove(&doc_id)
            .ok_or(SquidexError::DocumentNotFound(doc_id))?;

        // Remove from inverted index
        {
            let tokens = self.tokenizer.unique_terms(&doc.content);
            let mut idx = self.inverted_index.write();
            for token in tokens {
                if let Some(posting) = idx.get_mut(&token) {
                    posting.remove_document(doc_id);
                    if posting.is_empty() {
                        idx.remove(&token);
                    }
                }
            }
        }

        // Remove from HNSW index (soft delete)
        self.hnsw_index.write().remove(doc_id);

        // Remove from metadata indices
        self.remove_from_metadata_indices(&doc);

        self.total_documents.fetch_sub(1, Ordering::SeqCst);
        self.index_version.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Update a document
    pub fn update_document(&self, doc_id: DocumentId, updates: DocumentUpdate) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        // Get existing document
        let mut doc = {
            let docs = self.documents.read();
            docs.get(&doc_id)
                .cloned()
                .ok_or(SquidexError::DocumentNotFound(doc_id))?
        };

        // Remove old indices
        self.delete_document(doc_id)?;

        // Apply updates
        updates.apply_to(&mut doc);

        // Re-index
        self.index_document(doc)?;
        Ok(())
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: DocumentId) -> Option<Document> {
        self.documents.read().get(&doc_id).cloned()
    }

    /// Keyword search using BM25 ranking
    pub fn keyword_search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let idx = self.inverted_index.read();
        let docs = self.documents.read();

        let total_docs = self.total_documents.load(Ordering::SeqCst) as f32;
        if total_docs == 0.0 {
            return Vec::new();
        }

        let avg_doc_len = self.compute_avg_doc_length();

        let mut scores: HashMap<DocumentId, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(posting) = idx.get(token) {
                let df = posting.document_frequency() as f32;

                for &doc_id in &posting.doc_ids {
                    if let Some(doc) = docs.get(&doc_id) {
                        let tf = *posting.term_frequencies.get(&doc_id).unwrap_or(&0) as f32;
                        let doc_len = self.tokenizer.tokenize(&doc.content).len() as f32;
                        let score = bm25_score(tf, df, total_docs, doc_len, avg_doc_len);

                        *scores.entry(doc_id).or_insert(0.0) += score;
                    }
                }
            }
        }

        self.collect_top_k_results(scores, top_k)
    }

    /// Vector similarity search using HNSW with two-phase routing
    ///
    /// - If candidate_count <= brute_force_threshold: use PQ ADC brute force
    /// - Else: use HNSW with adaptive ef_search
    pub fn vector_search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let index = self.hnsw_index.read();
        let settings = self.settings.read();

        if query_embedding.len() != settings.vector_dimensions {
            return Vec::new();
        }

        // brute_force_threshold for filtered ANN
        const BRUTE_FORCE_THRESHOLD: usize = 1000;

        let total_vectors = index.len();
        drop(settings);

        // If HNSW is trained and we have enough vectors, use HNSW
        if index.is_trained() && total_vectors > BRUTE_FORCE_THRESHOLD {
            // Use adaptive ef_search based on collection size
            let ef = index.compute_adaptive_ef(total_vectors);

            match index.search_with_ef(query_embedding, top_k, ef) {
                Ok(results) => {
                    // Convert distance to similarity score (inverse relationship)
                    let max_dist = results.iter().map(|(_, d)| *d).fold(0.0f32, f32::max);
                    let max_dist = if max_dist == 0.0 { 1.0 } else { max_dist };

                    return results
                        .into_iter()
                        .map(|(doc_id, distance)| {
                            let score = 1.0 - (distance / (max_dist + 1.0));
                            SearchResult::new(doc_id, score)
                        })
                        .collect();
                }
                Err(e) => {
                    tracing::warn!("HNSW search failed, falling back to brute force: {}", e);
                }
            }
        }

        // Brute force path: use PQ ADC when trained, else raw embeddings
        if index.is_trained() {
            match index.brute_force_search(query_embedding, top_k) {
                Ok(results) => {
                    let max_dist = results.iter().map(|(_, d)| *d).fold(0.0f32, f32::max);
                    let max_dist = if max_dist == 0.0 { 1.0 } else { max_dist };

                    return results
                        .into_iter()
                        .map(|(doc_id, distance)| {
                            let score = 1.0 - (distance / (max_dist + 1.0));
                            SearchResult::new(doc_id, score)
                        })
                        .collect();
                }
                Err(e) => {
                    tracing::warn!("PQ brute force failed, falling back to raw: {}", e);
                }
            }
        }

        // Final fallback: brute force on raw embeddings
        let settings = self.settings.read();
        let docs = self.documents.read();
        let similarity_metric = settings.similarity_metric.clone();
        drop(settings);

        let mut scores: Vec<(DocumentId, f32)> = docs
            .iter()
            .map(|(doc_id, doc)| {
                let score = match similarity_metric {
                    SimilarityMetric::Cosine => cosine_similarity(query_embedding, &doc.embedding),
                    SimilarityMetric::Euclidean => {
                        euclidean_similarity(query_embedding, &doc.embedding)
                    }
                    SimilarityMetric::DotProduct => dot_product(query_embedding, &doc.embedding),
                };
                (*doc_id, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        scores
            .into_iter()
            .map(|(doc_id, score)| SearchResult::new(doc_id, score))
            .collect()
    }

    /// Hybrid search combining keyword and vector
    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        keyword_weight: f32,
    ) -> Vec<SearchResult> {
        let vector_weight = 1.0 - keyword_weight;

        // Get results from both methods
        let keyword_results = self.keyword_search(query, top_k * 2);
        let vector_results = self.vector_search(query_embedding, top_k * 2);

        // Normalize and combine scores
        let keyword_max = keyword_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);
        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);

        let mut combined: HashMap<DocumentId, f32> = HashMap::new();

        for result in keyword_results {
            let normalized = normalize_score(result.score, keyword_max);
            *combined.entry(result.doc_id).or_insert(0.0) += normalized * keyword_weight;
        }

        for result in vector_results {
            let normalized = normalize_score(result.score, vector_max);
            *combined.entry(result.doc_id).or_insert(0.0) += normalized * vector_weight;
        }

        self.collect_top_k_results(combined, top_k)
    }

    /// Apply filters to a set of document IDs
    pub fn apply_filters(&self, doc_ids: &mut HashSet<DocumentId>, filters: &[Filter]) {
        if filters.is_empty() {
            return;
        }

        for filter in filters {
            let matching = self.get_filtered_documents(filter);
            doc_ids.retain(|id| matching.contains(id));
        }
    }

    /// Get documents matching a specific filter
    fn get_filtered_documents(&self, filter: &Filter) -> HashSet<DocumentId> {
        match filter {
            Filter::Tag(tag) => self.tag_index.read().get(tag).cloned().unwrap_or_default(),
            Filter::Source(source) => self
                .source_index
                .read()
                .get(source)
                .cloned()
                .unwrap_or_default(),
            Filter::DateRange { start, end } => {
                let idx = self.date_index.read();
                idx.range(start..=end)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect()
            }
            Filter::Custom { key, value } => {
                let docs = self.documents.read();
                docs.iter()
                    .filter(|(_, doc)| doc.metadata.custom.get(key) == Some(value))
                    .map(|(id, _)| *id)
                    .collect()
            }
        }
    }

    /// Compact the index - remove empty structures
    pub fn optimize_index(&self) -> Result<()> {
        // Remove empty posting lists
        self.inverted_index
            .write()
            .retain(|_, posting| !posting.is_empty());

        // Shrink hash maps
        self.documents.write().shrink_to_fit();
        self.inverted_index.write().shrink_to_fit();
        // Note: HnswIndex uses compact storage by design (PQ + HNSW)

        Ok(())
    }

    // Helper methods

    fn collect_top_k_results(
        &self,
        scores: HashMap<DocumentId, f32>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        let mut results: Vec<_> = scores
            .into_iter()
            .map(|(doc_id, score)| (doc_id, score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
            .into_iter()
            .map(|(doc_id, score)| SearchResult::new(doc_id, score))
            .collect()
    }

    fn compute_avg_doc_length(&self) -> f32 {
        let docs = self.documents.read();
        if docs.is_empty() {
            return 1.0;
        }
        let total_tokens: usize = docs
            .values()
            .map(|doc| self.tokenizer.tokenize(&doc.content).len())
            .sum();
        total_tokens as f32 / docs.len() as f32
    }

    fn update_metadata_indices(&self, doc: &Document) {
        // Tag index
        {
            let mut idx = self.tag_index.write();
            for tag in &doc.metadata.tags {
                idx.entry(tag.clone()).or_default().insert(doc.id);
            }
        }

        // Source index
        if let Some(source) = &doc.metadata.source {
            self.source_index
                .write()
                .entry(source.clone())
                .or_default()
                .insert(doc.id);
        }

        // Date index
        self.date_index
            .write()
            .entry(doc.created_at)
            .or_default()
            .insert(doc.id);
    }

    fn remove_from_metadata_indices(&self, doc: &Document) {
        // Tag index
        {
            let mut idx = self.tag_index.write();
            for tag in &doc.metadata.tags {
                if let Some(set) = idx.get_mut(tag) {
                    set.remove(&doc.id);
                    if set.is_empty() {
                        idx.remove(tag);
                    }
                }
            }
        }

        // Source index
        if let Some(source) = &doc.metadata.source {
            let mut idx = self.source_index.write();
            if let Some(set) = idx.get_mut(source) {
                set.remove(&doc.id);
                if set.is_empty() {
                    idx.remove(source);
                }
            }
        }

        // Date index
        {
            let mut idx = self.date_index.write();
            if let Some(set) = idx.get_mut(&doc.created_at) {
                set.remove(&doc.id);
                if set.is_empty() {
                    idx.remove(&doc.created_at);
                }
            }
        }
    }

    // Public accessors for API layer

    pub fn total_documents(&self) -> u64 {
        self.total_documents.load(Ordering::SeqCst)
    }

    pub fn index_version(&self) -> u64 {
        self.index_version.load(Ordering::SeqCst)
    }

    pub fn settings(&self) -> IndexSettings {
        self.settings.read().clone()
    }

    /// Allocate and return the next document ID
    pub fn next_document_id(&self) -> DocumentId {
        self.next_doc_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Create a complete snapshot of the search index
    pub fn create_snapshot(&self) -> Vec<u8> {
        let hnsw_snapshot = self.hnsw_index.read().create_snapshot();

        let snapshot = SearchSnapshot::new(
            self.documents.read().clone(),
            self.inverted_index.read().clone(),
            hnsw_snapshot,
            self.tag_index.read().clone(),
            self.source_index.read().clone(),
            self.date_index.read().clone(),
            self.next_doc_id.load(Ordering::SeqCst),
            self.total_documents.load(Ordering::SeqCst),
            self.index_version.load(Ordering::SeqCst),
            self.settings.read().clone(),
        );

        snapshot.to_bytes().unwrap_or_default()
    }

    /// Restore from a snapshot
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let snapshot = SearchSnapshot::from_bytes(data).map_err(|e| {
            SquidexError::Internal(format!("Failed to deserialize snapshot: {}", e))
        })?;

        // Validate version compatibility
        if !snapshot.is_compatible() {
            return Err(SquidexError::IncompatibleSnapshot {
                expected: SNAPSHOT_VERSION,
                actual: snapshot.version,
            });
        }

        // Get cache size from settings
        let cache_size = snapshot.settings.pq_config.full_precision_cache_size;

        // Restore HNSW index from snapshot
        let restored_hnsw_index =
            HnswIndex::restore_from_snapshot(snapshot.hnsw_index, cache_size);

        // Atomically restore all state
        *self.documents.write() = snapshot.documents;
        *self.inverted_index.write() = snapshot.inverted_index;
        *self.hnsw_index.write() = restored_hnsw_index;
        *self.tag_index.write() = snapshot.tag_index;
        *self.source_index.write() = snapshot.source_index;
        *self.date_index.write() = snapshot.date_index;
        self.next_doc_id
            .store(snapshot.next_doc_id, Ordering::SeqCst);
        self.total_documents
            .store(snapshot.total_documents, Ordering::SeqCst);
        self.index_version
            .store(snapshot.index_version, Ordering::SeqCst);
        *self.settings.write() = snapshot.settings;

        Ok(())
    }

    /// Apply a command (used by Raft state machine trait)
    pub fn apply_command(&self, command: &[u8]) -> Result<Bytes> {
        let cmd: Command = bincode::deserialize(command)
            .map_err(|e| SquidexError::Internal(format!("Failed to deserialize command: {}", e)))?;

        match cmd {
            Command::IndexDocument(doc) => {
                let doc_id = self.index_document(doc)?;
                Ok(Bytes::from(doc_id.to_le_bytes().to_vec()))
            }

            Command::UpdateDocument { id, updates } => {
                self.update_document(id, updates)?;
                Ok(Bytes::from("OK"))
            }

            Command::DeleteDocument(doc_id) => {
                self.delete_document(doc_id)?;
                Ok(Bytes::from("OK"))
            }

            Command::BatchIndex(docs) => {
                let mut indexed = Vec::new();
                for doc in docs {
                    match self.index_document(doc) {
                        Ok(id) => indexed.push(id),
                        Err(e) => {
                            return Err(SquidexError::IndexError(format!(
                                "Batch index failed: {}",
                                e
                            )))
                        }
                    }
                }
                let response =
                    bincode::serialize(&indexed).map_err(|e| SquidexError::Serialization(e))?;
                Ok(Bytes::from(response))
            }

            Command::BatchDelete(doc_ids) => {
                for doc_id in doc_ids {
                    self.delete_document(doc_id)?;
                }
                Ok(Bytes::from("OK"))
            }

            Command::OptimizeIndex => {
                self.optimize_index()?;
                Ok(Bytes::from("OK"))
            }

            Command::CompactIndex => {
                self.optimize_index()?; // Same implementation for now
                Ok(Bytes::from("OK"))
            }

            Command::UpdateSettings(settings) => {
                *self.settings.write() = settings;
                Ok(Bytes::from("OK"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_document(id: u64, content: &str, embedding: Vec<f32>) -> Document {
        Document {
            id,
            content: content.to_string(),
            embedding,
            metadata: DocumentMetadata::default(),
            created_at: current_timestamp(),
            updated_at: current_timestamp(),
        }
    }

    #[test]
    fn test_index_and_retrieve_document() {
        let settings = IndexSettings::default();
        let machine = SearchStateMachine::new(settings);

        let doc = create_test_document(1, "hello world", vec![1.0; 384]);
        machine.index_document(doc.clone()).unwrap();

        let retrieved = machine.get_document(1).unwrap();
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.content, "hello world");
        assert_eq!(machine.total_documents(), 1);
    }

    #[test]
    fn test_delete_document() {
        let settings = IndexSettings::default();
        let machine = SearchStateMachine::new(settings);

        let doc = create_test_document(1, "hello world", vec![1.0; 384]);
        machine.index_document(doc).unwrap();
        assert_eq!(machine.total_documents(), 1);

        machine.delete_document(1).unwrap();
        assert_eq!(machine.total_documents(), 0);
        assert!(machine.get_document(1).is_none());
    }

    #[test]
    fn test_keyword_search() {
        let settings = IndexSettings::default();
        let machine = SearchStateMachine::new(settings);

        let doc1 = create_test_document(1, "rust programming language", vec![1.0; 384]);
        let doc2 = create_test_document(2, "python programming tutorial", vec![1.0; 384]);
        let doc3 = create_test_document(3, "rust async programming", vec![1.0; 384]);

        machine.index_document(doc1).unwrap();
        machine.index_document(doc2).unwrap();
        machine.index_document(doc3).unwrap();

        let results = machine.keyword_search("rust programming", 10);

        assert!(!results.is_empty());
        // Both doc1 and doc3 should match "rust", both have "programming"
        assert!(results.iter().any(|r| r.doc_id == 1 || r.doc_id == 3));
    }

    #[test]
    fn test_vector_search() {
        // Use 3 dimensions with 3 subspaces (1 dim per subspace) for testing
        let mut settings = IndexSettings::default();
        settings.vector_dimensions = 3;
        settings.pq_config.num_subspaces = 3;
        settings.pq_config.min_training_vectors = 10; // Won't trigger training with 3 docs

        let doc1 = create_test_document(1, "test doc 1", vec![1.0, 0.0, 0.0]);
        let doc2 = create_test_document(2, "test doc 2", vec![0.0, 1.0, 0.0]);
        let doc3 = create_test_document(3, "test doc 3", vec![0.9, 0.1, 0.0]);

        let machine = SearchStateMachine::new(settings);

        machine.index_document(doc1).unwrap();
        machine.index_document(doc2).unwrap();
        machine.index_document(doc3).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = machine.vector_search(&query, 2);

        assert_eq!(results.len(), 2);
        // doc1 should be most similar, then doc3
        assert_eq!(results[0].doc_id, 1);
    }

    #[test]
    fn test_snapshot_and_restore() {
        let settings = IndexSettings::default();
        let machine = SearchStateMachine::new(settings.clone());

        let doc = create_test_document(1, "hello world", vec![1.0; 384]);
        machine.index_document(doc).unwrap();

        // Create snapshot
        let snapshot_data = machine.create_snapshot();
        assert!(!snapshot_data.is_empty());

        // Create new machine and restore
        let machine2 = SearchStateMachine::new(settings);
        machine2.restore_snapshot(&snapshot_data).unwrap();

        assert_eq!(machine2.total_documents(), 1);
        let retrieved = machine2.get_document(1).unwrap();
        assert_eq!(retrieved.content, "hello world");
    }
}
