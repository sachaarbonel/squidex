use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use crossbeam::channel;
use dashmap::{DashMap, DashSet};
use ordered_float::OrderedFloat;
use parking_lot::{Condvar, Mutex, RwLock};

use crate::config::IndexSettings;
use crate::error::{Result, SquidexError};
use crate::models::*;
use crate::persistence::DocStore;
use crate::segment::{SegmentIndex, SegmentIndexConfig};
use crate::state_machine::indexer::{spawn_indexer, IndexOp};
use crate::state_machine::scoring::*;
use crate::state_machine::snapshot::{SearchSnapshot, SNAPSHOT_VERSION};
use crate::tokenizer::Tokenizer;
use crate::vector::HnswIndex;

/// Production-ready distributed search state machine (async indexing).
pub struct SearchStateMachine {
    doc_store: Arc<DocStore>,
    text_index: Arc<SegmentIndex>,
    hnsw_index: Arc<RwLock<HnswIndex>>,

    // Metadata indices for filtering (in-memory)
    tag_index: DashMap<String, DashSet<DocumentId>>,
    source_index: DashMap<String, DashSet<DocumentId>>,
    date_index: DashMap<u64, DashSet<DocumentId>>,

    // Counters and state
    next_doc_id: AtomicU64,
    total_documents: AtomicU64,
    index_applied_index: Arc<AtomicU64>,

    // Configuration
    settings: RwLock<IndexSettings>,
    tokenizer: Tokenizer,
    pq_training_pending: AtomicBool,

    // Async indexer
    index_cv: Arc<(Mutex<()>, Condvar)>,
    indexer_tx: channel::Sender<IndexOp>,
    _indexer_handle: std::thread::JoinHandle<()>,
}

struct RebuildOutput {
    next_doc_id: u64,
    total_docs: u64,
    max_applied: u64,
    tag_index: HashMap<String, HashSet<DocumentId>>,
    source_index: HashMap<String, HashSet<DocumentId>>,
    date_index: BTreeMap<u64, HashSet<DocumentId>>,
}

impl SearchStateMachine {
    pub fn new(settings: IndexSettings, data_dir: PathBuf) -> Result<Self> {
        let tokenizer = Tokenizer::new(&settings.tokenizer_config);
        let hnsw_index = HnswIndex::new(
            settings.vector_dimensions,
            settings.pq_config.num_subspaces,
            Default::default(),
        );
        let doc_store = Arc::new(DocStore::open(data_dir.join("docstore"))?);
        let text_index = Arc::new(
            SegmentIndex::open_with_store(SegmentIndexConfig::default(), data_dir.join("segments"))
                .map_err(|e| SquidexError::Io(e))?,
        );
        let hnsw_index = Arc::new(RwLock::new(hnsw_index));

        let rebuild = Self::rebuild_from_store(&doc_store, &text_index, &hnsw_index, &tokenizer)?;
        doc_store.set_index_applied_index(rebuild.max_applied)?;

        let index_applied_index = Arc::new(AtomicU64::new(
            doc_store
                .get_index_applied_index()
                .unwrap_or(rebuild.max_applied),
        ));

        let index_cv = Arc::new((Mutex::new(()), Condvar::new()));

        // Spawn indexer thread
        let (tx, rx) = channel::unbounded();
        let handles = spawn_indexer(
            tx.clone(),
            rx,
            text_index.clone(),
            hnsw_index.clone(),
            doc_store.clone(),
            index_applied_index.clone(),
            index_cv.clone(),
            settings.clone(),
        );

        let tag_index = DashMap::new();
        for (tag, ids) in rebuild.tag_index {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            tag_index.insert(tag, set);
        }
        let source_index = DashMap::new();
        for (source, ids) in rebuild.source_index {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            source_index.insert(source, set);
        }
        let date_index = DashMap::new();
        for (date, ids) in rebuild.date_index {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            date_index.insert(date, set);
        }

        Ok(Self {
            doc_store,
            text_index,
            hnsw_index,
            tag_index,
            source_index,
            date_index,
            next_doc_id: AtomicU64::new(rebuild.next_doc_id),
            total_documents: AtomicU64::new(rebuild.total_docs),
            index_applied_index,
            settings: RwLock::new(settings),
            tokenizer,
            pq_training_pending: AtomicBool::new(false),
            index_cv,
            indexer_tx: handles.tx,
            _indexer_handle: handles.join,
        })
    }

    fn rebuild_from_store(
        doc_store: &Arc<DocStore>,
        text_index: &Arc<SegmentIndex>,
        hnsw_index: &Arc<RwLock<HnswIndex>>,
        tokenizer: &Tokenizer,
    ) -> Result<RebuildOutput> {
        let pointers = doc_store.list_doc_pointers()?;
        let tombstones = doc_store.list_tombstones()?;
        let tombstone_set: HashSet<u64> = tombstones.iter().map(|(d, _)| *d).collect();

        let mut max_applied = tombstones.iter().map(|(_, r)| *r).max().unwrap_or(0);
        let mut total_docs = 0u64;
        let mut next_doc_id = 1u64;
        let mut tag_index: HashMap<String, HashSet<DocumentId>> = HashMap::new();
        let mut source_index: HashMap<String, HashSet<DocumentId>> = HashMap::new();
        let mut date_index: BTreeMap<u64, HashSet<DocumentId>> = BTreeMap::new();

        for (doc_id, ptr) in pointers {
            if tombstone_set.contains(&doc_id) {
                max_applied = max_applied.max(ptr.raft_index);
                continue;
            }

            if let Some(doc) = doc_store.get_document(doc_id)? {
                let term_freqs = tokenizer.compute_term_frequencies(&doc.content);
                let doc_len = tokenizer.tokenize(&doc.content).len() as u32;
                let version = crate::segment::Version::new(doc.updated_at);
                let _ =
                    text_index.index_document(doc.id, version, term_freqs, doc_len, ptr.raft_index);
                {
                    let mut hnsw = hnsw_index.write();
                    let _ = hnsw.insert(doc.id, &doc.embedding);
                }
                total_docs += 1;
                next_doc_id = next_doc_id.max(doc.id + 1);
                max_applied = max_applied.max(ptr.raft_index);
                Self::update_metadata_maps(
                    &doc,
                    &mut tag_index,
                    &mut source_index,
                    &mut date_index,
                );
            }
        }

        Ok(RebuildOutput {
            next_doc_id,
            total_docs,
            max_applied,
            tag_index,
            source_index,
            date_index,
        })
    }

    /// Allocate and return the next document ID
    pub fn next_document_id(&self) -> DocumentId {
        self.next_doc_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn total_documents(&self) -> u64 {
        self.total_documents.load(Ordering::SeqCst)
    }

    pub fn index_version(&self) -> u64 {
        self.index_applied_index.load(Ordering::SeqCst)
    }

    pub fn settings(&self) -> IndexSettings {
        self.settings.read().clone()
    }

    pub fn wait_for_index(&self, min_index: u64, timeout_ms: u64) -> Result<()> {
        let deadline = std::time::Instant::now()
            .checked_add(std::time::Duration::from_millis(timeout_ms))
            .unwrap_or(std::time::Instant::now());
        let (lock, cv) = &*self.index_cv;
        let mut guard = lock.lock();
        loop {
            if self.index_applied_index.load(Ordering::SeqCst) >= min_index {
                return Ok(());
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let remaining = deadline - now;
            let _ = cv.wait_for(&mut guard, remaining);
        }
        Err(SquidexError::SearchError("index_not_ready".to_string()))
    }

    pub fn get_document(&self, doc_id: DocumentId) -> Option<Document> {
        self.doc_store.get_document(doc_id).ok().flatten()
    }

    pub fn apply_command(&self, raft_index: u64, command: &[u8]) -> Result<Bytes> {
        let cmd: Command = bincode::deserialize(command)
            .map_err(|e| SquidexError::Internal(format!("Failed to deserialize command: {}", e)))?;

        self.apply_parsed_command(raft_index, cmd)
    }

    pub fn apply_parsed_command(&self, raft_index: u64, cmd: Command) -> Result<Bytes> {
        match cmd {
            Command::IndexDocument(doc) => {
                let doc_id = self.handle_index(raft_index, doc)?;
                Ok(Bytes::from(doc_id.to_le_bytes().to_vec()))
            }
            Command::UpdateDocument { id, updates } => {
                self.handle_update(raft_index, id, updates)?;
                Ok(Bytes::from("OK"))
            }
            Command::DeleteDocument(doc_id) => {
                self.handle_delete(raft_index, doc_id)?;
                Ok(Bytes::from("OK"))
            }
            Command::BatchIndex(docs) => {
                let mut indexed = Vec::new();
                for doc in docs {
                    indexed.push(self.handle_index(raft_index, doc)?);
                }
                let response = bincode::serialize(&indexed).map_err(SquidexError::Serialization)?;
                Ok(Bytes::from(response))
            }
            Command::BatchDelete(doc_ids) => {
                for doc_id in doc_ids {
                    self.handle_delete(raft_index, doc_id)?;
                }
                Ok(Bytes::from("OK"))
            }
            Command::OptimizeIndex | Command::CompactIndex => Ok(Bytes::from("OK")),
            Command::UpdateSettings(settings) => {
                *self.settings.write() = settings;
                Ok(Bytes::from("OK"))
            }
        }
    }

    fn handle_index(&self, raft_index: u64, doc: Document) -> Result<DocumentId> {
        let doc_id = doc.id;
        self.validate_embedding(&doc)?;
        let existed = self.doc_store.exists(doc_id)?;
        self.doc_store.put_document(&doc, raft_index)?;
        self.update_metadata_indices(&doc);
        self.total_documents
            .fetch_add((!existed) as u64, Ordering::SeqCst);
        self.indexer_tx
            .send(IndexOp::Upsert { doc, raft_index })
            .map_err(|e| SquidexError::Internal(format!("indexer send failed: {}", e)))?;
        Ok(doc_id)
    }

    fn handle_update(
        &self,
        raft_index: u64,
        doc_id: DocumentId,
        updates: DocumentUpdate,
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut doc = self
            .doc_store
            .get_document(doc_id)?
            .ok_or(SquidexError::DocumentNotFound(doc_id))?;
        updates.apply_to(&mut doc);
        self.handle_index(raft_index, doc)?;
        Ok(())
    }

    fn handle_delete(&self, raft_index: u64, doc_id: DocumentId) -> Result<()> {
        let existed = self.doc_store.exists(doc_id)?;
        if existed {
            if let Some(doc) = self.doc_store.get_document(doc_id)? {
                self.remove_from_metadata_indices(&doc);
            }
            self.total_documents.fetch_sub(1, Ordering::SeqCst);
        }
        self.doc_store.tombstone(doc_id, raft_index)?;
        self.indexer_tx
            .send(IndexOp::Delete { doc_id, raft_index })
            .map_err(|e| SquidexError::Internal(format!("indexer send failed: {}", e)))?;
        Ok(())
    }

    fn validate_embedding(&self, doc: &Document) -> Result<()> {
        let settings = self.settings.read();
        if doc.embedding.len() != settings.vector_dimensions {
            return Err(SquidexError::InvalidEmbeddingDimensions {
                expected: settings.vector_dimensions,
                actual: doc.embedding.len(),
            });
        }
        Ok(())
    }

    pub fn keyword_search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let query_terms = self.tokenizer.tokenize(query);
        if query_terms.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let results = self
            .text_index
            .keyword_search(&query_terms, top_k * 2)
            .unwrap_or_default();
        let mut filtered = Vec::new();
        for r in results {
            if self.doc_store.is_tombstoned(r.doc_id).unwrap_or(false) {
                continue;
            }
            filtered.push(SearchResult::new(r.doc_id, r.score));
        }
        filtered.truncate(top_k);
        filtered
    }

    pub fn vector_search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        if top_k == 0 {
            return Vec::new();
        }
        let settings = self.settings.read().clone();
        if query_embedding.len() != settings.vector_dimensions {
            return Vec::new();
        }

        let index = self.hnsw_index.read();
        // Prefer HNSW path
        if index.len() > 0 && index.is_trained() {
            if let Ok(results) = index.search_with_ef(
                query_embedding,
                top_k,
                index.compute_adaptive_ef(index.len()),
            ) {
                return results
                    .into_iter()
                    .filter(|(doc_id, _)| !self.doc_store.is_tombstoned(*doc_id).unwrap_or(false))
                    .map(|(doc_id, distance)| {
                        // Convert distance to similarity score
                        let score = 1.0 - distance;
                        SearchResult::new(doc_id, score)
                    })
                    .collect();
            }
        }
        drop(index);

        // Brute force over stored docs
        let mut scored: Vec<(DocumentId, f32)> = Vec::new();
        if let Ok(ptrs) = self.doc_store.list_doc_pointers() {
            for (doc_id, _) in ptrs {
                if self.doc_store.is_tombstoned(doc_id).unwrap_or(false) {
                    continue;
                }
                if let Some(doc) = self.doc_store.get_document(doc_id).ok().flatten() {
                    let score = match settings.similarity_metric {
                        SimilarityMetric::Cosine => {
                            cosine_similarity(query_embedding, &doc.embedding)
                        }
                        SimilarityMetric::Euclidean => {
                            euclidean_similarity(query_embedding, &doc.embedding)
                        }
                        SimilarityMetric::DotProduct => {
                            dot_product(query_embedding, &doc.embedding)
                        }
                    };
                    scored.push((doc_id, score));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
            .into_iter()
            .map(|(doc_id, score)| SearchResult::new(doc_id, score))
            .collect()
    }

    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        keyword_weight: f32,
    ) -> Vec<SearchResult> {
        if top_k == 0 {
            return Vec::new();
        }
        let vector_weight = 1.0 - keyword_weight;
        let keyword_results = self.keyword_search(query, top_k * 2);
        let vector_results = self.vector_search(query_embedding, top_k * 2);

        let keyword_max = keyword_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);
        let vector_max = vector_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);

        let mut combined: HashMap<DocumentId, f32> = HashMap::new();
        for r in keyword_results {
            let normalized = normalize_score(r.score, keyword_max);
            *combined.entry(r.doc_id).or_insert(0.0) += normalized * keyword_weight;
        }
        for r in vector_results {
            let normalized = normalize_score(r.score, vector_max);
            *combined.entry(r.doc_id).or_insert(0.0) += normalized * vector_weight;
        }

        self.collect_top_k_results(combined, top_k)
    }

    pub fn apply_filters(&self, doc_ids: &mut HashSet<DocumentId>, filters: &[Filter]) {
        if filters.is_empty() {
            return;
        }
        for filter in filters {
            let matching = self.get_filtered_documents(filter);
            doc_ids.retain(|id| matching.contains(id));
        }
    }

    fn get_filtered_documents(&self, filter: &Filter) -> HashSet<DocumentId> {
        match filter {
            Filter::Tag(tag) => self
                .tag_index
                .get(tag)
                .map(|set| set.iter().map(|id| *id).collect())
                .unwrap_or_default(),
            Filter::Source(source) => self
                .source_index
                .get(source)
                .map(|set| set.iter().map(|id| *id).collect())
                .unwrap_or_default(),
            Filter::DateRange { start, end } => {
                let mut ids = HashSet::new();
                for entry in self.date_index.iter() {
                    let ts = *entry.key();
                    if ts >= *start && ts <= *end {
                        ids.extend(entry.value().iter().map(|id| *id));
                    }
                }
                ids
            }
            Filter::Custom { key, value } => {
                // Fallback: scan metadata indexes is not available; use doc store
                if let Ok(ptrs) = self.doc_store.list_doc_pointers() {
                    ptrs.into_iter()
                        .filter_map(|(doc_id, _)| {
                            self.doc_store.get_document(doc_id).ok().flatten()
                        })
                        .filter(|doc| doc.metadata.custom.get(key) == Some(value))
                        .map(|doc| doc.id)
                        .collect()
                } else {
                    HashSet::new()
                }
            }
        }
    }

    fn collect_top_k_results(
        &self,
        scores: HashMap<DocumentId, f32>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        if top_k == 0 || scores.is_empty() {
            return Vec::new();
        }

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct ScoreEntry {
            score: OrderedFloat<f32>,
            doc_id: DocumentId,
        }

        impl Ord for ScoreEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.score
                    .cmp(&other.score)
                    .then_with(|| self.doc_id.cmp(&other.doc_id))
            }
        }

        impl PartialOrd for ScoreEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: BinaryHeap<Reverse<ScoreEntry>> = BinaryHeap::new();
        for (doc_id, score) in scores {
            let entry = ScoreEntry {
                score: OrderedFloat(score),
                doc_id,
            };
            if heap.len() < top_k {
                heap.push(Reverse(entry));
            } else if let Some(min) = heap.peek() {
                if entry > min.0 {
                    heap.pop();
                    heap.push(Reverse(entry));
                }
            }
        }

        let mut results = Vec::with_capacity(heap.len());
        while let Some(Reverse(entry)) = heap.pop() {
            results.push(entry);
        }
        results.sort_by(|a, b| b.cmp(a));
        results
            .into_iter()
            .map(|entry| SearchResult::new(entry.doc_id, entry.score.0))
            .collect()
    }

    fn remove_from_metadata_indices(&self, doc: &Document) {
        for tag in &doc.metadata.tags {
            if let Some(set) = self.tag_index.get(tag) {
                set.remove(&doc.id);
                let is_empty = set.len() == 0;
                drop(set);
                if is_empty {
                    self.tag_index.remove(tag);
                }
            }
        }
        if let Some(source) = &doc.metadata.source {
            if let Some(set) = self.source_index.get(source) {
                set.remove(&doc.id);
                let is_empty = set.len() == 0;
                drop(set);
                if is_empty {
                    self.source_index.remove(source);
                }
            }
        }
        if let Some(set) = self.date_index.get(&doc.created_at) {
            set.remove(&doc.id);
            let is_empty = set.len() == 0;
            drop(set);
            if is_empty {
                self.date_index.remove(&doc.created_at);
            }
        }
    }

    fn update_metadata_indices(&self, doc: &Document) {
        for tag in &doc.metadata.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(DashSet::new)
                .insert(doc.id);
        }
        if let Some(source) = &doc.metadata.source {
            self.source_index
                .entry(source.clone())
                .or_insert_with(DashSet::new)
                .insert(doc.id);
        }
        self.date_index
            .entry(doc.created_at)
            .or_insert_with(DashSet::new)
            .insert(doc.id);
    }

    fn update_metadata_maps(
        doc: &Document,
        tags: &mut HashMap<String, HashSet<DocumentId>>,
        sources: &mut HashMap<String, HashSet<DocumentId>>,
        dates: &mut BTreeMap<u64, HashSet<DocumentId>>,
    ) {
        for tag in &doc.metadata.tags {
            tags.entry(tag.clone()).or_default().insert(doc.id);
        }
        if let Some(source) = &doc.metadata.source {
            sources.entry(source.clone()).or_default().insert(doc.id);
        }
        dates.entry(doc.created_at).or_default().insert(doc.id);
    }

    /// Create a snapshot (serialize live docs + HNSW)
    pub fn create_snapshot(&self) -> Vec<u8> {
        let mut documents = HashMap::new();
        if let Ok(ptrs) = self.doc_store.list_doc_pointers() {
            for (doc_id, _) in ptrs {
                if let Some(doc) = self.doc_store.get_document(doc_id).ok().flatten() {
                    documents.insert(doc_id, doc);
                }
            }
        }
        let hnsw_snapshot = self.hnsw_index.read().create_snapshot();
        let mut tag_index = HashMap::new();
        for entry in self.tag_index.iter() {
            let mut set = HashSet::new();
            for id in entry.value().iter() {
                set.insert(*id);
            }
            tag_index.insert(entry.key().clone(), set);
        }
        let mut source_index = HashMap::new();
        for entry in self.source_index.iter() {
            let mut set = HashSet::new();
            for id in entry.value().iter() {
                set.insert(*id);
            }
            source_index.insert(entry.key().clone(), set);
        }
        let mut date_index = BTreeMap::new();
        for entry in self.date_index.iter() {
            let mut set = HashSet::new();
            for id in entry.value().iter() {
                set.insert(*id);
            }
            date_index.insert(*entry.key(), set);
        }
        let snapshot = SearchSnapshot::new(
            documents,
            HashMap::new(), // inverted index not used anymore
            hnsw_snapshot,
            tag_index,
            source_index,
            date_index,
            self.next_doc_id.load(Ordering::SeqCst),
            self.total_documents(),
            self.index_version(),
            self.settings(),
        );
        snapshot.to_bytes().unwrap_or_default()
    }

    /// Restore from snapshot by rewriting the doc store and rebuilding indexes.
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let snapshot = SearchSnapshot::from_bytes(data).map_err(|e| {
            SquidexError::Internal(format!("Failed to deserialize snapshot: {}", e))
        })?;
        if !snapshot.is_compatible() {
            return Err(SquidexError::IncompatibleSnapshot {
                expected: SNAPSHOT_VERSION,
                actual: snapshot.version,
            });
        }

        self.doc_store.clear_all()?;
        for (_, doc) in snapshot.documents.iter() {
            self.doc_store.put_document(doc, snapshot.index_version)?;
        }

        // Rebuild indexes and metadata
        let rebuild = Self::rebuild_from_store(
            &self.doc_store,
            &self.text_index,
            &self.hnsw_index,
            &self.tokenizer,
        )?;
        self.replace_metadata_indices(rebuild.tag_index, rebuild.source_index, rebuild.date_index);
        self.next_doc_id
            .store(rebuild.next_doc_id, Ordering::SeqCst);
        self.total_documents
            .store(rebuild.total_docs, Ordering::SeqCst);
        self.index_applied_index
            .store(rebuild.max_applied, Ordering::SeqCst);
        self.doc_store
            .set_index_applied_index(rebuild.max_applied)?;
        Ok(())
    }

    fn replace_metadata_indices(
        &self,
        tags: HashMap<String, HashSet<DocumentId>>,
        sources: HashMap<String, HashSet<DocumentId>>,
        dates: BTreeMap<u64, HashSet<DocumentId>>,
    ) {
        self.tag_index.clear();
        for (tag, ids) in tags {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            self.tag_index.insert(tag, set);
        }

        self.source_index.clear();
        for (source, ids) in sources {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            self.source_index.insert(source, set);
        }

        self.date_index.clear();
        for (date, ids) in dates {
            let set = DashSet::new();
            for id in ids {
                set.insert(id);
            }
            self.date_index.insert(date, set);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_settings(dim: usize) -> IndexSettings {
        let mut settings = IndexSettings::default();
        settings.vector_dimensions = dim;
        settings.pq_config.num_subspaces = dim; // 1 dim per subspace for tests
        settings.pq_config.min_training_vectors = 10;
        settings
    }

    fn create_doc(id: u64, content: &str, dims: usize) -> Document {
        Document {
            id,
            content: content.to_string(),
            embedding: vec![1.0; dims],
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn test_index_and_get_and_delete() {
        let tmp = TempDir::new().unwrap();
        let dims = 4;
        let settings = create_settings(dims);
        let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

        let doc = create_doc(1, "hello world", dims);
        machine
            .apply_parsed_command(1, Command::IndexDocument(doc))
            .unwrap();
        machine.wait_for_index(1, 2000).unwrap();

        let got = machine.get_document(1).unwrap();
        assert_eq!(got.id, 1);

        machine
            .apply_parsed_command(2, Command::DeleteDocument(1))
            .unwrap();
        machine.wait_for_index(2, 2000).unwrap();
        assert!(machine.get_document(1).is_none());
    }

    #[test]
    fn test_keyword_and_vector_search() {
        let tmp = TempDir::new().unwrap();
        let dims = 3;
        let settings = create_settings(dims);
        let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

        let d1 = Document {
            id: 1,
            content: "rust programming".into(),
            embedding: vec![1.0, 0.0, 0.0],
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        };
        let d2 = Document {
            id: 2,
            content: "python coding".into(),
            embedding: vec![0.0, 1.0, 0.0],
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        };
        machine
            .apply_parsed_command(1, Command::IndexDocument(d1))
            .unwrap();
        machine
            .apply_parsed_command(2, Command::IndexDocument(d2))
            .unwrap();
        machine.wait_for_index(2, 2000).unwrap();

        let kw = machine.keyword_search("rust", 5);
        assert_eq!(kw.len(), 1);
        assert_eq!(kw[0].doc_id, 1);

        let vec_res = machine.vector_search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(vec_res.len(), 2);
        assert_eq!(vec_res[0].doc_id, 1);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let dims = 3;
        let settings = create_settings(dims);
        let machine = SearchStateMachine::new(settings.clone(), tmp.path().to_path_buf()).unwrap();

        for i in 1..=3 {
            let doc = create_doc(i, "snapshot doc", dims);
            machine
                .apply_parsed_command(i as u64, Command::IndexDocument(doc))
                .unwrap();
        }
        machine.wait_for_index(3, 10_000).unwrap();

        let snapshot = machine.create_snapshot();
        assert!(!snapshot.is_empty());

        let tmp2 = TempDir::new().unwrap();
        let machine2 = SearchStateMachine::new(settings, tmp2.path().to_path_buf()).unwrap();
        machine2.restore_snapshot(&snapshot).unwrap();

        assert_eq!(machine2.total_documents(), 3);
        let results = machine2.keyword_search("snapshot", 3);
        assert_eq!(results.len(), 3);
    }
}
