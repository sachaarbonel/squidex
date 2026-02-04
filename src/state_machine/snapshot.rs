use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::IndexSettings;
use crate::models::{Document, DocumentId, PostingList};
use crate::vector::HnswSnapshot;

/// Snapshot version for compatibility checking
/// Bumped to v4 for schema/mapping support in IndexSettings
pub const SNAPSHOT_VERSION: u32 = 4;

/// Complete snapshot of the search state machine
#[derive(Clone, Serialize, Deserialize)]
pub struct SearchSnapshot {
    pub version: u32,
    pub documents: HashMap<DocumentId, Document>,
    pub inverted_index: HashMap<String, PostingList>,
    pub hnsw_index: HnswSnapshot,
    pub tag_index: HashMap<String, HashSet<DocumentId>>,
    pub source_index: HashMap<String, HashSet<DocumentId>>,
    pub date_index: BTreeMap<u64, HashSet<DocumentId>>,
    pub next_doc_id: u64,
    pub total_documents: u64,
    pub index_version: u64,
    pub settings: IndexSettings,
}

impl SearchSnapshot {
    /// Create a new snapshot with the current version
    pub fn new(
        documents: HashMap<DocumentId, Document>,
        inverted_index: HashMap<String, PostingList>,
        hnsw_index: HnswSnapshot,
        tag_index: HashMap<String, HashSet<DocumentId>>,
        source_index: HashMap<String, HashSet<DocumentId>>,
        date_index: BTreeMap<u64, HashSet<DocumentId>>,
        next_doc_id: u64,
        total_documents: u64,
        index_version: u64,
        settings: IndexSettings,
    ) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            documents,
            inverted_index,
            hnsw_index,
            tag_index,
            source_index,
            date_index,
            next_doc_id,
            total_documents,
            index_version,
            settings,
        }
    }

    /// Serialize snapshot to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize snapshot from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
    }

    /// Check if this snapshot version is compatible
    pub fn is_compatible(&self) -> bool {
        self.version <= SNAPSHOT_VERSION
    }

    /// Get the size of this snapshot in bytes (approximate)
    pub fn estimated_size(&self) -> usize {
        // Rough estimation for monitoring
        let vector_store = &self.hnsw_index.vector_store;
        let quantized_size = vector_store.quantized_vectors.len() * vector_store.num_subspaces;
        let codebook_size = vector_store.num_subspaces
            * 256
            * (vector_store.dimensions / vector_store.num_subspaces)
            * 4;
        let buffer_size = vector_store.training_buffer.len() * vector_store.dimensions * 4;
        let graph_size = self.hnsw_index.layers.len() * 1000; // Approximate per-layer

        std::mem::size_of::<Self>()
            + self.documents.len() * 1000
            + self.inverted_index.len() * 100
            + quantized_size
            + codebook_size
            + buffer_size
            + graph_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DocumentMetadata;
    use crate::vector::{HnswParams, VectorStoreSnapshot};

    fn create_empty_hnsw_snapshot() -> HnswSnapshot {
        HnswSnapshot {
            layers: Vec::new(),
            entry_point: None,
            max_layer: 0,
            node_levels: HashMap::new(),
            node_count: 0,
            deleted: HashSet::new(),
            vector_store: VectorStoreSnapshot {
                dimensions: 384,
                num_subspaces: 24,
                trained: false,
                codebook_centroids: Vec::new(),
                quantized_vectors: HashMap::new(),
                training_buffer: Vec::new(),
            },
            params: HnswParams::default(),
        }
    }

    #[test]
    fn test_snapshot_serialization_roundtrip() {
        let mut documents = HashMap::new();
        documents.insert(
            1,
            Document {
                id: 1,
                content: "test".to_string(),
                embedding: vec![1.0; 384],
                metadata: DocumentMetadata::default(),
                created_at: 1000,
                updated_at: 1000,
            },
        );

        let snapshot = SearchSnapshot::new(
            documents,
            HashMap::new(),
            create_empty_hnsw_snapshot(),
            HashMap::new(),
            HashMap::new(),
            BTreeMap::new(),
            2,
            1,
            1,
            IndexSettings::default(),
        );

        let bytes = snapshot.to_bytes().unwrap();
        let restored = SearchSnapshot::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version, SNAPSHOT_VERSION);
        assert_eq!(restored.next_doc_id, 2);
        assert_eq!(restored.total_documents, 1);
        assert_eq!(restored.documents.len(), 1);
    }

    #[test]
    fn test_snapshot_compatibility() {
        let snapshot = SearchSnapshot::new(
            HashMap::new(),
            HashMap::new(),
            create_empty_hnsw_snapshot(),
            HashMap::new(),
            HashMap::new(),
            BTreeMap::new(),
            1,
            0,
            0,
            IndexSettings::default(),
        );

        assert!(snapshot.is_compatible());
    }
}
