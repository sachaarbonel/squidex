use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::IndexSettings;
use crate::models::{Document, DocumentId, Embedding, PostingList};

/// Snapshot version for compatibility checking
pub const SNAPSHOT_VERSION: u32 = 1;

/// Complete snapshot of the search state machine
#[derive(Clone, Serialize, Deserialize)]
pub struct SearchSnapshot {
    pub version: u32,
    pub documents: HashMap<DocumentId, Document>,
    pub inverted_index: HashMap<String, PostingList>,
    pub vector_store: HashMap<DocumentId, Embedding>,
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
        vector_store: HashMap<DocumentId, Embedding>,
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
            vector_store,
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
        std::mem::size_of::<Self>()
            + self.documents.len() * 1000 // Approximate per-document size
            + self.inverted_index.len() * 100 // Approximate per-term size
            + self.vector_store.len() * self.settings.vector_dimensions * 4 // f32 = 4 bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DocumentMetadata;

    #[test]
    fn test_snapshot_serialization_roundtrip() {
        let mut documents = HashMap::new();
        documents.insert(
            1,
            Document {
                id: 1,
                content: "test".to_string(),
                embedding: vec![1.0, 2.0, 3.0],
                metadata: DocumentMetadata::default(),
                created_at: 1000,
                updated_at: 1000,
            },
        );

        let snapshot = SearchSnapshot::new(
            documents,
            HashMap::new(),
            HashMap::new(),
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
            HashMap::new(),
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
