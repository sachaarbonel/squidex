use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique document identifier
pub type DocumentId = u64;

/// Vector embedding (typically 384-1536 dimensions)
pub type Embedding = Vec<f32>;

/// Document with content and embedding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub content: String,
    pub embedding: Embedding,
    pub metadata: DocumentMetadata,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Document metadata
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub custom: HashMap<String, String>,
}

/// Inverted index entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PostingList {
    pub term: String,
    pub doc_ids: Vec<DocumentId>,
    pub term_frequencies: HashMap<DocumentId, u32>,
}

impl PostingList {
    pub fn new(term: String) -> Self {
        Self {
            term,
            doc_ids: Vec::new(),
            term_frequencies: HashMap::new(),
        }
    }

    /// Add a document to this posting list with its term frequency
    pub fn add_document(&mut self, doc_id: DocumentId, frequency: u32) {
        if !self.doc_ids.contains(&doc_id) {
            self.doc_ids.push(doc_id);
        }
        self.term_frequencies.insert(doc_id, frequency);
    }

    /// Remove a document from this posting list
    pub fn remove_document(&mut self, doc_id: DocumentId) {
        self.doc_ids.retain(|&id| id != doc_id);
        self.term_frequencies.remove(&doc_id);
    }

    /// Check if this posting list is empty
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Get the document frequency (number of documents containing this term)
    pub fn document_frequency(&self) -> usize {
        self.doc_ids.len()
    }
}

/// Get current Unix timestamp in seconds
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posting_list_operations() {
        let mut posting = PostingList::new("test".to_string());
        assert!(posting.is_empty());

        posting.add_document(1, 5);
        posting.add_document(2, 3);
        assert_eq!(posting.document_frequency(), 2);
        assert_eq!(posting.term_frequencies.get(&1), Some(&5));

        posting.remove_document(1);
        assert_eq!(posting.document_frequency(), 1);
        assert!(posting.term_frequencies.get(&1).is_none());
    }

    #[test]
    fn test_document_metadata_default() {
        let meta = DocumentMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.source.is_none());
        assert!(meta.tags.is_empty());
        assert!(meta.custom.is_empty());
    }
}
