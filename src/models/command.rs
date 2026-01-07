use serde::{Deserialize, Serialize};

use super::document::{Document, DocumentId, DocumentMetadata, Embedding};
use crate::config::IndexSettings;

/// Commands replicated via Raft consensus
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Command {
    // Document operations
    IndexDocument(Document),
    UpdateDocument {
        id: DocumentId,
        updates: DocumentUpdate,
    },
    DeleteDocument(DocumentId),

    // Batch operations
    BatchIndex(Vec<Document>),
    BatchDelete(Vec<DocumentId>),

    // Index maintenance
    OptimizeIndex,
    CompactIndex,

    // Administrative
    UpdateSettings(IndexSettings),
}

/// Partial document update
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentUpdate {
    pub content: Option<String>,
    pub embedding: Option<Embedding>,
    pub metadata: Option<DocumentMetadata>,
}

impl DocumentUpdate {
    /// Create an empty update
    pub fn new() -> Self {
        Self {
            content: None,
            embedding: None,
            metadata: None,
        }
    }

    /// Set content
    pub fn with_content(mut self, content: String) -> Self {
        self.content = Some(content);
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Check if this update is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_none() && self.embedding.is_none() && self.metadata.is_none()
    }

    /// Apply this update to a document
    pub fn apply_to(&self, doc: &mut Document) {
        if let Some(ref content) = self.content {
            doc.content = content.clone();
        }
        if let Some(ref embedding) = self.embedding {
            doc.embedding = embedding.clone();
        }
        if let Some(ref metadata) = self.metadata {
            doc.metadata = metadata.clone();
        }
        doc.updated_at = super::document::current_timestamp();
    }
}

impl Default for DocumentUpdate {
    fn default() -> Self {
        Self::new()
    }
}

impl Command {
    /// Get a human-readable name for this command (for logging)
    pub fn name(&self) -> &'static str {
        match self {
            Command::IndexDocument(_) => "IndexDocument",
            Command::UpdateDocument { .. } => "UpdateDocument",
            Command::DeleteDocument(_) => "DeleteDocument",
            Command::BatchIndex(_) => "BatchIndex",
            Command::BatchDelete(_) => "BatchDelete",
            Command::OptimizeIndex => "OptimizeIndex",
            Command::CompactIndex => "CompactIndex",
            Command::UpdateSettings(_) => "UpdateSettings",
        }
    }

    /// Check if this command modifies documents
    pub fn is_document_modification(&self) -> bool {
        matches!(
            self,
            Command::IndexDocument(_)
                | Command::UpdateDocument { .. }
                | Command::DeleteDocument(_)
                | Command::BatchIndex(_)
                | Command::BatchDelete(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_update_builder() {
        let update = DocumentUpdate::new()
            .with_content("test".to_string())
            .with_embedding(vec![1.0, 2.0, 3.0]);

        assert!(update.content.is_some());
        assert!(update.embedding.is_some());
        assert!(update.metadata.is_none());
        assert!(!update.is_empty());
    }

    #[test]
    fn test_command_name() {
        let cmd = Command::OptimizeIndex;
        assert_eq!(cmd.name(), "OptimizeIndex");

        let cmd2 = Command::DeleteDocument(42);
        assert_eq!(cmd2.name(), "DeleteDocument");
    }

    #[test]
    fn test_command_is_document_modification() {
        assert!(Command::IndexDocument(Document {
            id: 1,
            content: "test".to_string(),
            embedding: vec![],
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        })
        .is_document_modification());

        assert!(!Command::OptimizeIndex.is_document_modification());
    }
}
