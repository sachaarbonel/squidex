//! Instrumented state machine wrapper for testing
//!
//! This module provides a wrapper around SearchStateMachine that
//! captures all operations for invariant verification.

use crate::error::Result;
use crate::models::{Command, Document, DocumentId, SearchResult};
use crate::state_machine::SearchStateMachine;
use crate::testing::prelude::*;
use std::sync::Arc;

/// Wrapper around SearchStateMachine that captures events for testing
pub struct InstrumentedStateMachine {
    inner: Arc<SearchStateMachine>,
    event_log: EventLog,
}

impl InstrumentedStateMachine {
    /// Create a new instrumented wrapper around a state machine
    pub fn new(inner: Arc<SearchStateMachine>) -> Self {
        Self {
            inner,
            event_log: EventLog::new(),
        }
    }

    /// Create with a pre-existing event log (for continuation testing)
    pub fn with_event_log(inner: Arc<SearchStateMachine>, event_log: EventLog) -> Self {
        Self { inner, event_log }
    }

    /// Get the event log for inspection
    pub fn event_log(&self) -> &EventLog {
        &self.event_log
    }

    /// Index a document with event capture
    pub fn index_document(&self, doc: Document, raft_index: u64) -> Result<OperationId> {
        let doc_id = doc.id;
        let content = doc.content.clone();

        let op_id = self
            .event_log
            .record_invoke(OperationType::Index { doc_id, content });

        let result = self
            .inner
            .apply_parsed_command(raft_index, Command::IndexDocument(doc));

        self.event_log.update_raft_index(op_id, raft_index);

        match result {
            Ok(_) => {
                self.event_log
                    .record_return(op_id, OperationResult::IndexSuccess { doc_id });
                Ok(op_id)
            }
            Err(e) => {
                self.event_log.record_return(
                    op_id,
                    OperationResult::Error {
                        message: e.to_string(),
                    },
                );
                Err(e)
            }
        }
    }

    /// Delete a document with event capture
    pub fn delete_document(&self, doc_id: DocumentId, raft_index: u64) -> Result<OperationId> {
        let op_id = self
            .event_log
            .record_invoke(OperationType::Delete { doc_id });

        let result = self
            .inner
            .apply_parsed_command(raft_index, Command::DeleteDocument(doc_id));

        self.event_log.update_raft_index(op_id, raft_index);

        match result {
            Ok(_) => {
                self.event_log
                    .record_return(op_id, OperationResult::DeleteSuccess { doc_id });
                Ok(op_id)
            }
            Err(e) => {
                self.event_log.record_return(
                    op_id,
                    OperationResult::Error {
                        message: e.to_string(),
                    },
                );
                Err(e)
            }
        }
    }

    /// Get a document with event capture
    pub fn get_document(&self, doc_id: DocumentId) -> Result<(OperationId, Option<Document>)> {
        let op_id = self.event_log.record_invoke(OperationType::Get { doc_id });

        let doc = self.inner.get_document(doc_id);
        let found = doc.is_some();

        self.event_log
            .record_return(op_id, OperationResult::GetSuccess { doc_id, found });

        Ok((op_id, doc))
    }

    /// Keyword search with event capture
    pub fn keyword_search(&self, query: &str, limit: usize) -> (OperationId, Vec<SearchResult>) {
        let op_id = self.event_log.record_invoke(OperationType::Search {
            query: query.to_string(),
        });

        let results = self.inner.keyword_search(query, limit);
        let doc_ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();

        self.event_log
            .record_return(op_id, OperationResult::SearchSuccess { doc_ids });

        (op_id, results)
    }

    /// Vector search with event capture
    pub fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> (OperationId, Vec<SearchResult>) {
        let op_id = self.event_log.record_invoke(OperationType::Search {
            query: format!("vector_search:{}", limit),
        });

        let results = self.inner.vector_search(query_embedding, limit);
        let doc_ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();

        self.event_log
            .record_return(op_id, OperationResult::SearchSuccess { doc_ids });

        (op_id, results)
    }

    /// Hybrid search with event capture
    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        keyword_weight: f32,
    ) -> (OperationId, Vec<SearchResult>) {
        let op_id = self.event_log.record_invoke(OperationType::Search {
            query: format!("hybrid:{}:weight={}", query, keyword_weight),
        });

        let results = self
            .inner
            .hybrid_search(query, query_embedding, limit, keyword_weight);
        let doc_ids: Vec<_> = results.iter().map(|r| r.doc_id).collect();

        self.event_log
            .record_return(op_id, OperationResult::SearchSuccess { doc_ids });

        (op_id, results)
    }

    /// Batch index with event capture
    pub fn batch_index(&self, docs: Vec<Document>, raft_index: u64) -> Result<OperationId> {
        let doc_ids: Vec<_> = docs.iter().map(|d| d.id).collect();
        let count = docs.len();

        let op_id = self.event_log.record_invoke(OperationType::BatchIndex {
            doc_ids: doc_ids.clone(),
        });

        let result = self
            .inner
            .apply_parsed_command(raft_index, Command::BatchIndex(docs));

        self.event_log.update_raft_index(op_id, raft_index);

        match result {
            Ok(_) => {
                self.event_log
                    .record_return(op_id, OperationResult::BatchSuccess { count });
                Ok(op_id)
            }
            Err(e) => {
                self.event_log.record_return(
                    op_id,
                    OperationResult::Error {
                        message: e.to_string(),
                    },
                );
                Err(e)
            }
        }
    }

    /// Batch delete with event capture
    pub fn batch_delete(&self, doc_ids: Vec<DocumentId>, raft_index: u64) -> Result<OperationId> {
        let count = doc_ids.len();

        let op_id = self.event_log.record_invoke(OperationType::BatchDelete {
            doc_ids: doc_ids.clone(),
        });

        let result = self
            .inner
            .apply_parsed_command(raft_index, Command::BatchDelete(doc_ids));

        self.event_log.update_raft_index(op_id, raft_index);

        match result {
            Ok(_) => {
                self.event_log
                    .record_return(op_id, OperationResult::BatchSuccess { count });
                Ok(op_id)
            }
            Err(e) => {
                self.event_log.record_return(
                    op_id,
                    OperationResult::Error {
                        message: e.to_string(),
                    },
                );
                Err(e)
            }
        }
    }

    /// Check invariants against captured events
    pub fn check_invariants(&self, invariants: &[Box<dyn Invariant>]) -> Vec<Violation> {
        check_all_invariants(&self.event_log, invariants)
    }

    /// Check default invariants
    pub fn check_default_invariants(&self) -> Vec<Violation> {
        self.check_invariants(&default_invariants())
    }

    /// Access the inner state machine
    pub fn inner(&self) -> &Arc<SearchStateMachine> {
        &self.inner
    }

    /// Wait for async indexer to catch up
    pub fn wait_for_index(&self, min_index: u64, timeout_ms: u64) -> Result<()> {
        self.inner.wait_for_index(min_index, timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexSettings;
    use crate::models::DocumentMetadata;
    use tempfile::TempDir;

    fn create_test_settings() -> IndexSettings {
        let mut settings = IndexSettings::default();
        settings.vector_dimensions = 3;
        settings.pq_config.num_subspaces = 3;
        settings.pq_config.min_training_vectors = 100;
        settings
    }

    fn create_test_doc(id: u64, content: &str) -> Document {
        Document {
            id,
            content: content.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn test_instrumented_index_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let settings = create_test_settings();
        let machine =
            Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

        let instrumented = InstrumentedStateMachine::new(machine.clone());

        // Index a document
        let doc = create_test_doc(1, "test content");
        instrumented.index_document(doc, 1).unwrap();
        machine.wait_for_index(1, 5000).unwrap();

        // Get the document
        let (_, result) = instrumented.get_document(1).unwrap();
        assert!(result.is_some());

        // Verify event log
        assert_eq!(instrumented.event_log().len(), 2);
        assert_eq!(instrumented.event_log().completed_events().len(), 2);
    }

    #[test]
    fn test_instrumented_delete() {
        let temp_dir = TempDir::new().unwrap();
        let settings = create_test_settings();
        let machine =
            Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

        let instrumented = InstrumentedStateMachine::new(machine.clone());

        // Index then delete
        let doc = create_test_doc(1, "test content");
        instrumented.index_document(doc, 1).unwrap();
        machine.wait_for_index(1, 5000).unwrap();

        instrumented.delete_document(1, 2).unwrap();
        machine.wait_for_index(2, 5000).unwrap();

        // Verify deletion
        let (_, result) = instrumented.get_document(1).unwrap();
        assert!(result.is_none());

        // Verify event log
        assert_eq!(instrumented.event_log().len(), 3);
    }

    #[test]
    fn test_instrumented_invariant_check() {
        let temp_dir = TempDir::new().unwrap();
        let settings = create_test_settings();
        let machine =
            Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

        let instrumented = InstrumentedStateMachine::new(machine.clone());

        // Index documents
        instrumented
            .index_document(create_test_doc(1, "rust programming"), 1)
            .unwrap();
        instrumented
            .index_document(create_test_doc(2, "python coding"), 2)
            .unwrap();
        machine.wait_for_index(2, 5000).unwrap();

        // Get documents
        instrumented.get_document(1).unwrap();
        instrumented.get_document(2).unwrap();

        // Check invariants - should pass
        let violations = instrumented.check_default_invariants();
        assert!(
            violations.is_empty(),
            "Expected no violations: {:?}",
            violations
        );
    }
}
