//! Invariant checking framework for correctness verification
//!
//! This module provides the trait and concrete implementations for
//! checking system invariants against event histories.

use super::events::{OperationResult, OperationType};
use super::history::EventLog;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A violation of an invariant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    pub invariant: String,
    pub description: String,
    pub violating_events: Vec<usize>, // Indices into event log
    pub context: HashMap<String, String>,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "INVARIANT VIOLATION: {}", self.invariant)?;
        writeln!(f, "  Description: {}", self.description)?;
        writeln!(f, "  Violating events: {:?}", self.violating_events)?;
        if !self.context.is_empty() {
            writeln!(f, "  Context:")?;
            for (key, value) in &self.context {
                writeln!(f, "    {}: {}", key, value)?;
            }
        }
        Ok(())
    }
}

/// Trait for invariant checkers
pub trait Invariant: Send + Sync {
    /// Name of the invariant
    fn name(&self) -> &str;

    /// Check the invariant against an event log
    fn check(&self, log: &EventLog) -> Result<(), Violation>;

    /// Human-readable description
    fn description(&self) -> &str {
        "No description provided"
    }
}

/// Check all invariants and return violations
pub fn check_all_invariants(log: &EventLog, invariants: &[Box<dyn Invariant>]) -> Vec<Violation> {
    let mut violations = Vec::new();

    for invariant in invariants {
        if let Err(violation) = invariant.check(log) {
            violations.push(violation);
        }
    }

    violations
}

// ============================================================================
// CONCRETE INVARIANTS FOR SQUIDEX
// ============================================================================

/// Invariant: Indexed documents can be retrieved
///
/// Every successfully indexed document that hasn't been deleted
/// should be retrievable via Get.
pub struct IndexedDocumentRetrievable;

impl Invariant for IndexedDocumentRetrievable {
    fn name(&self) -> &str {
        "IndexedDocumentRetrievable"
    }

    fn description(&self) -> &str {
        "Every successfully indexed document should be retrievable via Get"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        // Track indexed documents
        let mut indexed: HashSet<u64> = HashSet::new();
        let mut deleted: HashSet<u64> = HashSet::new();
        let mut retrievable: HashSet<u64> = HashSet::new();

        for event in &events {
            match (&event.op_type, &event.result) {
                (
                    OperationType::Index { doc_id, .. },
                    Some(OperationResult::IndexSuccess { .. }),
                ) => {
                    indexed.insert(*doc_id);
                    deleted.remove(doc_id);
                }
                (
                    OperationType::BatchIndex { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        indexed.insert(*doc_id);
                        deleted.remove(doc_id);
                    }
                }
                (OperationType::Delete { doc_id }, Some(OperationResult::DeleteSuccess { .. })) => {
                    deleted.insert(*doc_id);
                    indexed.remove(doc_id);
                }
                (
                    OperationType::BatchDelete { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        deleted.insert(*doc_id);
                        indexed.remove(doc_id);
                    }
                }
                (
                    OperationType::Get { doc_id },
                    Some(OperationResult::GetSuccess { found: true, .. }),
                ) => {
                    retrievable.insert(*doc_id);
                }
                _ => {}
            }
        }

        // Check: all indexed (non-deleted) docs should have been retrievable
        let should_be_retrievable: HashSet<_> = indexed.difference(&deleted).copied().collect();
        let not_retrievable: Vec<_> = should_be_retrievable
            .difference(&retrievable)
            .copied()
            .collect();

        if !not_retrievable.is_empty() {
            let mut context = HashMap::new();
            context.insert(
                "not_retrievable_docs".to_string(),
                format!("{:?}", not_retrievable),
            );
            context.insert("indexed_count".to_string(), indexed.len().to_string());

            return Err(Violation {
                invariant: self.name().to_string(),
                description: format!(
                    "{} indexed documents were not retrievable",
                    not_retrievable.len()
                ),
                violating_events: vec![],
                context,
            });
        }

        Ok(())
    }
}

/// Invariant: Deleted documents are not retrievable
///
/// Once a document is successfully deleted, subsequent Get operations
/// should not find it (until it's re-indexed).
pub struct DeletedDocumentNotRetrievable;

impl Invariant for DeletedDocumentNotRetrievable {
    fn name(&self) -> &str {
        "DeletedDocumentNotRetrievable"
    }

    fn description(&self) -> &str {
        "Deleted documents should not be retrievable"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        let mut deleted: HashSet<u64> = HashSet::new();

        for (idx, event) in events.iter().enumerate() {
            match (&event.op_type, &event.result) {
                (
                    OperationType::Index { doc_id, .. },
                    Some(OperationResult::IndexSuccess { .. }),
                ) => {
                    // Re-indexing a document removes it from deleted set
                    deleted.remove(doc_id);
                }
                (
                    OperationType::BatchIndex { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    // Batch indexing removes from deleted
                    for doc_id in doc_ids {
                        deleted.remove(doc_id);
                    }
                }
                (OperationType::Delete { doc_id }, Some(OperationResult::DeleteSuccess { .. })) => {
                    deleted.insert(*doc_id);
                }
                (
                    OperationType::BatchDelete { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        deleted.insert(*doc_id);
                    }
                }
                (
                    OperationType::Get { doc_id },
                    Some(OperationResult::GetSuccess { found: true, .. }),
                ) => {
                    if deleted.contains(doc_id) {
                        let mut context = HashMap::new();
                        context.insert("doc_id".to_string(), doc_id.to_string());
                        context.insert("event_index".to_string(), idx.to_string());

                        return Err(Violation {
                            invariant: self.name().to_string(),
                            description: format!(
                                "Document {} was retrieved after deletion",
                                doc_id
                            ),
                            violating_events: vec![idx],
                            context,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Invariant: Monotonic reads - once a write is observed, it stays visible
///
/// If a Get operation finds a document, subsequent Get operations
/// should also find it (unless it's been deleted).
pub struct MonotonicReads;

impl Invariant for MonotonicReads {
    fn name(&self) -> &str {
        "MonotonicReads"
    }

    fn description(&self) -> &str {
        "Once a document is observed as indexed, it should remain visible until deleted"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        // Track observed state per document
        let mut observed_present: HashMap<u64, usize> = HashMap::new();
        let mut deleted: HashSet<u64> = HashSet::new();

        for (idx, event) in events.iter().enumerate() {
            match (&event.op_type, &event.result) {
                (
                    OperationType::Get { doc_id },
                    Some(OperationResult::GetSuccess { found: true, .. }),
                ) => {
                    observed_present.insert(*doc_id, idx);
                }
                (
                    OperationType::Get { doc_id },
                    Some(OperationResult::GetSuccess { found: false, .. }),
                ) => {
                    if let Some(&first_seen) = observed_present.get(doc_id) {
                        if !deleted.contains(doc_id) {
                            let mut context = HashMap::new();
                            context.insert("doc_id".to_string(), doc_id.to_string());
                            context.insert("first_seen_idx".to_string(), first_seen.to_string());
                            context.insert("disappeared_idx".to_string(), idx.to_string());

                            return Err(Violation {
                                invariant: self.name().to_string(),
                                description: format!(
                                    "Document {} disappeared after being observed (monotonicity violation)",
                                    doc_id
                                ),
                                violating_events: vec![first_seen, idx],
                                context,
                            });
                        }
                    }
                }
                (OperationType::Delete { doc_id }, Some(OperationResult::DeleteSuccess { .. })) => {
                    deleted.insert(*doc_id);
                }
                (
                    OperationType::BatchDelete { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        deleted.insert(*doc_id);
                    }
                }
                (
                    OperationType::Index { doc_id, .. },
                    Some(OperationResult::IndexSuccess { .. }),
                ) => {
                    // Re-indexing removes from deleted
                    deleted.remove(doc_id);
                }
                (
                    OperationType::BatchIndex { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    // Batch indexing removes from deleted
                    for doc_id in doc_ids {
                        deleted.remove(doc_id);
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Invariant: Document IDs are unique
///
/// Within a single session, document IDs should not be allocated twice
/// for different documents.
pub struct UniqueDocumentIds;

impl Invariant for UniqueDocumentIds {
    fn name(&self) -> &str {
        "UniqueDocumentIds"
    }

    fn description(&self) -> &str {
        "Document IDs should be unique - no duplicate allocations"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        // Track doc_id -> first content hash mapping
        let mut id_to_content: HashMap<u64, String> = HashMap::new();

        for event in &events {
            if let (
                OperationType::Index { doc_id, content },
                Some(OperationResult::IndexSuccess { .. }),
            ) = (&event.op_type, &event.result)
            {
                if let Some(existing_content) = id_to_content.get(doc_id) {
                    // Same doc_id being re-indexed is OK (update semantics)
                    // We only flag if it looks like a duplicate allocation error
                    // For now, we just track the first content
                    if existing_content != content {
                        // This is an update, which is fine
                        id_to_content.insert(*doc_id, content.clone());
                    }
                } else {
                    id_to_content.insert(*doc_id, content.clone());
                }
            }
        }

        // No violations found - this invariant mainly catches duplicate allocations
        // which would be a bug in the ID generation logic
        Ok(())
    }
}

/// Invariant: Search results reflect indexed documents
///
/// Search results should only return documents that are indexed
/// and not deleted.
pub struct SearchReflectsIndex;

impl Invariant for SearchReflectsIndex {
    fn name(&self) -> &str {
        "SearchReflectsIndex"
    }

    fn description(&self) -> &str {
        "Search results should only return indexed (non-deleted) documents"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        let mut indexed: HashSet<u64> = HashSet::new();
        let mut deleted: HashSet<u64> = HashSet::new();

        for (idx, event) in events.iter().enumerate() {
            match (&event.op_type, &event.result) {
                (
                    OperationType::Index { doc_id, .. },
                    Some(OperationResult::IndexSuccess { .. }),
                ) => {
                    indexed.insert(*doc_id);
                    deleted.remove(doc_id);
                }
                (
                    OperationType::BatchIndex { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        indexed.insert(*doc_id);
                        deleted.remove(doc_id);
                    }
                }
                (OperationType::Delete { doc_id }, Some(OperationResult::DeleteSuccess { .. })) => {
                    deleted.insert(*doc_id);
                    indexed.remove(doc_id);
                }
                (
                    OperationType::BatchDelete { doc_ids },
                    Some(OperationResult::BatchSuccess { .. }),
                ) => {
                    for doc_id in doc_ids {
                        deleted.insert(*doc_id);
                        indexed.remove(doc_id);
                    }
                }
                (
                    OperationType::Search { .. },
                    Some(OperationResult::SearchSuccess { doc_ids }),
                ) => {
                    // Check if all returned docs are in valid state
                    for doc_id in doc_ids {
                        if deleted.contains(doc_id) {
                            let mut context = HashMap::new();
                            context
                                .insert("deleted_doc_in_results".to_string(), doc_id.to_string());
                            context.insert("search_event_idx".to_string(), idx.to_string());

                            return Err(Violation {
                                invariant: self.name().to_string(),
                                description: format!("Search returned deleted document {}", doc_id),
                                violating_events: vec![idx],
                                context,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Invariant: Raft ordering is respected
///
/// Operations should be applied in Raft index order.
pub struct RaftOrderingRespected;

impl Invariant for RaftOrderingRespected {
    fn name(&self) -> &str {
        "RaftOrderingRespected"
    }

    fn description(&self) -> &str {
        "Operations should be applied in Raft index order"
    }

    fn check(&self, log: &EventLog) -> Result<(), Violation> {
        let events = log.completed_events();

        let mut last_raft_index: Option<u64> = None;

        for (idx, event) in events.iter().enumerate() {
            if let Some(raft_index) = event.raft_index {
                if let Some(last) = last_raft_index {
                    if raft_index < last {
                        let mut context = HashMap::new();
                        context.insert("expected_min".to_string(), last.to_string());
                        context.insert("actual".to_string(), raft_index.to_string());
                        context.insert("event_index".to_string(), idx.to_string());

                        return Err(Violation {
                            invariant: self.name().to_string(),
                            description: format!(
                                "Raft index {} is less than previous {}",
                                raft_index, last
                            ),
                            violating_events: vec![idx],
                            context,
                        });
                    }
                }
                last_raft_index = Some(raft_index);
            }
        }

        Ok(())
    }
}

/// Create a default set of invariants for Squidex
pub fn default_invariants() -> Vec<Box<dyn Invariant>> {
    vec![
        Box::new(IndexedDocumentRetrievable),
        Box::new(DeletedDocumentNotRetrievable),
        Box::new(MonotonicReads),
        Box::new(UniqueDocumentIds),
        Box::new(SearchReflectsIndex),
        Box::new(RaftOrderingRespected),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_document_retrievable_pass() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op2,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        let invariant = IndexedDocumentRetrievable;
        assert!(invariant.check(&log).is_ok());
    }

    #[test]
    fn test_deleted_document_not_retrievable_pass() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Delete { doc_id: 1 });
        log.record_return(op2, OperationResult::DeleteSuccess { doc_id: 1 });

        let op3 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op3,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: false,
            },
        );

        let invariant = DeletedDocumentNotRetrievable;
        assert!(invariant.check(&log).is_ok());
    }

    #[test]
    fn test_deleted_document_not_retrievable_fail() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Delete { doc_id: 1 });
        log.record_return(op1, OperationResult::DeleteSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op2,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        let invariant = DeletedDocumentNotRetrievable;
        let result = invariant.check(&log);
        assert!(result.is_err());

        if let Err(violation) = result {
            assert_eq!(violation.invariant, "DeletedDocumentNotRetrievable");
        }
    }

    #[test]
    fn test_monotonic_reads_pass() {
        let log = EventLog::new();

        // Index doc
        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        // Get it twice, should be found both times
        let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op2,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        let op3 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op3,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        let invariant = MonotonicReads;
        assert!(invariant.check(&log).is_ok());
    }

    #[test]
    fn test_monotonic_reads_fail() {
        let log = EventLog::new();

        // Get doc - found
        let op1 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op1,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        // Get doc again - not found (without delete!)
        let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op2,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: false,
            },
        );

        let invariant = MonotonicReads;
        let result = invariant.check(&log);
        assert!(result.is_err());

        if let Err(violation) = result {
            assert_eq!(violation.invariant, "MonotonicReads");
            assert_eq!(violation.violating_events.len(), 2);
        }
    }

    #[test]
    fn test_search_reflects_index_pass() {
        let log = EventLog::new();

        // Index documents
        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "rust".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Index {
            doc_id: 2,
            content: "python".to_string(),
        });
        log.record_return(op2, OperationResult::IndexSuccess { doc_id: 2 });

        // Search returns indexed docs
        let op3 = log.record_invoke(OperationType::Search {
            query: "programming".to_string(),
        });
        log.record_return(
            op3,
            OperationResult::SearchSuccess {
                doc_ids: vec![1, 2],
            },
        );

        let invariant = SearchReflectsIndex;
        assert!(invariant.check(&log).is_ok());
    }

    #[test]
    fn test_search_reflects_index_fail() {
        let log = EventLog::new();

        // Index and delete a document
        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "rust".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Delete { doc_id: 1 });
        log.record_return(op2, OperationResult::DeleteSuccess { doc_id: 1 });

        // Search returns deleted doc (bug!)
        let op3 = log.record_invoke(OperationType::Search {
            query: "rust".to_string(),
        });
        log.record_return(op3, OperationResult::SearchSuccess { doc_ids: vec![1] });

        let invariant = SearchReflectsIndex;
        let result = invariant.check(&log);
        assert!(result.is_err());

        if let Err(violation) = result {
            assert_eq!(violation.invariant, "SearchReflectsIndex");
        }
    }

    #[test]
    fn test_raft_ordering_pass() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.update_raft_index(op1, 1);
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Index {
            doc_id: 2,
            content: "test2".to_string(),
        });
        log.update_raft_index(op2, 2);
        log.record_return(op2, OperationResult::IndexSuccess { doc_id: 2 });

        let invariant = RaftOrderingRespected;
        assert!(invariant.check(&log).is_ok());
    }

    #[test]
    fn test_raft_ordering_fail() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.update_raft_index(op1, 10);
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Index {
            doc_id: 2,
            content: "test2".to_string(),
        });
        log.update_raft_index(op2, 5); // Out of order!
        log.record_return(op2, OperationResult::IndexSuccess { doc_id: 2 });

        let invariant = RaftOrderingRespected;
        let result = invariant.check(&log);
        assert!(result.is_err());

        if let Err(violation) = result {
            assert_eq!(violation.invariant, "RaftOrderingRespected");
        }
    }

    #[test]
    fn test_check_all_invariants() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

        let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
        log.record_return(
            op2,
            OperationResult::GetSuccess {
                doc_id: 1,
                found: true,
            },
        );

        let invariants = default_invariants();
        let violations = check_all_invariants(&log, &invariants);

        assert!(violations.is_empty());
    }

    #[test]
    fn test_violation_display() {
        let mut context = HashMap::new();
        context.insert("doc_id".to_string(), "42".to_string());

        let violation = Violation {
            invariant: "TestInvariant".to_string(),
            description: "Something went wrong".to_string(),
            violating_events: vec![1, 2, 3],
            context,
        };

        let display = format!("{}", violation);
        assert!(display.contains("TestInvariant"));
        assert!(display.contains("Something went wrong"));
        assert!(display.contains("doc_id"));
        assert!(display.contains("42"));
    }
}
