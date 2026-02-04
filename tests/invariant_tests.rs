//! Integration tests for invariant-based testing infrastructure
//!
//! These tests verify the correctness of the event capture and
//! invariant checking system.

use squidex::config::IndexSettings;
use squidex::models::{Document, DocumentMetadata};
use squidex::state_machine::{InstrumentedStateMachine, SearchStateMachine};
use squidex::testing::prelude::*;
use squidex::testing::{
    DeletedDocumentNotRetrievable, IndexedDocumentRetrievable, MonotonicReads,
    RaftOrderingRespected, SearchReflectsIndex, UniqueDocumentIds,
};
use std::sync::Arc;
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
fn test_basic_invariants() {
    let temp_dir = TempDir::new().unwrap();
    let settings = create_test_settings();
    let machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

    let instrumented = InstrumentedStateMachine::new(machine.clone());

    // Index some documents
    let doc1 = create_test_doc(1, "rust programming language");
    let doc2 = create_test_doc(2, "python coding tutorial");
    let doc3 = create_test_doc(3, "rust systems programming");

    instrumented.index_document(doc1, 1).unwrap();
    instrumented.index_document(doc2, 2).unwrap();
    instrumented.index_document(doc3, 3).unwrap();

    machine.wait_for_index(3, 5000).unwrap();

    // Retrieve documents
    instrumented.get_document(1).unwrap();
    instrumented.get_document(2).unwrap();
    instrumented.get_document(3).unwrap();

    // Search
    instrumented.keyword_search("rust", 10);

    // Delete one
    instrumented.delete_document(2, 4).unwrap();
    machine.wait_for_index(4, 5000).unwrap();

    // Try to get deleted document
    let (_, deleted_doc) = instrumented.get_document(2).unwrap();
    assert!(deleted_doc.is_none());

    // Check invariants
    let invariants: Vec<Box<dyn Invariant>> = vec![
        Box::new(IndexedDocumentRetrievable),
        Box::new(DeletedDocumentNotRetrievable),
        Box::new(UniqueDocumentIds),
        Box::new(SearchReflectsIndex),
        Box::new(RaftOrderingRespected),
    ];

    let violations = instrumented.check_invariants(&invariants);

    if !violations.is_empty() {
        for violation in &violations {
            eprintln!("{}", violation);
        }
        panic!("Invariant violations detected!");
    }

    println!("All invariants passed!");
    println!("Event log: {} events", instrumented.event_log().len());
}

#[test]
fn test_invariant_violation_detection() {
    // This test demonstrates how violations are detected using a simulated buggy scenario
    let log = EventLog::new();

    // Index a document
    let op1 = log.record_invoke(OperationType::Index {
        doc_id: 100,
        content: "test".to_string(),
    });
    log.record_return(op1, OperationResult::IndexSuccess { doc_id: 100 });

    // Delete it
    let op2 = log.record_invoke(OperationType::Delete { doc_id: 100 });
    log.record_return(op2, OperationResult::DeleteSuccess { doc_id: 100 });

    // But somehow it's still retrievable (simulated bug)
    let op3 = log.record_invoke(OperationType::Get { doc_id: 100 });
    log.record_return(
        op3,
        OperationResult::GetSuccess {
            doc_id: 100,
            found: true,
        },
    );

    // Check invariant
    let invariant = DeletedDocumentNotRetrievable;
    let result = invariant.check(&log);

    assert!(result.is_err(), "Should detect violation");

    if let Err(violation) = result {
        println!("Detected violation:");
        println!("{}", violation);
        assert_eq!(violation.invariant, "DeletedDocumentNotRetrievable");
    }
}

#[test]
fn test_event_log_export_import() {
    let temp_dir = TempDir::new().unwrap();
    let settings = create_test_settings();
    let machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

    let instrumented = InstrumentedStateMachine::new(machine.clone());

    // Perform some operations
    let doc = create_test_doc(1, "test document");
    instrumented.index_document(doc, 1).unwrap();
    machine.wait_for_index(1, 5000).unwrap();
    instrumented.get_document(1).unwrap();

    // Export event log
    let json = instrumented.event_log().to_json().unwrap();
    println!("Exported event log:\n{}", json);

    // Import into new log
    let restored = EventLog::from_json(&json).unwrap();
    assert_eq!(restored.len(), instrumented.event_log().len());

    // Check invariants on restored log
    let invariants: Vec<Box<dyn Invariant>> = vec![
        Box::new(IndexedDocumentRetrievable),
        Box::new(UniqueDocumentIds),
    ];

    let violations = check_all_invariants(&restored, &invariants);
    assert!(violations.is_empty(), "Restored log should pass invariants");
}

#[test]
fn test_monotonic_reads_invariant() {
    let log = EventLog::new();

    // Index a document
    let op1 = log.record_invoke(OperationType::Index {
        doc_id: 1,
        content: "test".to_string(),
    });
    log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

    // Get it - found
    let op2 = log.record_invoke(OperationType::Get { doc_id: 1 });
    log.record_return(
        op2,
        OperationResult::GetSuccess {
            doc_id: 1,
            found: true,
        },
    );

    // Get it again - still found (monotonic)
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
fn test_raft_ordering_invariant() {
    let log = EventLog::new();

    // Operations with increasing raft indices
    let op1 = log.record_invoke(OperationType::Index {
        doc_id: 1,
        content: "first".to_string(),
    });
    log.update_raft_index(op1, 1);
    log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

    let op2 = log.record_invoke(OperationType::Index {
        doc_id: 2,
        content: "second".to_string(),
    });
    log.update_raft_index(op2, 2);
    log.record_return(op2, OperationResult::IndexSuccess { doc_id: 2 });

    let op3 = log.record_invoke(OperationType::Delete { doc_id: 1 });
    log.update_raft_index(op3, 3);
    log.record_return(op3, OperationResult::DeleteSuccess { doc_id: 1 });

    let invariant = RaftOrderingRespected;
    assert!(invariant.check(&log).is_ok());
}

#[test]
fn test_search_reflects_index_invariant() {
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
        .index_document(create_test_doc(2, "rust systems"), 2)
        .unwrap();
    machine.wait_for_index(2, 5000).unwrap();

    // Search
    let (_, results) = instrumented.keyword_search("rust", 10);
    assert!(!results.is_empty());

    // Check invariant
    let invariant = SearchReflectsIndex;
    assert!(invariant.check(instrumented.event_log()).is_ok());
}

#[test]
fn test_batch_operations() {
    let temp_dir = TempDir::new().unwrap();
    let settings = create_test_settings();
    let machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf()).unwrap());

    let instrumented = InstrumentedStateMachine::new(machine.clone());

    // Batch index
    let docs = vec![
        create_test_doc(1, "document one"),
        create_test_doc(2, "document two"),
        create_test_doc(3, "document three"),
    ];
    instrumented.batch_index(docs, 1).unwrap();
    machine.wait_for_index(1, 5000).unwrap();

    // Verify all indexed
    for id in 1..=3 {
        let (_, doc) = instrumented.get_document(id).unwrap();
        assert!(doc.is_some(), "Document {} should exist", id);
    }

    // Batch delete
    instrumented.batch_delete(vec![1, 2], 2).unwrap();
    machine.wait_for_index(2, 5000).unwrap();

    // Verify deleted
    let (_, doc1) = instrumented.get_document(1).unwrap();
    let (_, doc2) = instrumented.get_document(2).unwrap();
    let (_, doc3) = instrumented.get_document(3).unwrap();

    assert!(doc1.is_none());
    assert!(doc2.is_none());
    assert!(doc3.is_some());

    // Check all default invariants
    let violations = instrumented.check_default_invariants();
    assert!(violations.is_empty(), "Violations: {:?}", violations);
}

#[test]
fn test_event_log_filtering() {
    let log = EventLog::new();

    // Mix of operations
    let op1 = log.record_invoke(OperationType::Index {
        doc_id: 1,
        content: "test".to_string(),
    });
    log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });

    let op2 = log.record_invoke(OperationType::Search {
        query: "test".to_string(),
    });
    log.record_return(op2, OperationResult::SearchSuccess { doc_ids: vec![1] });

    let op3 = log.record_invoke(OperationType::Delete { doc_id: 1 });
    log.record_return(op3, OperationResult::DeleteSuccess { doc_id: 1 });

    // Filter by type
    assert_eq!(log.index_operations().len(), 1);
    assert_eq!(log.search_operations().len(), 1);
    assert_eq!(log.delete_operations().len(), 1);
}
