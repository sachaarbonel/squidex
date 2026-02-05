//! Integration tests for Query DSL infrastructure
//!
//! Tests end-to-end query execution from parsing through to results.

use squidex::config::IndexSettings;
use squidex::models::{Command, Document, DocumentMetadata};
use squidex::query::QueryParser;
use squidex::state_machine::SearchStateMachine;
use tempfile::TempDir;

fn create_test_settings(dims: usize) -> IndexSettings {
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = dims;
    settings.pq_config.num_subspaces = dims;
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

fn setup_test_index() -> (TempDir, SearchStateMachine) {
    let tmp = TempDir::new().unwrap();
    let dims = 4;
    let settings = create_test_settings(dims);
    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

    // Index test documents
    let docs = vec![
        create_doc(1, "rust programming language systems", dims),
        create_doc(2, "python programming scripting language", dims),
        create_doc(3, "rust systems programming performance", dims),
        create_doc(4, "javascript web programming frontend", dims),
        create_doc(5, "rust cargo package manager", dims),
    ];

    for (i, doc) in docs.into_iter().enumerate() {
        machine
            .apply_parsed_command((i + 1) as u64, Command::IndexDocument(doc))
            .unwrap();
    }

    // Wait for indexing to complete
    machine.wait_for_index(5, 5000).unwrap();

    (tmp, machine)
}

#[test]
fn test_term_query() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "term": { "content": "rust" }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs 1, 3, 5 which contain "rust"
    assert_eq!(results.len(), 3);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&3));
    assert!(doc_ids.contains(&5));
}

#[test]
fn test_match_query_or() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "match": { "content": "rust python" }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs containing "rust" OR "python" (1, 2, 3, 5)
    assert_eq!(results.len(), 4);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&2));
    assert!(doc_ids.contains(&3));
    assert!(doc_ids.contains(&5));
}

#[test]
fn test_match_query_and() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "match": {
            "content": {
                "query": "rust programming",
                "operator": "and"
            }
        }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs containing both "rust" AND "programming" (1, 3)
    assert_eq!(results.len(), 2);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&3));
}

#[test]
fn test_terms_query() {
    let (_tmp, machine) = setup_test_index();

    // Use terms that don't change significantly with stemming
    // "cargo" and "javascript" stay the same after Porter stemming
    let dsl = serde_json::json!({
        "terms": { "content": ["cargo", "javascript"] }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs containing "cargo" (5) or "javascript" (4)
    assert_eq!(results.len(), 2);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&4));
    assert!(doc_ids.contains(&5));
}

#[test]
fn test_bool_query_must() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "bool": {
            "must": [
                { "match": { "content": "programming" } },
                { "match": { "content": "language" } }
            ]
        }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs with both "programming" AND "language" (1, 2)
    assert_eq!(results.len(), 2);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&2));
}

#[test]
fn test_bool_query_should() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "bool": {
            "should": [
                { "term": { "content": "cargo" } },
                { "term": { "content": "frontend" } }
            ]
        }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs with "cargo" (5) or "frontend" (4)
    assert_eq!(results.len(), 2);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&4));
    assert!(doc_ids.contains(&5));
}

#[test]
fn test_bool_query_must_not() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "bool": {
            "must": [
                { "match": { "content": "programming" } }
            ],
            "must_not": [
                { "term": { "content": "python" } }
            ]
        }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs with "programming" but not "python" (1, 3, 4)
    assert_eq!(results.len(), 3);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&3));
    assert!(doc_ids.contains(&4));
    assert!(!doc_ids.contains(&2)); // doc 2 has python
}

#[test]
fn test_match_all_query() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "match_all": {}
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should return all 5 documents
    assert_eq!(results.len(), 5);
}

#[test]
fn test_no_results() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "term": { "content": "nonexistent" }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    assert_eq!(results.len(), 0);
}

#[test]
fn test_top_k_limit() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "match_all": {}
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 2);

    // Should only return top 2
    assert_eq!(results.len(), 2);
}

#[test]
fn test_scoring_order() {
    let (_tmp, machine) = setup_test_index();

    // Doc 3 has "rust" twice ("rust systems programming performance" - but after stemming)
    // Doc 1 has "rust" once
    // Results should be ordered by score
    let dsl = serde_json::json!({
        "match": { "content": "rust" }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    assert!(!results.is_empty());
    // Verify scores are in descending order
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be sorted by score descending"
        );
    }
}

#[test]
fn test_tombstone_filtering() {
    let tmp = TempDir::new().unwrap();
    let dims = 4;
    let settings = create_test_settings(dims);
    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

    // Index documents
    let doc1 = create_doc(1, "rust programming", dims);
    let doc2 = create_doc(2, "rust language", dims);

    machine
        .apply_parsed_command(1, Command::IndexDocument(doc1))
        .unwrap();
    machine
        .apply_parsed_command(2, Command::IndexDocument(doc2))
        .unwrap();
    machine.wait_for_index(2, 5000).unwrap();

    // Delete doc 1
    machine
        .apply_parsed_command(3, Command::DeleteDocument(1))
        .unwrap();
    machine.wait_for_index(3, 5000).unwrap();

    // Search for rust - should only find doc 2
    let dsl = serde_json::json!({
        "term": { "content": "rust" }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id, 2);
}

#[test]
fn test_nested_bool_query() {
    let (_tmp, machine) = setup_test_index();

    let dsl = serde_json::json!({
        "bool": {
            "must": [
                {
                    "bool": {
                        "should": [
                            { "term": { "content": "rust" } },
                            { "term": { "content": "python" } }
                        ]
                    }
                }
            ],
            "must_not": [
                { "term": { "content": "cargo" } }
            ]
        }
    });

    let query = QueryParser::parse(&dsl).unwrap();
    let results = machine.structured_search(query, 10);

    // Should find docs with (rust OR python) AND NOT cargo
    // Doc 1: rust - yes
    // Doc 2: python - yes
    // Doc 3: rust - yes
    // Doc 5: rust + cargo - no (excluded)
    assert_eq!(results.len(), 3);
    let doc_ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
    assert!(doc_ids.contains(&1));
    assert!(doc_ids.contains(&2));
    assert!(doc_ids.contains(&3));
    assert!(!doc_ids.contains(&5));
}
