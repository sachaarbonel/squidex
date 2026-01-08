use squidex::{
    Command, Document, DocumentMetadata, IndexSettings, NodeConfig, SearchStateMachine, SquidexNode,
};
use std::sync::Arc;
use tempfile::TempDir;

/// Create test settings with proper PQ configuration for 3-dimensional vectors
fn create_test_settings() -> IndexSettings {
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 3;
    // PQ requires dimensions to be divisible by num_subspaces
    settings.pq_config.num_subspaces = 3;
    settings.pq_config.min_training_vectors = 100; // Don't trigger auto-training in tests
    settings
}

/// Create a test document
fn create_test_document(id: u64, content: &str) -> Document {
    Document {
        id,
        content: content.to_string(),
        embedding: vec![0.1, 0.2, 0.3],
        metadata: DocumentMetadata::default(),
        created_at: 0,
        updated_at: 0,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_single_node_operations() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;

    let settings = create_test_settings();
    let state_machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf())?);

    let config = NodeConfig::new(
        1,
        "127.0.0.1:15001".to_string(),
        vec![],
        temp_dir.path().to_path_buf(),
        true,
    );

    let node = SquidexNode::new(1, config, state_machine.clone()).await?;

    // Start the node
    node.start().await?;

    // Give it time to initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Index a document directly (bypassing Raft for simplicity in this test)
    let doc = create_test_document(1, "hello world from squidex");
    state_machine.apply_parsed_command(1, Command::IndexDocument(doc))?;
    state_machine.wait_for_index(1, 1000)?;

    // Verify document was indexed
    assert_eq!(state_machine.total_documents(), 1);

    // Search for the document
    let results = state_machine.keyword_search("squidex", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 1);

    Ok(())
}

#[tokio::test]
async fn test_snapshot_and_restore() -> Result<(), Box<dyn std::error::Error>> {
    let _temp_dir = TempDir::new()?;

    let temp_dir = TempDir::new()?;
    let settings = create_test_settings();
    let state_machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf())?);

    // Index some documents
    for i in 1..=10 {
        let doc = create_test_document(i, &format!("test document {}", i));
        state_machine.apply_parsed_command(i as u64, Command::IndexDocument(doc))?;
    }

    assert_eq!(state_machine.total_documents(), 10);
    state_machine.wait_for_index(10, 5_000)?;

    // Create a snapshot
    let snapshot = state_machine.create_snapshot();
    assert!(!snapshot.is_empty());

    // Create a new state machine and restore
    let temp_dir2 = TempDir::new()?;
    let settings2 = create_test_settings();
    let state_machine2 =
        Arc::new(SearchStateMachine::new(settings2, temp_dir2.path().to_path_buf())?);

    state_machine2.restore_snapshot(&snapshot)?;

    // Verify restored state
    assert_eq!(state_machine2.total_documents(), 10);

    // Verify documents can be searched
    let results = state_machine2.keyword_search("test document", 10);
    assert_eq!(results.len(), 10);

    Ok(())
}

#[tokio::test]
async fn test_vector_and_hybrid_search() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let settings = create_test_settings();
    let state_machine =
        Arc::new(SearchStateMachine::new(settings, temp_dir.path().to_path_buf())?);

    // Index documents with different embeddings
    let docs = vec![
        (1, "rust programming", vec![1.0, 0.0, 0.0]),
        (2, "python coding", vec![0.0, 1.0, 0.0]),
        (3, "rust development", vec![0.9, 0.1, 0.0]),
    ];

    for (id, content, embedding) in docs {
        let doc = Document {
            id,
            content: content.to_string(),
            embedding,
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        };
        state_machine.apply_parsed_command(id, Command::IndexDocument(doc))?;
    }
    state_machine.wait_for_index(3, 5_000)?;

    // Test keyword search
    let keyword_results = state_machine.keyword_search("rust", 10);
    assert_eq!(keyword_results.len(), 2); // Should find doc 1 and 3

    // Test vector search
    let query_embedding = vec![1.0, 0.0, 0.0];
    let vector_results = state_machine.vector_search(&query_embedding, 10);
    assert_eq!(vector_results.len(), 3);
    // Doc 1 should be most similar
    assert_eq!(vector_results[0].doc_id, 1);

    // Test hybrid search
    let hybrid_results = state_machine.hybrid_search("rust", &query_embedding, 10, 0.5);
    assert!(!hybrid_results.is_empty());

    Ok(())
}
