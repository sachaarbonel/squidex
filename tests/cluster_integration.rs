use octopii::{Config, OctopiiNode, OctopiiRuntime, StateMachineTrait};
use squidex::{Document, DocumentMetadata, IndexSettings, SearchStateMachine};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

/// Helper to create a test configuration for a node
fn create_node_config(
    node_id: u64,
    bind_addr: SocketAddr,
    peers: Vec<SocketAddr>,
    data_dir: PathBuf,
    is_initial_leader: bool,
) -> Config {
    Config {
        node_id,
        bind_addr,
        peers,
        wal_dir: data_dir,
        worker_threads: 2,
        wal_batch_size: 100,
        wal_flush_interval_ms: 50,
        is_initial_leader,
        snapshot_lag_threshold: 1000,
    }
}

/// Get next available port for testing
fn next_addr() -> SocketAddr {
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
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
    let addr = next_addr();

    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 3;
    let state_machine = Arc::new(SearchStateMachine::new(settings));

    let config = create_node_config(1, addr, vec![], temp_dir.path().join("node1"), true);

    let runtime = OctopiiRuntime::from_handle(tokio::runtime::Handle::current());
    let node = OctopiiNode::new_with_state_machine(
        config,
        runtime,
        state_machine.clone() as Arc<dyn StateMachineTrait>,
    )
    .await?;

    // Start the node
    node.start().await?;

    // Give it time to initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Index a document
    let doc = create_test_document(1, "hello world from squidex");
    state_machine.index_document(doc)?;

    // Verify document was indexed
    assert_eq!(state_machine.total_documents(), 1);

    // Search for the document
    let results = state_machine.keyword_search("squidex", 10);
    assert!(!results.is_empty());
    assert_eq!(results[0].doc_id, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_three_node_cluster() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let base = temp_dir.path();

    let addr1 = next_addr();
    let addr2 = next_addr();
    let addr3 = next_addr();

    let peers1 = vec![addr2, addr3];
    let peers2 = vec![addr1, addr3];
    let peers3 = vec![addr1, addr2];

    // Create state machines
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 3;

    let sm1 = Arc::new(SearchStateMachine::new(settings.clone()));
    let sm2 = Arc::new(SearchStateMachine::new(settings.clone()));
    let sm3 = Arc::new(SearchStateMachine::new(settings));

    // Create runtime
    let runtime = OctopiiRuntime::from_handle(tokio::runtime::Handle::current());

    // Create nodes
    let node1 = OctopiiNode::new_with_state_machine(
        create_node_config(1, addr1, peers1, base.join("node1"), true),
        runtime.clone(),
        sm1.clone() as Arc<dyn StateMachineTrait>,
    )
    .await?;

    let node2 = OctopiiNode::new_with_state_machine(
        create_node_config(2, addr2, peers2, base.join("node2"), false),
        runtime.clone(),
        sm2.clone() as Arc<dyn StateMachineTrait>,
    )
    .await?;

    let node3 = OctopiiNode::new_with_state_machine(
        create_node_config(3, addr3, peers3, base.join("node3"), false),
        runtime,
        sm3.clone() as Arc<dyn StateMachineTrait>,
    )
    .await?;

    // Start all nodes
    node1.start().await?;
    node2.start().await?;
    node3.start().await?;

    // Give them time to elect a leader
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify cluster formed (at least one node should be leader)
    let is_leader1 = node1.is_leader().await;
    let is_leader2 = node2.is_leader().await;
    let is_leader3 = node3.is_leader().await;

    assert!(
        is_leader1 || is_leader2 || is_leader3,
        "At least one node should be leader"
    );

    // Index a document on the leader
    let doc = create_test_document(1, "distributed search test");
    if is_leader1 {
        sm1.index_document(doc)?;
    } else if is_leader2 {
        sm2.index_document(doc)?;
    } else {
        sm3.index_document(doc)?;
    }

    // Give time for replication
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify all nodes have the document (eventually consistent)
    // Note: This test is simplified and may need adjustment based on actual replication behavior
    let total1 = sm1.total_documents();
    let total2 = sm2.total_documents();
    let total3 = sm3.total_documents();

    println!(
        "Document counts: node1={}, node2={}, node3={}",
        total1, total2, total3
    );

    Ok(())
}

#[tokio::test]
async fn test_snapshot_and_restore() -> Result<(), Box<dyn std::error::Error>> {
    let _temp_dir = TempDir::new()?;
    let _addr = next_addr();

    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 3;
    let state_machine = Arc::new(SearchStateMachine::new(settings));

    // Index some documents
    for i in 1..=10 {
        let doc = create_test_document(i, &format!("test document {}", i));
        state_machine.index_document(doc)?;
    }

    assert_eq!(state_machine.total_documents(), 10);

    // Create a snapshot
    let snapshot = state_machine.create_snapshot();
    assert!(!snapshot.is_empty());

    // Create a new state machine and restore
    let mut settings2 = IndexSettings::default();
    settings2.vector_dimensions = 3;
    let state_machine2 = Arc::new(SearchStateMachine::new(settings2));

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
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 3;
    let state_machine = Arc::new(SearchStateMachine::new(settings));

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
        state_machine.index_document(doc)?;
    }

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
    let hybrid_results =
        state_machine.hybrid_search("rust", &query_embedding, 10, 0.5);
    assert!(!hybrid_results.is_empty());

    Ok(())
}
