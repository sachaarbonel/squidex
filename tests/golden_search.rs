use tempfile::TempDir;

use squidex::config::IndexSettings;
use squidex::models::{Command, Document, DocumentMetadata};
use squidex::SearchStateMachine;

fn create_settings(dim: usize) -> IndexSettings {
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = dim;
    settings.pq_config.num_subspaces = dim; // 1 dim per subspace for tests
    settings.pq_config.min_training_vectors = 10_000;
    settings
}

fn create_doc(id: u64, content: &str, embedding: Vec<f32>) -> Document {
    Document {
        id,
        content: content.to_string(),
        embedding,
        metadata: DocumentMetadata::default(),
        created_at: 0,
        updated_at: 0,
    }
}

fn setup_machine() -> (TempDir, SearchStateMachine) {
    let dims = 3;
    let settings = create_settings(dims);
    let tmp = TempDir::new().unwrap();
    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();
    (tmp, machine)
}

fn index_fixture(machine: &SearchStateMachine) {
    let docs = vec![
        create_doc(1, "rust rust rust programming", vec![1.0, 0.0, 0.0]),
        create_doc(2, "rust programming language", vec![0.9, 0.1, 0.0]),
        create_doc(3, "python programming language", vec![0.0, 1.0, 0.0]),
        create_doc(4, "garden tools", vec![0.0, 0.0, 1.0]),
    ];

    for (i, doc) in docs.into_iter().enumerate() {
        machine
            .apply_parsed_command((i + 1) as u64, Command::IndexDocument(doc))
            .unwrap();
    }
    machine.wait_for_index(4, 10_000).unwrap();
}

#[test]
fn golden_keyword_results_include_expected_docs() {
    let (_tmp, machine) = setup_machine();
    index_fixture(&machine);

    let results = machine.keyword_search("rust programming", 2);
    let mut ids: Vec<u64> = results.into_iter().map(|r| r.doc_id).collect();
    ids.sort_unstable();

    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn golden_vector_results_rank_expected_docs() {
    let (_tmp, machine) = setup_machine();
    index_fixture(&machine);

    let results = machine.vector_search(&[1.0, 0.0, 0.0], 2);
    let ids: Vec<u64> = results.into_iter().map(|r| r.doc_id).collect();

    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn golden_hybrid_weight_extremes_match_single_modes() {
    let (_tmp, machine) = setup_machine();
    index_fixture(&machine);

    let keyword_top = machine.keyword_search("rust programming", 1);
    let vector_top = machine.vector_search(&[1.0, 0.0, 0.0], 1);

    let hybrid_keyword = machine.hybrid_search("rust programming", &[1.0, 0.0, 0.0], 1, 1.0);
    let hybrid_vector = machine.hybrid_search("rust programming", &[1.0, 0.0, 0.0], 1, 0.0);

    assert_eq!(hybrid_keyword[0].doc_id, keyword_top[0].doc_id);
    assert_eq!(hybrid_vector[0].doc_id, vector_top[0].doc_id);
}
