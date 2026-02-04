use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tempfile::TempDir;

use squidex::config::IndexSettings;
use squidex::models::{Command, Document, DocumentMetadata};
use squidex::SearchStateMachine;

fn choose_num_subspaces(dims: usize) -> usize {
    let candidates = [24, 16, 12, 8, 6, 4, 3, 2, 1];
    candidates
        .iter()
        .copied()
        .find(|c| dims % c == 0)
        .unwrap_or(1)
}

fn create_doc(id: u64, dims: usize) -> Document {
    Document {
        id,
        content: format!("test document {}", id),
        embedding: vec![0.5; dims],
        metadata: DocumentMetadata::default(),
        created_at: 0,
        updated_at: 0,
    }
}

#[test]
fn test_write_throughput_metrics() {
    let tmp = TempDir::new().unwrap();
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 384;
    settings.pq_config.num_subspaces = choose_num_subspaces(384);

    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

    let num_docs = 1000u64;
    let start = Instant::now();

    for i in 0..num_docs {
        let doc_id = i + 1;
        let doc = create_doc(doc_id, 384);
        machine
            .apply_parsed_command(doc_id, Command::IndexDocument(doc))
            .unwrap();
    }

    machine.wait_for_index(num_docs, 30_000).unwrap();

    let elapsed = start.elapsed();
    let writes_per_sec = num_docs as f64 / elapsed.as_secs_f64();

    println!("\n=== Throughput Metrics ===");
    println!("Total documents: {}", num_docs);
    println!("Total time: {:.2}s", elapsed.as_secs_f64());
    println!("Throughput: {:.2} writes/sec", writes_per_sec);
    println!(
        "Avg latency: {:.2}ms",
        elapsed.as_millis() as f64 / num_docs as f64
    );

    let snapshot = machine.create_snapshot();
    let size_mb = snapshot.len() as f64 / (1024.0 * 1024.0);

    println!("\n=== Memory Metrics ===");
    println!("Snapshot size: {:.2} MB", size_mb);
    println!(
        "Bytes per document: {:.2}",
        snapshot.len() as f64 / num_docs as f64
    );

    let min_writes_per_sec = if cfg!(debug_assertions) { 25.0 } else { 100.0 };
    assert!(
        writes_per_sec > min_writes_per_sec,
        "Throughput too low: {:.2} writes/sec",
        writes_per_sec
    );
    assert!(size_mb < 100.0, "Memory usage too high: {:.2} MB", size_mb);
}

#[test]
fn test_concurrent_write_throughput() {
    let tmp = TempDir::new().unwrap();
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = 384;
    settings.pq_config.num_subspaces = choose_num_subspaces(384);

    let machine = Arc::new(SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap());
    let apply_lock = Arc::new(Mutex::new(()));
    let next_index = Arc::new(AtomicU64::new(1));

    let num_threads = 4u64;
    let docs_per_thread = 250u64;
    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let machine = Arc::clone(&machine);
            let apply_lock = Arc::clone(&apply_lock);
            let next_index = Arc::clone(&next_index);
            std::thread::spawn(move || {
                let offset = thread_id * docs_per_thread;
                for i in 0..docs_per_thread {
                    let doc_id = offset + i + 1;
                    let doc = create_doc(doc_id, 384);
                    let _guard = apply_lock.lock().unwrap();
                    let raft_index = next_index.fetch_add(1, Ordering::SeqCst);
                    machine
                        .apply_parsed_command(raft_index, Command::IndexDocument(doc))
                        .unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let last_index = next_index.load(Ordering::SeqCst) - 1;
    machine.wait_for_index(last_index, 30_000).unwrap();

    let elapsed = start.elapsed();
    let total_docs = num_threads * docs_per_thread;
    let writes_per_sec = total_docs as f64 / elapsed.as_secs_f64();

    println!("\n=== Concurrent Throughput Metrics ===");
    println!("Threads: {}", num_threads);
    println!("Total documents: {}", total_docs);
    println!("Throughput: {:.2} writes/sec", writes_per_sec);

    let min_writes_per_sec = if cfg!(debug_assertions) { 50.0 } else { 200.0 };
    assert!(
        writes_per_sec > min_writes_per_sec,
        "Concurrent throughput too low: {:.2} writes/sec",
        writes_per_sec
    );
}
