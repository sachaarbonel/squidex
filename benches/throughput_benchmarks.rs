use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use tempfile::TempDir;

use squidex::config::{IndexSettings, PerformanceProfile};
use squidex::models::{Command, Document, DocumentMetadata};
use squidex::SearchStateMachine;

fn choose_num_subspaces(dims: usize) -> usize {
    // Keep PQ configuration valid for the given dimensions.
    let candidates = [24, 16, 12, 8, 6, 4, 3, 2, 1];
    candidates
        .iter()
        .copied()
        .find(|c| dims % c == 0)
        .unwrap_or(1)
}

fn create_test_doc(id: u64, dims: usize) -> Document {
    Document {
        id,
        content: format!("benchmark document {}", id),
        embedding: vec![0.5; dims],
        metadata: DocumentMetadata::default(),
        created_at: 0,
        updated_at: 0,
    }
}

fn setup_machine(dims: usize, num_subspaces: usize) -> (SearchStateMachine, TempDir) {
    let tmp = TempDir::new().unwrap();
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = dims;
    settings.pq_config.num_subspaces = num_subspaces;
    settings.pq_config.min_training_vectors = 10;

    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();
    (machine, tmp)
}

// Benchmark: Single document write throughput
fn bench_single_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_writes");
    group.measurement_time(Duration::from_secs(10));

    for &dims in [128usize, 384, 768].iter() {
        let subspaces = choose_num_subspaces(dims);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(dims), &dims, |b, &dims| {
            let (machine, _tmp) = setup_machine(dims, subspaces);
            let mut doc_id = 1u64;

            b.iter(|| {
                let doc = create_test_doc(doc_id, dims);
                machine
                    .apply_parsed_command(doc_id, Command::IndexDocument(doc))
                    .unwrap();
                doc_id += 1;
                black_box(doc_id)
            });
        });
    }
    group.finish();
}

// Benchmark: Batch write throughput
fn bench_batch_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_writes");
    group.measurement_time(Duration::from_secs(15));

    let dims = 384;
    let subspaces = choose_num_subspaces(dims);

    for &batch_size in [10usize, 100, 1000].iter() {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_batched(
                    || setup_machine(dims, subspaces),
                    |(machine, _tmp)| {
                        for i in 0..batch_size {
                            let doc_id = (i + 1) as u64;
                            let doc = create_test_doc(doc_id, dims);
                            machine
                                .apply_parsed_command(doc_id, Command::IndexDocument(doc))
                                .unwrap();
                        }
                        machine.wait_for_index(batch_size as u64, 5_000).unwrap();
                        black_box(batch_size)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

// Benchmark: Sustained write throughput over time
fn bench_sustained_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sustained_writes");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    let dims = 384;
    let subspaces = choose_num_subspaces(dims);
    let total_docs = 5_000usize;

    group.throughput(Throughput::Elements(total_docs as u64));
    group.bench_function("5000_docs", |b| {
        b.iter_batched(
            || setup_machine(dims, subspaces),
            |(machine, _tmp)| {
                let start = std::time::Instant::now();

                for i in 0..total_docs {
                    let doc_id = (i + 1) as u64;
                    let doc = create_test_doc(doc_id, dims);
                    machine
                        .apply_parsed_command(doc_id, Command::IndexDocument(doc))
                        .unwrap();
                }

                machine.wait_for_index(total_docs as u64, 30_000).unwrap();

                let elapsed = start.elapsed();
                let writes_per_sec = total_docs as f64 / elapsed.as_secs_f64();

                eprintln!("Sustained throughput: {:.2} writes/sec", writes_per_sec);
                black_box(writes_per_sec)
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

// Benchmark: Write throughput with different performance profiles
fn bench_performance_profiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_profiles");
    group.measurement_time(Duration::from_secs(10));

    let dims = 384;
    let num_docs = 100usize;

    let profiles = [
        (PerformanceProfile::LowLatency, 48usize, "low-latency"),
        (PerformanceProfile::Balanced, 24usize, "balanced"),
        (
            PerformanceProfile::HighThroughput,
            12usize,
            "high-throughput",
        ),
    ];

    for (profile, subspaces, label) in profiles {
        let subspaces = if dims % subspaces == 0 {
            subspaces
        } else {
            choose_num_subspaces(dims)
        };
        group.throughput(Throughput::Elements(num_docs as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &profile, |b, _| {
            b.iter_batched(
                || setup_machine(dims, subspaces),
                |(machine, _tmp)| {
                    for i in 0..num_docs {
                        let doc_id = (i + 1) as u64;
                        let doc = create_test_doc(doc_id, dims);
                        machine
                            .apply_parsed_command(doc_id, Command::IndexDocument(doc))
                            .unwrap();
                    }
                    machine.wait_for_index(num_docs as u64, 5_000).unwrap();
                    black_box(num_docs)
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// Benchmark: Memory usage during sustained writes
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");
    group.measurement_time(Duration::from_secs(15));

    let dims = 384;
    let subspaces = choose_num_subspaces(dims);

    for &num_docs in [100usize, 1000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_docs),
            &num_docs,
            |b, &num_docs| {
                b.iter_batched(
                    || setup_machine(dims, subspaces),
                    |(machine, _tmp)| {
                        for i in 0..num_docs {
                            let doc_id = (i + 1) as u64;
                            let doc = create_test_doc(doc_id, dims);
                            machine
                                .apply_parsed_command(doc_id, Command::IndexDocument(doc))
                                .unwrap();
                        }

                        machine.wait_for_index(num_docs as u64, 30_000).unwrap();

                        let snapshot = machine.create_snapshot();
                        let size_mb = snapshot.len() as f64 / (1024.0 * 1024.0);
                        let bytes_per_doc = snapshot.len() as f64 / num_docs as f64;

                        eprintln!(
                            "{} docs: {:.2} MB total, {:.2} bytes/doc",
                            num_docs, size_mb, bytes_per_doc
                        );

                        black_box((snapshot.len(), num_docs))
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_writes,
    bench_batch_writes,
    bench_sustained_writes,
    bench_performance_profiles,
    bench_memory_overhead
);
criterion_main!(benches);
