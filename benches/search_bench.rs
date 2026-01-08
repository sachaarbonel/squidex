use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::TempDir;

use squidex::config::IndexSettings;
use squidex::models::{Command, Document, DocumentMetadata};
use squidex::SearchStateMachine;

struct BenchEnv {
    _tmp: TempDir,
    machine: SearchStateMachine,
}

fn create_settings(dim: usize, min_training_vectors: usize) -> IndexSettings {
    let mut settings = IndexSettings::default();
    settings.vector_dimensions = dim;
    settings.pq_config.num_subspaces = dim; // 1 dim per subspace for benches
    settings.pq_config.min_training_vectors = min_training_vectors;
    settings
}

fn make_embedding(id: u64, dims: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dims);
    for i in 0..dims {
        let val = ((id as usize + i) % 10) as f32 / 10.0;
        v.push(val);
    }
    v
}

fn build_env(doc_count: usize, dims: usize) -> BenchEnv {
    let settings = create_settings(dims, doc_count + 1);
    let tmp = TempDir::new().unwrap();
    let machine = SearchStateMachine::new(settings, tmp.path().to_path_buf()).unwrap();

    for i in 1..=doc_count as u64 {
        let content = format!("rust programming language doc {}", i);
        let doc = Document {
            id: i,
            content,
            embedding: make_embedding(i, dims),
            metadata: DocumentMetadata::default(),
            created_at: 0,
            updated_at: 0,
        };
        machine
            .apply_parsed_command(i, Command::IndexDocument(doc))
            .unwrap();
    }
    machine.wait_for_index(doc_count as u64, 30_000).unwrap();

    BenchEnv { _tmp: tmp, machine }
}

fn bench_keyword_search(c: &mut Criterion) {
    let dims = 24;
    let counts = [1_000usize, 5_000, 10_000];
    let mut envs: Vec<(usize, BenchEnv)> = Vec::new();
    for &count in &counts {
        envs.push((count, build_env(count, dims)));
    }

    let mut group = c.benchmark_group("keyword_search");
    for (count, env) in envs.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), env, |b, env| {
            b.iter(|| {
                black_box(env.machine.keyword_search("rust programming", 10));
            });
        });
    }
    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let dims = 24;
    let counts = [1_000usize, 5_000, 10_000];
    let mut envs: Vec<(usize, BenchEnv)> = Vec::new();
    for &count in &counts {
        envs.push((count, build_env(count, dims)));
    }

    let mut group = c.benchmark_group("vector_search");
    let query = vec![0.2f32; dims];
    for (count, env) in envs.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), env, |b, env| {
            b.iter(|| {
                black_box(env.machine.vector_search(&query, 10));
            });
        });
    }
    group.finish();
}

fn bench_hybrid_search(c: &mut Criterion) {
    let dims = 24;
    let counts = [1_000usize, 5_000, 10_000];
    let mut envs: Vec<(usize, BenchEnv)> = Vec::new();
    for &count in &counts {
        envs.push((count, build_env(count, dims)));
    }

    let mut group = c.benchmark_group("hybrid_search");
    let query = vec![0.2f32; dims];
    for (count, env) in envs.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(count), env, |b, env| {
            b.iter(|| {
                black_box(
                    env.machine
                        .hybrid_search("rust programming", &query, 10, 0.5),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_keyword_search,
    bench_vector_search,
    bench_hybrid_search
);
criterion_main!(benches);
