# Bench Baselines

These baselines are for regression tracking only. Vector and hybrid currently use the brute-force path
because HNSW codebook training is not wired into the bench harness yet.

## Throughput Benchmarks

The throughput suite lives in `benches/throughput_benchmarks.rs` and focuses on ingest rate and
snapshot size trends. Add new dated baselines here once you capture stable numbers on representative
hardware.

### Throughput Baseline

- Date: 2026-01-11
- Command: `cargo bench --bench throughput_benchmarks`
- Notes: Criterion sample-time warnings for batch/perf/memory groups; moderate outliers observed. Values below use the median column from Criterion output.

#### Results (median-ish)

| Benchmark | Time | Throughput |
| --- | --- | --- |
| single_writes/128 | 12.814 µs | 78.042 Kelem/s |
| single_writes/384 | 13.863 µs | 72.132 Kelem/s |
| single_writes/768 | 15.947 µs | 62.708 Kelem/s |
| batch_writes/10 | 13.536 ms | 738.75 elem/s |
| batch_writes/100 | 15.239 ms | 6.5623 Kelem/s |
| batch_writes/1000 | 33.160 ms | 30.157 Kelem/s |
| sustained_writes/5000_docs | 80.647 ms | 61.999 Kelem/s |
| performance_profiles/low-latency | 13.772 ms | 7.2610 Kelem/s |
| performance_profiles/balanced | 14.486 ms | 6.9033 Kelem/s |
| performance_profiles/high-throughput | 15.306 ms | 6.5334 Kelem/s |
| memory_overhead/100 | 16.532 ms | n/a (snapshot 0.30 MB, 3187.69 bytes/doc) |
| memory_overhead/1000 | 37.098 ms | n/a (snapshot 3.04 MB, 3184.37 bytes/doc) |
| memory_overhead/10000 | 188.82 ms | n/a (snapshot 30.37 MB, 3184.94 bytes/doc) |

## Latest Baseline

- Date: 2026-01-11
- Command: `cargo bench`
- Notes: gnuplot not found (plotters backend). Keyword/hybrid/vector improved after tombstone in-memory checks.

### Results (median-ish)

| Benchmark | Docs | Time |
| --- | --- | --- |
| keyword_search | 1,000 | 71.704 µs |
| keyword_search | 5,000 | 335.10 µs |
| keyword_search | 10,000 | 667.43 µs |
| vector_search | 1,000 | 2.7256 ms |
| vector_search | 5,000 | 14.249 ms |
| vector_search | 10,000 | 29.021 ms |
| hybrid_search | 1,000 | 2.9146 ms |
| hybrid_search | 5,000 | 15.207 ms |
| hybrid_search | 10,000 | 30.398 ms |

If you re-run on different hardware or after enabling HNSW training in the bench path, add a new
dated section and keep the old baseline for comparison.

## Baseline (post low-risk perf fixes)

- Date: 2026-01-08
- Command: `cargo bench`
- Notes: gnuplot not found (plotters backend). 10k hybrid hit Criterion's sample time warning. No meaningful perf change (all deltas within noise).

### Results (median-ish)

| Benchmark | Docs | Time |
| --- | --- | --- |
| keyword_search | 1,000 | 96.022 µs |
| keyword_search | 5,000 | 417.08 µs |
| keyword_search | 10,000 | 827.53 µs |
| vector_search | 1,000 | 4.4923 ms |
| vector_search | 5,000 | 23.774 ms |
| vector_search | 10,000 | 48.551 ms |
| hybrid_search | 1,000 | 4.6197 ms |
| hybrid_search | 5,000 | 24.172 ms |
| hybrid_search | 10,000 | 49.504 ms |
