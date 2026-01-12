# Bench Baselines

These baselines are for regression tracking only. Vector and hybrid currently use the brute-force path
because HNSW codebook training is not wired into the bench harness yet.

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
