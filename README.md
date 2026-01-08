# Squidex

![Rust](https://img.shields.io/badge/rust-1.91%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)

Squidex is a Rust search engine that combines BM25 keyword search with vector similarity search and exposes an HTTP API. It uses OpenRaft for replicated state and gRPC for Raft RPCs. 

## Features

- **Hybrid search**: keyword + vector queries with configurable weighting.
- **Vector search**: HNSW-based ANN with Product Quantization for memory efficiency.
- **Distributed core**: OpenRaft replication with gRPC transport.
- **Operational basics**: health checks and Prometheus metrics.
- **Single-binary deploy**: no JVM or external services required.

## Table of Contents

- [Squidex](#squidex)
  - [Features (SPEC v2.1 highlights)](#features-spec-v21-highlights)
  - [Table of Contents](#table-of-contents)
  - [What is implemented](#what-is-implemented)
  - [Architecture](#architecture)
  - [Requirements](#requirements)
  - [Build](#build)
  - [Run (single node)](#run-single-node)
  - [HTTP API](#http-api)
    - [Index a document](#index-a-document)
    - [Get a document](#get-a-document)
    - [Batch index documents](#batch-index-documents)
    - [Delete a document](#delete-a-document)
    - [Search (keyword)](#search-keyword)
    - [Search (vector)](#search-vector)
    - [Search (hybrid)](#search-hybrid)
    - [Cluster status](#cluster-status)
  - [Health and metrics](#health-and-metrics)
  - [Configuration](#configuration)
    - [CLI flags and environment variables](#cli-flags-and-environment-variables)
    - [Performance profiles](#performance-profiles)
    - [Vector dimension constraint](#vector-dimension-constraint)
  - [Use cases](#use-cases)
  - [Known limitations (verified in code)](#known-limitations-verified-in-code)
  - [Troubleshooting](#troubleshooting)
    - [Build fails with `protoc` not found](#build-fails-with-protoc-not-found)
    - [Process panics when changing `--vector-dimensions`](#process-panics-when-changing---vector-dimensions)
    - [Indexing returns `not_leader`](#indexing-returns-not_leader)
    - [Vector search returns no results](#vector-search-returns-no-results)
  - [Development](#development)
  - [Contributing](#contributing)
  - [License](#license)

## What is implemented

- **Keyword search**: BM25 scoring with \(K1 = 1.2\) and \(B = 0.75\) (see `src/state_machine/scoring.rs`).
- **Vector search**: Cosine, Euclidean, and dot product scoring with an HNSW index and Product Quantization backed vector store (see `src/state_machine/machine.rs` and `src/vector/`).
- **Hybrid search**: Combines keyword and vector scores with a configurable keyword weight (see `src/state_machine/machine.rs`).
- **HTTP API**: Document indexing, retrieval, search, cluster status, health, and Prometheus metrics (see `src/api/router.rs`).

## Architecture

At a high level:

- **Consensus**: OpenRaft for replicated state and membership.
- **API**: HTTP via Axum; Raft RPCs via gRPC/tonic.
- **Search**: BM25 keyword search + vector search, fused for hybrid ranking.
- **Storage**: append-only logs + RocksDB metadata, with PQ-compressed vectors.

Directory structure (partial):

```
src/
├── api/          # HTTP API handlers and routing
├── consensus/    # Raft node and cluster management
├── state_machine/# Search engine state machine
├── vector/       # Vector storage (PQ + HNSW)
├── metrics/      # Prometheus metrics
└── config.rs     # Configuration and profiles
```

## Requirements

- **Rust**: 1.91+ (see `Cargo.toml` `rust-version`).
- **Protocol Buffers compiler**: `protoc` is required to build because `build.rs` compiles `proto/raft.proto` via `tonic-build`.

Install `protoc`:

```bash
# macOS
brew install protobuf

# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y protobuf-compiler

protoc --version
```

## Build

```bash
git clone https://github.com/sachaarbonel/squidex.git
cd squidex
cargo build --release
```

## Run (single node)

Writes are accepted only on the leader. A single node becomes leader when started with `--is-initial-leader`.

This example uses `--vector-dimensions 24` so the HTTP examples can use short embeddings. The value must be divisible by the default PQ subspace count (24).

```bash
cargo run --release -- \
  --node-id 1 \
  --bind-addr 127.0.0.1:5001 \
  --http-port 8080 \
  --data-dir ./data \
  --is-initial-leader \
  --profile balanced \
  --vector-dimensions 24
```

## HTTP API

All API routes (except `/health` and `/metrics`) are under the `/v1` prefix.

### Index a document

The server assigns a numeric document ID and returns it in the response.

```bash
curl -sS -X POST "http://localhost:8080/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Rust is a systems programming language focused on safety and concurrency.",
    "embedding": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
    "metadata": {
      "title": "About Rust",
      "source": "example",
      "tags": ["rust", "systems"],
      "custom": {"author": "squidex"}
    }
  }'
```

### Get a document

```bash
curl -sS "http://localhost:8080/v1/documents/1"
```

### Batch index documents

```bash
curl -sS -X POST "http://localhost:8080/v1/batch/index" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "content": "Document 1",
      "embedding": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
      "metadata": {"title": "Doc 1"}
    },
    {
      "content": "Document 2",
      "embedding": [0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
      "metadata": {"title": "Doc 2"}
    }
  ]'
```

### Delete a document

```bash
curl -sS -X DELETE "http://localhost:8080/v1/documents/1"
```

### Search (keyword)

```bash
curl -sS -X POST "http://localhost:8080/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "keyword",
    "query": "safety concurrency",
    "top_k": 10
  }'
```

### Search (vector)

```bash
curl -sS -X POST "http://localhost:8080/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "vector",
    "embedding": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
    "top_k": 10
  }'
```

### Search (hybrid)

```bash
curl -sS -X POST "http://localhost:8080/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "hybrid",
    "query": "rust safety",
    "embedding": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24],
    "keyword_weight": 0.5,
    "top_k": 10
  }'
```

### Cluster status

```bash
curl -sS "http://localhost:8080/v1/cluster/status"
```

## Health and metrics

```bash
curl -sS "http://localhost:8080/health"
curl -sS "http://localhost:8080/metrics"
```

The Prometheus registry includes (at least) these metrics (see `src/metrics/mod.rs`):

- `squidex_documents_indexed_total`
- `squidex_documents_deleted_total`
- `squidex_documents_updated_total`
- `squidex_searches_total`
- `squidex_search_errors_total`
- `squidex_total_documents`
- `squidex_index_size_bytes`
- `squidex_cluster_leader`
- `squidex_cluster_size`
- `squidex_index_latency_seconds`
- `squidex_search_latency_seconds`
- `squidex_batch_size`

Note: the API handlers do not currently record metrics; only the endpoint and registry are implemented.

## Configuration

### CLI flags and environment variables

The CLI is implemented in `bin/squidex.rs` with `clap`. These flags can also be set via environment variables:

| Environment variable | Flag | Default |
|---|---|---|
| `SQUIDEX_NODE_ID` | `--node-id` | required |
| `SQUIDEX_BIND_ADDR` | `--bind-addr` | `127.0.0.1:5001` |
| `SQUIDEX_PEERS` | `--peers` | empty |
| `SQUIDEX_DATA_DIR` | `--data-dir` | `./data` |
| `SQUIDEX_INITIAL_LEADER` | `--is-initial-leader` | `false` |
| `SQUIDEX_PROFILE` | `--profile` | `balanced` |
| `SQUIDEX_VECTOR_DIM` | `--vector-dimensions` | `384` |
| `SQUIDEX_HTTP_PORT` | `--http-port` | `8080` |

### Performance profiles

Performance profiles are applied to the Raft node WAL settings (see `src/config.rs`):

| Profile | WAL batch size | WAL flush interval (ms) |
|---|---:|---:|
| `low-latency` | 10 | 10 |
| `balanced` | 100 | 50 |
| `high-throughput` | 1000 | 200 |
| `durable` | 1 | 0 |

### Vector dimension constraint

The vector store uses Product Quantization by default. The configured vector dimensions must be divisible by the PQ subspace count (default is 24 in `src/config.rs`). If you change `--vector-dimensions`, keep it a multiple of 24.

## Use cases

- **Semantic search**: combine keyword and embedding-based retrieval.
- **Document search**: full-text + vector similarity in one index.
- **RAG pipelines**: retrieval layer for LLM applications.
- **Distributed search**: replicated nodes for availability.

## Known limitations (verified in code)

- **Filters are not applied by the HTTP API**: `SearchRequestApi` includes a `filters` field, but `src/api/handlers.rs` does not use it when executing searches.
- **Multi-node membership management is not exposed over HTTP**: `SquidexNode` has `add_node` and `remove_node` methods (see `src/consensus/node.rs`), but there are no HTTP routes for them in `src/api/router.rs`.
- **Raft RPC TLS is not configured**: The gRPC server is started with `tonic::transport::Server::builder().serve(...)` without TLS configuration (see `bin/squidex.rs`).
- **Authentication and authorization are not implemented**: the HTTP API does not enforce auth.

## Troubleshooting

### Build fails with `protoc` not found

- Install the Protocol Buffers compiler (see Requirements).

### Process panics when changing `--vector-dimensions`

- The PQ store asserts that `vector_dimensions % num_subspaces == 0`. The default subspace count is 24. Use a multiple of 24 (24, 48, 96, 384, 768).

### Indexing returns `not_leader`

- Write endpoints (`POST /v1/documents`, `DELETE /v1/documents/:id`, `POST /v1/batch/index`) require the node to be leader. Start a single node with `--is-initial-leader`.

### Vector search returns no results

- If the query embedding length does not match the configured `--vector-dimensions`, `vector_search` returns an empty result set (see `src/state_machine/machine.rs`).

## Development

```bash
cargo test
cargo fmt --check
cargo clippy -- -D warnings
```

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the checks (see Development)
5. Commit your changes (`git commit -m "Add amazing feature"`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

The crate is licensed as `MIT OR Apache-2.0` (see `Cargo.toml`).
