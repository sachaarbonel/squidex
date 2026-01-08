//! Vector storage module with Product Quantization and HNSW for memory-efficient storage and search.
//!
//! This module provides a production-grade vector storage system using:
//! - Product Quantization (PQ) for 64x memory compression
//! - HNSW (Hierarchical Navigable Small World) for O(log N) approximate nearest neighbor search

mod codebook;
mod hnsw;
mod quantized_store;

pub use codebook::Codebook;
pub use hnsw::{HnswIndex, HnswParams, HnswSnapshot, Layer, LayerSnapshot};
pub use quantized_store::{QuantizedVector, QuantizedVectorStore, VectorStoreSnapshot};
