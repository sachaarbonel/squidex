//! Vector storage module with Product Quantization for memory-efficient storage.
//!
//! This module provides a production-grade vector storage system using Product Quantization (PQ)
//! to achieve significant memory compression (64x for typical configurations) while maintaining
//! reasonable search accuracy.

mod codebook;
mod quantized_store;

pub use codebook::Codebook;
pub use quantized_store::{QuantizedVector, QuantizedVectorStore, VectorStoreSnapshot};
