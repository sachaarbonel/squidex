//! Quantized Vector Store using Product Quantization (PQ).
//!
//! Product Quantization compresses high-dimensional vectors by:
//! 1. Splitting the vector into M subspaces
//! 2. Quantizing each subspace independently using a codebook of 256 centroids
//! 3. Storing only the centroid indices (1 byte per subspace)
//!
//! Example compression (384 dimensions, 24 subspaces):
//! - Original: 384 × 4 bytes = 1,536 bytes
//! - Quantized: 24 bytes (1 byte per subspace)
//! - Compression ratio: 64x

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SquidexError};
use crate::models::DocumentId;
use crate::vector::codebook::{euclidean_distance_sq, Codebook, NUM_CENTROIDS};

/// Default cache size for full-precision vectors
const DEFAULT_CACHE_SIZE: usize = 10_000;

/// Minimum training samples required for quality codebooks
const MIN_TRAINING_SAMPLES: usize = 256;

/// Quantized representation of a vector
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedVector {
    /// Subspace codes (one byte per subspace)
    pub codes: Vec<u8>,
}

impl QuantizedVector {
    /// Create a new quantized vector with the given codes
    pub fn new(codes: Vec<u8>) -> Self {
        Self { codes }
    }

    /// Get the number of subspaces
    pub fn num_subspaces(&self) -> usize {
        self.codes.len()
    }

    /// Get the memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.codes.len()
    }
}

/// Product Quantization for memory-efficient vector storage.
///
/// This implementation provides:
/// - 64x compression for typical configurations (384 dims, 24 subspaces)
/// - Asymmetric Distance Computation (ADC) for fast similarity search
/// - LRU cache for frequently accessed full-precision vectors
/// - Batch training on corpus vectors
pub struct QuantizedVectorStore {
    /// Original vector dimensions
    dimensions: usize,

    /// Number of subspaces (M)
    num_subspaces: usize,

    /// Dimensions per subspace
    subspace_dim: usize,

    /// Codebooks (one per subspace)
    codebooks: Vec<Codebook>,

    /// Quantized vectors storage: doc_id -> codes
    /// NOTE: In production, this would use dense arrays indexed by docno
    quantized: RwLock<HashMap<DocumentId, Vec<u8>>>,

    /// LRU cache for full-precision vectors (optional hot cache)
    full_precision_cache: RwLock<LruCache<DocumentId, Vec<f32>>>,

    /// Whether codebooks have been trained
    trained: bool,

    /// Training vectors buffer (used before training)
    training_buffer: RwLock<Vec<(DocumentId, Vec<f32>)>>,
}

/// Simple LRU cache implementation
pub struct LruCache<K, V> {
    capacity: usize,
    items: HashMap<K, V>,
    access_order: Vec<K>,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: HashMap::with_capacity(capacity),
            access_order: Vec::with_capacity(capacity),
        }
    }

    /// Get a value from the cache, updating access order
    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.items.get(key) {
            // Move to end of access order (most recently used)
            self.access_order.retain(|k| k != key);
            self.access_order.push(key.clone());
            Some(value.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&mut self, key: K, value: V) {
        // Evict if at capacity
        if self.items.len() >= self.capacity && !self.items.contains_key(&key) {
            if let Some(oldest) = self.access_order.first().cloned() {
                self.items.remove(&oldest);
                self.access_order.remove(0);
            }
        }

        // Update access order
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key.clone());

        self.items.insert(key, value);
    }

    /// Remove a value from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.access_order.retain(|k| k != key);
        self.items.remove(key)
    }

    /// Check if a key exists in the cache
    pub fn contains_key(&self, key: &K) -> bool {
        self.items.contains_key(key)
    }

    /// Get the current size of the cache
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.items.clear();
        self.access_order.clear();
    }
}

impl std::fmt::Debug for QuantizedVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedVectorStore")
            .field("dimensions", &self.dimensions)
            .field("num_subspaces", &self.num_subspaces)
            .field("subspace_dim", &self.subspace_dim)
            .field("trained", &self.trained)
            .field("vector_count", &self.quantized.read().len())
            .finish()
    }
}

impl QuantizedVectorStore {
    /// Create a new quantized vector store.
    ///
    /// # Arguments
    /// * `dimensions` - Total vector dimensions (must be divisible by num_subspaces)
    /// * `num_subspaces` - Number of subspaces for PQ (typically 16-48)
    ///
    /// # Panics
    /// Panics if dimensions is not divisible by num_subspaces
    pub fn new(dimensions: usize, num_subspaces: usize) -> Self {
        assert!(
            dimensions % num_subspaces == 0,
            "Dimensions ({}) must be divisible by num_subspaces ({})",
            dimensions,
            num_subspaces
        );
        assert!(num_subspaces > 0, "num_subspaces must be positive");
        assert!(dimensions > 0, "dimensions must be positive");

        let subspace_dim = dimensions / num_subspaces;

        Self {
            dimensions,
            num_subspaces,
            subspace_dim,
            codebooks: (0..num_subspaces).map(|_| Codebook::new()).collect(),
            quantized: RwLock::new(HashMap::new()),
            full_precision_cache: RwLock::new(LruCache::new(DEFAULT_CACHE_SIZE)),
            trained: false,
            training_buffer: RwLock::new(Vec::new()),
        }
    }

    /// Create with a custom cache size
    pub fn with_cache_size(dimensions: usize, num_subspaces: usize, cache_size: usize) -> Self {
        let store = Self::new(dimensions, num_subspaces);
        *store.full_precision_cache.write() = LruCache::new(cache_size);
        store
    }

    /// Get the vector dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the number of subspaces
    pub fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Get the subspace dimension
    pub fn subspace_dim(&self) -> usize {
        self.subspace_dim
    }

    /// Check if the codebooks have been trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get the number of stored vectors
    pub fn len(&self) -> usize {
        self.quantized.read().len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.quantized.read().is_empty()
    }

    /// Get the compression ratio (original bytes / quantized bytes)
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimensions * 4; // f32 = 4 bytes
        let quantized_bytes = self.num_subspaces; // 1 byte per subspace
        original_bytes as f32 / quantized_bytes as f32
    }

    /// Train codebooks using K-means on the provided training vectors.
    ///
    /// # Arguments
    /// * `training_vectors` - Representative vectors from the corpus
    ///
    /// # Returns
    /// Ok(()) on success, or an error if training fails
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        if training_vectors.is_empty() {
            return Err(SquidexError::Internal(
                "Cannot train with empty training set".to_string(),
            ));
        }

        // Validate dimensions
        for (i, vec) in training_vectors.iter().enumerate() {
            if vec.len() != self.dimensions {
                return Err(SquidexError::InvalidEmbeddingDimensions {
                    expected: self.dimensions,
                    actual: vec.len(),
                });
            }
            if i == 0 && training_vectors.len() < MIN_TRAINING_SAMPLES {
                // Warning: small training set may produce poor codebooks
                // In production, you might want to handle this differently
            }
        }

        // Train each codebook on its corresponding subspace
        for subspace_id in 0..self.num_subspaces {
            let start = subspace_id * self.subspace_dim;
            let end = start + self.subspace_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = training_vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Train the codebook
            self.codebooks[subspace_id].train(&subvectors, NUM_CENTROIDS)?;
        }

        self.trained = true;

        // Process any buffered vectors
        let buffer: Vec<_> = std::mem::take(&mut *self.training_buffer.write());
        for (doc_id, vector) in buffer {
            self.quantize_and_store(doc_id, &vector)?;
        }

        Ok(())
    }

    /// Quantize a vector to its PQ codes.
    ///
    /// # Arguments
    /// * `vector` - The vector to quantize
    ///
    /// # Returns
    /// The quantized vector codes
    pub fn quantize(&self, vector: &[f32]) -> Result<QuantizedVector> {
        if !self.trained {
            return Err(SquidexError::Internal(
                "Codebooks not trained yet".to_string(),
            ));
        }

        if vector.len() != self.dimensions {
            return Err(SquidexError::InvalidEmbeddingDimensions {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }

        let mut codes = Vec::with_capacity(self.num_subspaces);

        for subspace_id in 0..self.num_subspaces {
            let start = subspace_id * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subvector = &vector[start..end];

            // Find nearest centroid
            let code = self.codebooks[subspace_id].encode(subvector);
            codes.push(code);
        }

        Ok(QuantizedVector { codes })
    }

    /// Quantize and store a vector.
    ///
    /// # Arguments
    /// * `doc_id` - Document ID for this vector
    /// * `vector` - The vector to quantize and store
    ///
    /// # Returns
    /// The quantized vector representation
    pub fn quantize_and_store(
        &mut self,
        doc_id: DocumentId,
        vector: &[f32],
    ) -> Result<QuantizedVector> {
        if vector.len() != self.dimensions {
            return Err(SquidexError::InvalidEmbeddingDimensions {
                expected: self.dimensions,
                actual: vector.len(),
            });
        }

        // If not trained, buffer the vector for later
        if !self.trained {
            self.training_buffer.write().push((doc_id, vector.to_vec()));
            // Return empty codes as placeholder
            return Ok(QuantizedVector {
                codes: vec![0; self.num_subspaces],
            });
        }

        let codes = self.quantize(vector)?;
        self.quantized.write().insert(doc_id, codes.codes.clone());

        Ok(codes)
    }

    /// Store a vector (quantize if trained, buffer if not).
    ///
    /// This is a convenience method that handles both trained and untrained states.
    pub fn store(&mut self, doc_id: DocumentId, vector: &[f32]) -> Result<()> {
        self.quantize_and_store(doc_id, vector)?;
        Ok(())
    }

    /// Store a vector and optionally cache full precision.
    pub fn store_with_cache(
        &mut self,
        doc_id: DocumentId,
        vector: &[f32],
        cache_full_precision: bool,
    ) -> Result<()> {
        self.quantize_and_store(doc_id, vector)?;

        if cache_full_precision {
            self.full_precision_cache
                .write()
                .insert(doc_id, vector.to_vec());
        }

        Ok(())
    }

    /// Remove a vector from the store
    pub fn remove(&mut self, doc_id: DocumentId) -> bool {
        let removed = self.quantized.write().remove(&doc_id).is_some();
        self.full_precision_cache.write().remove(&doc_id);
        removed
    }

    /// Check if a document has a stored vector
    pub fn contains(&self, doc_id: DocumentId) -> bool {
        self.quantized.read().contains_key(&doc_id)
    }

    /// Get the quantized codes for a document
    pub fn get_codes(&self, doc_id: DocumentId) -> Option<Vec<u8>> {
        self.quantized.read().get(&doc_id).cloned()
    }

    /// Build ADC (Asymmetric Distance Computation) lookup tables for a query.
    ///
    /// The returned tables allow O(M) distance computation to any quantized vector,
    /// where M is the number of subspaces.
    ///
    /// # Arguments
    /// * `query` - The query vector
    ///
    /// # Returns
    /// A vector of lookup tables, one per subspace
    pub fn build_distance_table(&self, query: &[f32]) -> Result<Vec<[f32; NUM_CENTROIDS]>> {
        if !self.trained {
            return Err(SquidexError::Internal(
                "Codebooks not trained yet".to_string(),
            ));
        }

        if query.len() != self.dimensions {
            return Err(SquidexError::InvalidEmbeddingDimensions {
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        let mut tables = Vec::with_capacity(self.num_subspaces);

        for subspace_id in 0..self.num_subspaces {
            let start = subspace_id * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_subvec = &query[start..end];

            let mut table = [0.0_f32; NUM_CENTROIDS];
            for (centroid_id, centroid) in self.codebooks[subspace_id].centroids.iter().enumerate()
            {
                table[centroid_id] = euclidean_distance_sq(query_subvec, centroid);
            }
            tables.push(table);
        }

        Ok(tables)
    }

    /// Compute asymmetric distance from query to a quantized vector using lookup tables.
    ///
    /// # Arguments
    /// * `tables` - Pre-computed distance tables from build_distance_table
    /// * `doc_id` - Document ID to compute distance to
    ///
    /// # Returns
    /// The Euclidean distance (not squared)
    pub fn distance_quantized(
        &self,
        tables: &[[f32; NUM_CENTROIDS]],
        doc_id: DocumentId,
    ) -> Result<f32> {
        let codes = self
            .quantized
            .read()
            .get(&doc_id)
            .cloned()
            .ok_or(SquidexError::VectorNotFound(doc_id))?;

        let mut distance_sq = 0.0;

        for (subspace_id, &code) in codes.iter().enumerate() {
            distance_sq += tables[subspace_id][code as usize];
        }

        Ok(distance_sq.sqrt())
    }

    /// Compute asymmetric distance using codes directly (no lookup needed)
    pub fn distance_quantized_codes(&self, tables: &[[f32; NUM_CENTROIDS]], codes: &[u8]) -> f32 {
        let mut distance_sq = 0.0;

        for (subspace_id, &code) in codes.iter().enumerate() {
            distance_sq += tables[subspace_id][code as usize];
        }

        distance_sq.sqrt()
    }

    /// Reconstruct an approximate vector from quantized codes.
    ///
    /// This gives a lossy reconstruction using the centroid values.
    pub fn reconstruct(&self, doc_id: DocumentId) -> Result<Vec<f32>> {
        if !self.trained {
            return Err(SquidexError::Internal(
                "Codebooks not trained yet".to_string(),
            ));
        }

        let codes = self
            .quantized
            .read()
            .get(&doc_id)
            .cloned()
            .ok_or(SquidexError::VectorNotFound(doc_id))?;

        let mut reconstructed = Vec::with_capacity(self.dimensions);

        for (subspace_id, &code) in codes.iter().enumerate() {
            let centroid = self.codebooks[subspace_id].decode(code);
            reconstructed.extend_from_slice(centroid);
        }

        Ok(reconstructed)
    }

    /// Get full precision vector from cache if available
    pub fn get_full_precision(&self, doc_id: DocumentId) -> Option<Vec<f32>> {
        self.full_precision_cache.write().get(&doc_id)
    }

    /// Search for nearest neighbors using ADC.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// Vector of (doc_id, distance) tuples, sorted by distance ascending
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(DocumentId, f32)>> {
        let tables = self.build_distance_table(query)?;
        let quantized = self.quantized.read();

        let mut results: Vec<(DocumentId, f32)> = quantized
            .iter()
            .map(|(doc_id, codes)| {
                let distance = self.distance_quantized_codes(&tables, codes);
                (*doc_id, distance)
            })
            .collect();

        // Sort by distance ascending
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Get all stored document IDs
    pub fn document_ids(&self) -> Vec<DocumentId> {
        self.quantized.read().keys().cloned().collect()
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let quantized = self.quantized.read();
        let cache = self.full_precision_cache.read();

        let quantized_bytes = quantized.len() * self.num_subspaces;
        let codebook_bytes = self.num_subspaces * NUM_CENTROIDS * self.subspace_dim * 4;
        let cache_bytes = cache.len() * self.dimensions * 4;

        MemoryStats {
            vector_count: quantized.len(),
            quantized_bytes,
            codebook_bytes,
            cache_bytes,
            total_bytes: quantized_bytes + codebook_bytes + cache_bytes,
            compression_ratio: self.compression_ratio(),
        }
    }
}

/// Memory usage statistics for the vector store
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of stored vectors
    pub vector_count: usize,
    /// Bytes used for quantized codes
    pub quantized_bytes: usize,
    /// Bytes used for codebooks
    pub codebook_bytes: usize,
    /// Bytes used for full-precision cache
    pub cache_bytes: usize,
    /// Total memory usage
    pub total_bytes: usize,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Serializable snapshot of the vector store for persistence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorStoreSnapshot {
    /// Vector dimensions
    pub dimensions: usize,
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Whether codebooks are trained
    pub trained: bool,
    /// Codebook centroids (flattened: num_subspaces * 256 * subspace_dim)
    pub codebook_centroids: Vec<Vec<Vec<f32>>>,
    /// Quantized vectors: doc_id -> codes
    pub quantized_vectors: HashMap<DocumentId, Vec<u8>>,
    /// Buffered vectors awaiting training
    pub training_buffer: Vec<(DocumentId, Vec<f32>)>,
}

impl QuantizedVectorStore {
    /// Create a snapshot of the vector store for persistence
    pub fn create_snapshot(&self) -> VectorStoreSnapshot {
        let codebook_centroids: Vec<Vec<Vec<f32>>> = self
            .codebooks
            .iter()
            .map(|cb| cb.centroids.clone())
            .collect();

        VectorStoreSnapshot {
            dimensions: self.dimensions,
            num_subspaces: self.num_subspaces,
            trained: self.trained,
            codebook_centroids,
            quantized_vectors: self.quantized.read().clone(),
            training_buffer: self.training_buffer.read().clone(),
        }
    }

    /// Restore from a snapshot
    pub fn restore_from_snapshot(snapshot: VectorStoreSnapshot, cache_size: usize) -> Self {
        let subspace_dim = snapshot.dimensions / snapshot.num_subspaces;

        // Restore codebooks
        let codebooks: Vec<Codebook> = snapshot
            .codebook_centroids
            .into_iter()
            .map(|centroids| {
                let mut cb = Codebook::new();
                if !centroids.is_empty() {
                    cb.centroids = centroids;
                }
                cb
            })
            .collect();

        Self {
            dimensions: snapshot.dimensions,
            num_subspaces: snapshot.num_subspaces,
            subspace_dim,
            codebooks,
            quantized: RwLock::new(snapshot.quantized_vectors),
            full_precision_cache: RwLock::new(LruCache::new(cache_size)),
            trained: snapshot.trained,
            training_buffer: RwLock::new(snapshot.training_buffer),
        }
    }

    /// Check if auto-training should be triggered
    pub fn should_auto_train(&self, min_vectors: usize) -> bool {
        !self.trained && self.training_buffer.read().len() >= min_vectors
    }

    /// Perform auto-training using buffered vectors
    pub fn auto_train(&mut self) -> Result<bool> {
        if self.trained {
            return Ok(false);
        }

        let buffer = self.training_buffer.read();
        if buffer.is_empty() {
            return Ok(false);
        }

        // Extract vectors for training
        let training_vectors: Vec<Vec<f32>> = buffer.iter().map(|(_, v)| v.clone()).collect();
        drop(buffer);

        // Train the codebooks
        self.train(&training_vectors)?;

        Ok(true)
    }

    /// Get the number of buffered vectors awaiting training
    pub fn buffered_count(&self) -> usize {
        self.training_buffer.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut hasher = DefaultHasher::new();
                        (i * dim + j).hash(&mut hasher);
                        (hasher.finish() % 1000) as f32 / 1000.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_store_creation() {
        let store = QuantizedVectorStore::new(384, 24);
        assert_eq!(store.dimensions(), 384);
        assert_eq!(store.num_subspaces(), 24);
        assert_eq!(store.subspace_dim(), 16);
        assert!(!store.is_trained());
        assert!(store.is_empty());
    }

    #[test]
    fn test_compression_ratio() {
        let store = QuantizedVectorStore::new(384, 24);
        let ratio = store.compression_ratio();
        assert!((ratio - 64.0).abs() < 0.1); // 384*4 / 24 = 64
    }

    #[test]
    #[should_panic]
    fn test_invalid_dimensions() {
        // 385 is not divisible by 24
        QuantizedVectorStore::new(385, 24);
    }

    #[test]
    fn test_training_and_quantization() {
        let mut store = QuantizedVectorStore::new(16, 4);

        // Create training data
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        assert!(store.is_trained());

        // Quantize a vector
        let test_vec = create_random_vectors(1, 16).pop().unwrap();
        let quantized = store.quantize(&test_vec).unwrap();

        assert_eq!(quantized.codes.len(), 4);
        assert_eq!(quantized.num_subspaces(), 4);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        // Store a vector
        let vec = create_random_vectors(1, 16).pop().unwrap();
        store.quantize_and_store(42, &vec).unwrap();

        assert!(store.contains(42));
        assert!(!store.contains(43));
        assert_eq!(store.len(), 1);

        // Get codes
        let codes = store.get_codes(42).unwrap();
        assert_eq!(codes.len(), 4);
    }

    #[test]
    fn test_distance_computation() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        // Store some vectors
        let vecs = create_random_vectors(10, 16);
        for (i, vec) in vecs.iter().enumerate() {
            store.quantize_and_store(i as u64, vec).unwrap();
        }

        // Compute distances
        let query = &vecs[0];
        let tables = store.build_distance_table(query).unwrap();

        // Distance to self should be small (not zero due to quantization error)
        let self_dist = store.distance_quantized(&tables, 0).unwrap();
        assert!(self_dist < 1.0, "Self-distance should be small");
    }

    #[test]
    fn test_search() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        // Store vectors
        let vecs = create_random_vectors(100, 16);
        for (i, vec) in vecs.iter().enumerate() {
            store.quantize_and_store(i as u64, vec).unwrap();
        }

        // Search using the first vector as query
        let results = store.search(&vecs[0], 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the query itself (or very close)
        assert_eq!(results[0].0, 0);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_reconstruction() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        let original = create_random_vectors(1, 16).pop().unwrap();
        store.quantize_and_store(1, &original).unwrap();

        let reconstructed = store.reconstruct(1).unwrap();
        assert_eq!(reconstructed.len(), 16);

        // Reconstruction should be approximate but not exact
        let mut diff = 0.0;
        for (a, b) in original.iter().zip(reconstructed.iter()) {
            diff += (a - b).abs();
        }
        // Average difference per dimension should be reasonable
        let avg_diff = diff / 16.0;
        assert!(
            avg_diff < 0.5,
            "Reconstruction error too high: {}",
            avg_diff
        );
    }

    #[test]
    fn test_remove() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        let vec = create_random_vectors(1, 16).pop().unwrap();
        store.quantize_and_store(1, &vec).unwrap();

        assert!(store.contains(1));
        assert!(store.remove(1));
        assert!(!store.contains(1));
        assert!(!store.remove(1)); // Already removed
    }

    #[test]
    fn test_memory_stats() {
        let mut store = QuantizedVectorStore::new(16, 4);
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        for i in 0..100 {
            let vec = create_random_vectors(1, 16).pop().unwrap();
            store.quantize_and_store(i, &vec).unwrap();
        }

        let stats = store.memory_stats();
        assert_eq!(stats.vector_count, 100);
        assert_eq!(stats.quantized_bytes, 100 * 4); // 100 vectors * 4 subspaces
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_buffering_before_training() {
        let mut store = QuantizedVectorStore::new(16, 4);

        // Store vectors before training
        let vec1 = create_random_vectors(1, 16).pop().unwrap();
        let vec2 = create_random_vectors(1, 16).pop().unwrap();

        store.store(1, &vec1).unwrap();
        store.store(2, &vec2).unwrap();

        assert!(!store.is_trained());

        // Now train
        let training_vectors = create_random_vectors(500, 16);
        store.train(&training_vectors).unwrap();

        // Buffered vectors should now be stored
        assert!(store.contains(1));
        assert!(store.contains(2));
    }

    #[test]
    fn test_lru_cache() {
        let mut cache: LruCache<u64, String> = LruCache::new(3);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.len(), 3);

        // Insert 4th item, should evict oldest (1)
        cache.insert(4, "four".to_string());
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains_key(&1));
        assert!(cache.contains_key(&4));

        // Access 2, making it most recent
        cache.get(&2);

        // Insert 5th item, should evict 3 (not 2)
        cache.insert(5, "five".to_string());
        assert!(!cache.contains_key(&3));
        assert!(cache.contains_key(&2));
    }
}
