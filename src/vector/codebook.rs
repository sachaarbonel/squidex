//! Codebook implementation for Product Quantization.
//!
//! Each codebook contains 256 centroids (cluster centers) trained using K-means clustering.
//! Subvectors are encoded as the index (0-255) of their nearest centroid.

use parking_lot::Mutex;
use std::collections::HashMap;

use crate::error::{Result, SquidexError};

/// Number of centroids per codebook (2^8 = 256 for 1-byte codes)
pub const NUM_CENTROIDS: usize = 256;

/// Maximum K-means iterations for training
const MAX_KMEANS_ITERATIONS: usize = 100;

/// Convergence threshold for K-means
const CONVERGENCE_THRESHOLD: f32 = 1e-6;

/// Codebook for one subspace in Product Quantization.
///
/// Contains 256 centroids and provides encoding/decoding of subvectors.
pub struct Codebook {
    /// Cluster centroids (256 per codebook)
    pub centroids: Vec<Vec<f32>>,

    /// Dimensionality of each centroid
    dim: usize,

    /// Distance lookup table cache: query hash -> per-centroid distances
    /// Note: In production, this cache should be per-query, not persisted
    distance_cache: Option<Mutex<HashMap<u64, [f32; NUM_CENTROIDS]>>>,
}

impl Clone for Codebook {
    fn clone(&self) -> Self {
        Self {
            centroids: self.centroids.clone(),
            dim: self.dim,
            // Create a new empty cache for the clone
            distance_cache: Some(Mutex::new(HashMap::new())),
        }
    }
}

impl std::fmt::Debug for Codebook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Codebook")
            .field("num_centroids", &self.centroids.len())
            .field("dim", &self.dim)
            .finish()
    }
}

impl Default for Codebook {
    fn default() -> Self {
        Self::new()
    }
}

impl Codebook {
    /// Create an empty codebook (must be trained before use)
    pub fn new() -> Self {
        Self {
            centroids: Vec::new(),
            dim: 0,
            distance_cache: Some(Mutex::new(HashMap::new())),
        }
    }

    /// Create a codebook with specified dimensionality and random initialization
    pub fn with_dim(dim: usize) -> Self {
        Self {
            centroids: vec![vec![0.0; dim]; NUM_CENTROIDS],
            dim,
            distance_cache: Some(Mutex::new(HashMap::new())),
        }
    }

    /// Check if the codebook has been trained
    pub fn is_trained(&self) -> bool {
        !self.centroids.is_empty() && self.dim > 0
    }

    /// Get the dimensionality of this codebook
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Train the codebook using K-means clustering.
    ///
    /// # Arguments
    /// * `subvectors` - Training data: subvectors from the corpus
    /// * `k` - Number of clusters (must be 256 for 1-byte codes)
    ///
    /// # Returns
    /// Ok(()) on successful training, or an error if training fails
    pub fn train(&mut self, subvectors: &[Vec<f32>], k: usize) -> Result<()> {
        if subvectors.is_empty() {
            return Err(SquidexError::Internal(
                "Cannot train codebook with empty data".to_string(),
            ));
        }

        if k != NUM_CENTROIDS {
            return Err(SquidexError::Internal(format!(
                "Codebook requires {} centroids, got {}",
                NUM_CENTROIDS, k
            )));
        }

        let dim = subvectors[0].len();
        if dim == 0 {
            return Err(SquidexError::Internal(
                "Subvector dimension cannot be zero".to_string(),
            ));
        }

        self.dim = dim;

        // Initialize centroids using k-means++ for better convergence
        self.centroids = self.kmeans_plusplus_init(subvectors, k);

        // Run Lloyd's algorithm
        let mut prev_inertia = f32::MAX;

        for _iteration in 0..MAX_KMEANS_ITERATIONS {
            // Assignment step: assign each subvector to nearest centroid
            let assignments = self.assign_to_centroids(subvectors);

            // Update step: recompute centroids as mean of assigned subvectors
            let (new_centroids, inertia) = self.compute_centroids(subvectors, &assignments, k);

            // Check for convergence
            if (prev_inertia - inertia).abs() < CONVERGENCE_THRESHOLD * prev_inertia {
                break;
            }

            self.centroids = new_centroids;
            prev_inertia = inertia;
        }

        // Clear the distance cache after training
        if let Some(cache) = &self.distance_cache {
            cache.lock().clear();
        }

        Ok(())
    }

    /// Encode a subvector to its nearest centroid index (0-255)
    pub fn encode(&self, subvector: &[f32]) -> u8 {
        let mut min_dist = f32::MAX;
        let mut best_idx = 0u8;

        for (idx, centroid) in self.centroids.iter().enumerate() {
            let dist = euclidean_distance_sq(subvector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best_idx = idx as u8;
            }
        }

        best_idx
    }

    /// Decode a centroid index back to the centroid vector
    pub fn decode(&self, code: u8) -> &[f32] {
        &self.centroids[code as usize]
    }

    /// Compute distance table for a query subvector.
    ///
    /// Returns an array of 256 distances from the query to each centroid.
    /// This table can be reused for computing distances to many quantized vectors.
    pub fn compute_distance_table(&self, query_subvector: &[f32]) -> [f32; NUM_CENTROIDS] {
        let mut table = [0.0_f32; NUM_CENTROIDS];

        for (centroid_id, centroid) in self.centroids.iter().enumerate() {
            table[centroid_id] = euclidean_distance_sq(query_subvector, centroid);
        }

        table
    }

    // Private helper methods

    /// K-means++ initialization for better centroid starting positions
    fn kmeans_plusplus_init(&self, data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = data.len();
        let dim = data[0].len();
        let mut centroids = Vec::with_capacity(k);

        // Use deterministic seed based on data characteristics
        let mut hasher = DefaultHasher::new();
        n.hash(&mut hasher);
        dim.hash(&mut hasher);
        let seed = hasher.finish();

        // Choose first centroid randomly (deterministically based on seed)
        let first_idx = (seed as usize) % n;
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids with probability proportional to D(x)^2
        let mut distances: Vec<f32> = vec![f32::MAX; n];

        for _c in 1..k.min(n) {
            // Update distances to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let dist = euclidean_distance_sq(point, centroids.last().unwrap());
                distances[i] = distances[i].min(dist);
            }

            // Select next centroid using weighted probability
            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                // All points are already centroids, just pick remaining points
                for point in data.iter().take(k).skip(centroids.len()) {
                    centroids.push(point.clone());
                }
                break;
            }

            // Deterministic selection based on cumulative distance
            let threshold = (((seed.wrapping_mul((centroids.len() + 1) as u64)) % 10000) as f32
                / 10000.0)
                * total;
            let mut cumsum = 0.0;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }

        // If we have fewer centroids than k (edge case), pad with random points
        while centroids.len() < k {
            let idx = (seed as usize + centroids.len()) % n;
            centroids.push(data[idx].clone());
        }

        centroids
    }

    /// Assign each subvector to its nearest centroid
    fn assign_to_centroids(&self, data: &[Vec<f32>]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                let mut min_dist = f32::MAX;
                let mut best_idx = 0;

                for (idx, centroid) in self.centroids.iter().enumerate() {
                    let dist = euclidean_distance_sq(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = idx;
                    }
                }

                best_idx
            })
            .collect()
    }

    /// Compute new centroids as mean of assigned points
    fn compute_centroids(
        &self,
        data: &[Vec<f32>],
        assignments: &[usize],
        k: usize,
    ) -> (Vec<Vec<f32>>, f32) {
        let dim = self.dim;
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        let mut inertia = 0.0_f32;

        // Accumulate sums and counts
        for (point, &assignment) in data.iter().zip(assignments.iter()) {
            for (j, &val) in point.iter().enumerate() {
                sums[assignment][j] += val;
            }
            counts[assignment] += 1;

            // Compute inertia (sum of squared distances to centroids)
            inertia += euclidean_distance_sq(point, &self.centroids[assignment]);
        }

        // Compute means
        let mut new_centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        for i in 0..k {
            if counts[i] > 0 {
                let centroid: Vec<f32> = sums[i].iter().map(|&s| s / counts[i] as f32).collect();
                new_centroids.push(centroid);
            } else {
                // Keep existing centroid if no points assigned
                new_centroids.push(self.centroids[i].clone());
            }
        }

        (new_centroids, inertia)
    }
}

/// Compute squared Euclidean distance between two vectors
#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute Euclidean distance between two vectors
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_creation() {
        let codebook = Codebook::new();
        assert!(!codebook.is_trained());

        let codebook = Codebook::with_dim(16);
        assert_eq!(codebook.dimension(), 16);
        assert_eq!(codebook.centroids.len(), NUM_CENTROIDS);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let dist_sq = euclidean_distance_sq(&a, &b);
        assert!((dist_sq - 2.0).abs() < 1e-6);

        let dist = euclidean_distance(&a, &b);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_codebook_training() {
        // Create simple training data: 4 distinct clusters
        let mut training_data = Vec::new();

        // Cluster 1: around (0, 0)
        for _ in 0..100 {
            training_data.push(vec![0.1, 0.1]);
        }
        // Cluster 2: around (10, 0)
        for _ in 0..100 {
            training_data.push(vec![10.0, 0.1]);
        }
        // Cluster 3: around (0, 10)
        for _ in 0..100 {
            training_data.push(vec![0.1, 10.0]);
        }
        // Cluster 4: around (10, 10)
        for _ in 0..100 {
            training_data.push(vec![10.0, 10.0]);
        }

        let mut codebook = Codebook::new();
        codebook.train(&training_data, NUM_CENTROIDS).unwrap();

        assert!(codebook.is_trained());
        assert_eq!(codebook.dimension(), 2);
        assert_eq!(codebook.centroids.len(), NUM_CENTROIDS);
    }

    #[test]
    fn test_encode_decode() {
        // Train a simple codebook
        let training_data: Vec<Vec<f32>> = (0..1000).map(|i| vec![i as f32, (i * 2) as f32]).collect();

        let mut codebook = Codebook::new();
        codebook.train(&training_data, NUM_CENTROIDS).unwrap();

        // Encode and check that decoding gives reasonable results
        let test_vector = vec![500.0, 1000.0];
        let code = codebook.encode(&test_vector);
        let decoded = codebook.decode(code);

        // The decoded vector should be somewhat close to the original
        let dist = euclidean_distance(&test_vector, decoded);
        assert!(dist < 100.0, "Decoded vector should be reasonably close");
    }

    #[test]
    fn test_distance_table() {
        let training_data: Vec<Vec<f32>> = (0..500).map(|i| vec![i as f32]).collect();

        let mut codebook = Codebook::new();
        codebook.train(&training_data, NUM_CENTROIDS).unwrap();

        let query = vec![250.0];
        let table = codebook.compute_distance_table(&query);

        assert_eq!(table.len(), NUM_CENTROIDS);

        // All distances should be non-negative
        for &dist in &table {
            assert!(dist >= 0.0);
        }
    }
}
