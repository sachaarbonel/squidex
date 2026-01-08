//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! This implementation provides:
//! - Logarithmic search complexity O(log N) average case
//! - Deterministic level assignment for Raft-compatible replay
//! - Integration with Product Quantization for memory-efficient distance computation
//! - Delete bitset for soft deletes without graph modification

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::{Result, SquidexError};
use crate::models::DocumentId;
use crate::vector::codebook::NUM_CENTROIDS;
use crate::vector::QuantizedVectorStore;

/// HNSW configuration parameters
#[derive(Clone, Debug)]
pub struct HnswParams {
    /// Max connections per node at layers > 0 (M)
    pub max_connections: usize,

    /// Max connections at layer 0 (M0 = 2*M typically)
    pub max_connections_layer0: usize,

    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search
    pub ef_search: usize,

    /// Level multiplier (ml = 1/ln(M))
    pub level_multiplier: f64,

    /// Global seed for deterministic level assignment
    pub level_seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_connections: 16,
            max_connections_layer0: 32,
            ef_construction: 200,
            ef_search: 50,
            level_multiplier: 1.0 / (16_f64).ln(),
            level_seed: 0x5EED5EED,
        }
    }
}

/// Adjacency storage for a single layer
#[derive(Clone, Debug, Default)]
pub struct Layer {
    /// Adjacency lists: node_id -> list of (neighbor_id, distance)
    adjacency: HashMap<DocumentId, Vec<(DocumentId, f32)>>,
}

impl Layer {
    /// Create a new empty layer
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node: DocumentId) -> Option<&Vec<(DocumentId, f32)>> {
        self.adjacency.get(&node)
    }

    /// Set neighbors for a node
    pub fn set_neighbors(&mut self, node: DocumentId, neighbors: Vec<(DocumentId, f32)>) {
        self.adjacency.insert(node, neighbors);
    }

    /// Add a node with empty neighbors
    pub fn add_node(&mut self, node: DocumentId) {
        self.adjacency.entry(node).or_insert_with(Vec::new);
    }

    /// Check if node exists in layer
    pub fn contains(&self, node: DocumentId) -> bool {
        self.adjacency.contains_key(&node)
    }

    /// Get all nodes in layer
    pub fn nodes(&self) -> impl Iterator<Item = &DocumentId> {
        self.adjacency.keys()
    }

    /// Number of nodes in layer
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Check if layer is empty
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }
}

/// HNSW index for approximate nearest neighbor search
pub struct HnswIndex {
    /// Graph layers (layer 0 = all vectors, higher layers = fewer nodes)
    layers: Vec<Layer>,

    /// Entry point (node at the highest layer)
    entry_point: Option<DocumentId>,

    /// Maximum layer of the entry point
    max_layer: usize,

    /// Node levels: doc_id -> assigned level
    node_levels: HashMap<DocumentId, usize>,

    /// Vector storage with product quantization
    vectors: QuantizedVectorStore,

    /// HNSW parameters
    params: HnswParams,

    /// Delete bitset (soft deletes)
    deleted: RwLock<HashSet<DocumentId>>,

    /// Node count
    node_count: usize,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("node_count", &self.node_count)
            .field("max_layer", &self.max_layer)
            .field("entry_point", &self.entry_point)
            .field("params", &self.params)
            .finish()
    }
}

impl HnswIndex {
    /// Create a new HNSW index
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensions
    /// * `num_subspaces` - PQ subspaces (dimensions must be divisible by this)
    /// * `params` - HNSW configuration
    pub fn new(dimensions: usize, num_subspaces: usize, params: HnswParams) -> Self {
        Self {
            layers: Vec::new(),
            entry_point: None,
            max_layer: 0,
            node_levels: HashMap::new(),
            vectors: QuantizedVectorStore::new(dimensions, num_subspaces),
            params,
            deleted: RwLock::new(HashSet::new()),
            node_count: 0,
        }
    }

    /// Create with default parameters
    pub fn with_defaults(dimensions: usize, num_subspaces: usize) -> Self {
        Self::new(dimensions, num_subspaces, HnswParams::default())
    }

    /// Get immutable reference to parameters
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Get the vector store
    pub fn vectors(&self) -> &QuantizedVectorStore {
        &self.vectors
    }

    /// Get mutable reference to vector store (for training)
    pub fn vectors_mut(&mut self) -> &mut QuantizedVectorStore {
        &mut self.vectors
    }

    /// Get node count
    pub fn len(&self) -> usize {
        self.node_count
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Check if vectors are trained
    pub fn is_trained(&self) -> bool {
        self.vectors.is_trained()
    }

    /// Train the vector store codebooks
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        self.vectors.train(training_vectors)
    }

    /// Compute deterministic level for a document ID
    ///
    /// Uses hash of (doc_id, seed) to generate reproducible levels
    fn compute_level(&self, doc_id: DocumentId) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        doc_id.hash(&mut hasher);
        self.params.level_seed.hash(&mut hasher);
        let hash = hasher.finish();

        // Convert hash to uniform [0, 1)
        let uniform = (hash as f64) / (u64::MAX as f64);

        // Geometric distribution: level = floor(-ln(uniform) * ml)
        let level = (-uniform.ln() * self.params.level_multiplier).floor() as usize;

        // Cap at reasonable maximum to prevent degenerate graphs
        level.min(16)
    }

    /// Insert a vector into the index
    ///
    /// # Arguments
    /// * `doc_id` - Document ID
    /// * `vector` - Vector to insert
    ///
    /// # Returns
    /// Ok(()) on success
    pub fn insert(&mut self, doc_id: DocumentId, vector: &[f32]) -> Result<()> {
        // Validate and store vector
        self.vectors.quantize_and_store(doc_id, vector)?;

        // If not trained yet, just buffer the vector
        if !self.vectors.is_trained() {
            return Ok(());
        }

        // Compute level for this node
        let level = self.compute_level(doc_id);
        self.node_levels.insert(doc_id, level);

        // Ensure layers exist up to this level
        while self.layers.len() <= level {
            self.layers.push(Layer::new());
        }

        // First insertion
        if self.entry_point.is_none() {
            self.entry_point = Some(doc_id);
            self.max_layer = level;
            for l in 0..=level {
                self.layers[l].add_node(doc_id);
            }
            self.node_count += 1;
            return Ok(());
        }

        // Build distance table for this vector
        let tables = self.vectors.build_distance_table(vector)?;

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = entry_point;

        // Phase 1: Greedy search from top layer down to level+1
        for layer_idx in (level + 1..=self.max_layer).rev() {
            current_nearest = self.search_layer_single(&tables, current_nearest, layer_idx)?;
        }

        // Phase 2: Insert at each layer from level down to 0
        for layer_idx in (0..=level.min(self.max_layer)).rev() {
            // Find ef_construction nearest neighbors at this layer
            let neighbors = self.search_layer(
                &tables,
                vec![current_nearest],
                self.params.ef_construction,
                layer_idx,
            )?;

            // Add node to this layer
            self.layers[layer_idx].add_node(doc_id);

            // Select M best neighbors and connect
            let max_conn = if layer_idx == 0 {
                self.params.max_connections_layer0
            } else {
                self.params.max_connections
            };

            self.connect_neighbors(doc_id, &neighbors, max_conn, layer_idx)?;

            // Update entry for next layer
            if !neighbors.is_empty() {
                current_nearest = neighbors[0].0;
            }
        }

        // Update entry point if new node has higher level
        if level > self.max_layer {
            self.entry_point = Some(doc_id);
            self.max_layer = level;
        }

        self.node_count += 1;
        Ok(())
    }

    /// Connect a node to its neighbors with bidirectional edges
    fn connect_neighbors(
        &mut self,
        node: DocumentId,
        candidates: &[(DocumentId, f32)],
        max_connections: usize,
        layer: usize,
    ) -> Result<()> {
        // Select best neighbors (already sorted by distance)
        let selected: Vec<(DocumentId, f32)> = candidates
            .iter()
            .filter(|(id, _)| *id != node)
            .take(max_connections)
            .cloned()
            .collect();

        // Set outgoing edges from node
        self.layers[layer].set_neighbors(node, selected.clone());

        // Add reverse edges (bidirectional)
        for (neighbor_id, dist) in &selected {
            if let Some(neighbor_edges) = self.layers[layer].adjacency.get_mut(neighbor_id) {
                // Check if edge already exists
                if !neighbor_edges.iter().any(|(id, _)| *id == node) {
                    neighbor_edges.push((node, *dist));

                    // Prune if over limit
                    if neighbor_edges.len() > max_connections {
                        // Sort by distance and keep best
                        neighbor_edges.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        neighbor_edges.truncate(max_connections);
                    }
                }
            }
        }

        Ok(())
    }

    /// Greedy search to find single nearest node in a layer
    fn search_layer_single(
        &self,
        tables: &[[f32; NUM_CENTROIDS]],
        start: DocumentId,
        layer: usize,
    ) -> Result<DocumentId> {
        let mut current = start;
        let mut current_dist = self.vectors.distance_quantized(tables, current)?;

        loop {
            let mut changed = false;

            if let Some(neighbors) = self.layers[layer].get_neighbors(current) {
                for (neighbor_id, _) in neighbors {
                    // Skip deleted nodes
                    if self.deleted.read().contains(neighbor_id) {
                        continue;
                    }

                    let dist = self.vectors.distance_quantized(tables, *neighbor_id)?;
                    if dist < current_dist {
                        current = *neighbor_id;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Search for ef nearest neighbors within a layer
    fn search_layer(
        &self,
        tables: &[[f32; NUM_CENTROIDS]],
        entry_points: Vec<DocumentId>,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<(DocumentId, f32)>> {
        let deleted = self.deleted.read();

        let mut visited: HashSet<DocumentId> = HashSet::new();
        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, DocumentId)>> =
            BinaryHeap::new();
        // Max-heap for results (furthest first for easy pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, DocumentId)> = BinaryHeap::new();

        // Initialize with entry points
        for ep in entry_points {
            if visited.insert(ep) && !deleted.contains(&ep) {
                let dist = self.vectors.distance_quantized(tables, ep)?;
                candidates.push(Reverse((OrderedFloat(dist), ep)));
                results.push((OrderedFloat(dist), ep));
            }
        }

        while let Some(Reverse((dist, current))) = candidates.pop() {
            // If candidate is further than worst result, we're done
            if let Some(&(worst_dist, _)) = results.peek() {
                if dist > worst_dist && results.len() >= ef {
                    break;
                }
            }

            // Explore neighbors
            if let Some(neighbors) = self.layers[layer].get_neighbors(current) {
                for (neighbor_id, _) in neighbors {
                    if !visited.insert(*neighbor_id) {
                        continue;
                    }
                    if deleted.contains(neighbor_id) {
                        continue;
                    }

                    let neighbor_dist = self.vectors.distance_quantized(tables, *neighbor_id)?;

                    // Add to results if closer than worst or we have room
                    let dominated = results
                        .peek()
                        .map(|(d, _)| neighbor_dist > d.0 && results.len() >= ef)
                        .unwrap_or(false);

                    if !dominated {
                        candidates.push(Reverse((OrderedFloat(neighbor_dist), *neighbor_id)));
                        results.push((OrderedFloat(neighbor_dist), *neighbor_id));

                        // Prune results to ef
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec (ascending distance)
        let mut result_vec: Vec<(DocumentId, f32)> =
            results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(result_vec)
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    /// Vector of (doc_id, distance) sorted by distance ascending
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(DocumentId, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        if !self.vectors.is_trained() {
            return Err(SquidexError::Internal(
                "HNSW search requires trained codebooks".to_string(),
            ));
        }

        // Build distance table for query
        let tables = self.vectors.build_distance_table(query)?;

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = entry_point;

        // Phase 1: Descend from top layer to layer 1, finding nearest neighbor
        for layer_idx in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer_single(&tables, current_nearest, layer_idx)?;
        }

        // Phase 2: Search at layer 0 with ef_search
        let ef = self.params.ef_search.max(k);
        let candidates = self.search_layer(&tables, vec![current_nearest], ef, 0)?;

        // Optional: Rerank with full precision if available
        let mut results: Vec<(DocumentId, f32)> = candidates
            .into_iter()
            .map(|(doc_id, pq_dist)| {
                // Try to get full precision distance
                if let Some(full_vec) = self.vectors.get_full_precision(doc_id) {
                    let dist = euclidean_distance(query, &full_vec);
                    (doc_id, dist)
                } else {
                    (doc_id, pq_dist)
                }
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Mark a document as deleted (soft delete)
    pub fn delete(&self, doc_id: DocumentId) {
        self.deleted.write().insert(doc_id);
    }

    /// Check if a document is deleted
    pub fn is_deleted(&self, doc_id: DocumentId) -> bool {
        self.deleted.read().contains(&doc_id)
    }

    /// Get delete count
    pub fn deleted_count(&self) -> usize {
        self.deleted.read().len()
    }

    /// Get layer count
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get nodes at a specific layer
    pub fn layer_size(&self, layer: usize) -> usize {
        self.layers.get(layer).map(|l| l.len()).unwrap_or(0)
    }

    /// Search with custom ef_search (for adaptive filtering)
    ///
    /// ef = min(max_ef, base_ef + α * log2(filtered_count + 1))
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(DocumentId, f32)>> {
        if self.entry_point.is_none() {
            return Ok(Vec::new());
        }

        if !self.vectors.is_trained() {
            return Err(SquidexError::Internal(
                "HNSW search requires trained codebooks".to_string(),
            ));
        }

        let tables = self.vectors.build_distance_table(query)?;

        let entry_point = self.entry_point.unwrap();
        let mut current_nearest = entry_point;

        // Descend from top layer to layer 1
        for layer_idx in (1..=self.max_layer).rev() {
            current_nearest = self.search_layer_single(&tables, current_nearest, layer_idx)?;
        }

        // Search at layer 0 with custom ef_search
        let ef = ef_search.max(k);
        let candidates = self.search_layer(&tables, vec![current_nearest], ef, 0)?;

        // Rerank with full precision if available
        let mut results: Vec<(DocumentId, f32)> = candidates
            .into_iter()
            .map(|(doc_id, pq_dist)| {
                if let Some(full_vec) = self.vectors.get_full_precision(doc_id) {
                    let dist = euclidean_distance(query, &full_vec);
                    (doc_id, dist)
                } else {
                    (doc_id, pq_dist)
                }
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
    }

    /// Brute force search using PQ ADC (for small filtered sets)
    ///
    /// Use when filtered_count <= brute_force_threshold
    pub fn brute_force_search(&self, query: &[f32], k: usize) -> Result<Vec<(DocumentId, f32)>> {
        self.vectors.search(query, k)
    }

    /// Check if should use brute force based on candidate count
    ///
    ///  brute_force when filtered_count <= threshold
    pub fn should_use_brute_force(&self, candidate_count: usize, threshold: usize) -> bool {
        candidate_count <= threshold
    }

    /// Get the configured params for adaptive ef calculation
    pub fn compute_adaptive_ef(&self, filtered_count: usize) -> usize {
        // ef = min(max_ef, base_ef + α * log2(filtered_count + 1))
        const BASE_EF: usize = 50;
        const ALPHA: f64 = 10.0;
        const MAX_EF: usize = 400;

        let log_factor = ((filtered_count + 1) as f64).log2();
        let computed = BASE_EF + (ALPHA * log_factor) as usize;
        computed.min(MAX_EF)
    }

    /// Create a snapshot for persistence
    pub fn create_snapshot(&self) -> HnswSnapshot {
        // Serialize layers
        let layers_data: Vec<LayerSnapshot> = self
            .layers
            .iter()
            .map(|layer| LayerSnapshot {
                adjacency: layer.adjacency.clone(),
            })
            .collect();

        HnswSnapshot {
            layers: layers_data,
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            node_levels: self.node_levels.clone(),
            node_count: self.node_count,
            deleted: self.deleted.read().clone(),
            vector_store: self.vectors.create_snapshot(),
            params: self.params.clone(),
        }
    }

    /// Restore from a snapshot
    pub fn restore_from_snapshot(snapshot: HnswSnapshot, cache_size: usize) -> Self {
        let layers: Vec<Layer> = snapshot
            .layers
            .into_iter()
            .map(|layer_snap| Layer {
                adjacency: layer_snap.adjacency,
            })
            .collect();

        let vectors =
            QuantizedVectorStore::restore_from_snapshot(snapshot.vector_store, cache_size);

        Self {
            layers,
            entry_point: snapshot.entry_point,
            max_layer: snapshot.max_layer,
            node_levels: snapshot.node_levels,
            vectors,
            params: snapshot.params,
            deleted: RwLock::new(snapshot.deleted),
            node_count: snapshot.node_count,
        }
    }

    /// Check if auto-training should be triggered
    pub fn should_auto_train(&self, min_vectors: usize) -> bool {
        self.vectors.should_auto_train(min_vectors)
    }

    /// Perform auto-training using buffered vectors
    pub fn auto_train(&mut self) -> Result<bool> {
        self.vectors.auto_train()
    }

    /// Store a vector (delegates to underlying vector store)
    pub fn store(&mut self, doc_id: DocumentId, vector: &[f32]) -> Result<()> {
        self.vectors.store(doc_id, vector)
    }

    /// Remove from underlying vector store (in addition to delete bitset)
    pub fn remove(&mut self, doc_id: DocumentId) -> bool {
        self.deleted.write().insert(doc_id);
        self.vectors.remove(doc_id)
    }

    /// Get the number of buffered vectors awaiting training
    pub fn buffered_count(&self) -> usize {
        self.vectors.buffered_count()
    }
}

/// Snapshot of a single layer for serialization
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LayerSnapshot {
    pub adjacency: HashMap<DocumentId, Vec<(DocumentId, f32)>>,
}

/// Serializable snapshot of the HNSW index
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HnswSnapshot {
    pub layers: Vec<LayerSnapshot>,
    pub entry_point: Option<DocumentId>,
    pub max_layer: usize,
    pub node_levels: HashMap<DocumentId, usize>,
    pub node_count: usize,
    pub deleted: HashSet<DocumentId>,
    pub vector_store: crate::vector::VectorStoreSnapshot,
    pub params: HnswParams,
}

// Make HnswParams serializable
impl serde::Serialize for HnswParams {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("HnswParams", 6)?;
        state.serialize_field("max_connections", &self.max_connections)?;
        state.serialize_field("max_connections_layer0", &self.max_connections_layer0)?;
        state.serialize_field("ef_construction", &self.ef_construction)?;
        state.serialize_field("ef_search", &self.ef_search)?;
        state.serialize_field("level_multiplier", &self.level_multiplier)?;
        state.serialize_field("level_seed", &self.level_seed)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for HnswParams {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct HnswParamsData {
            max_connections: usize,
            max_connections_layer0: usize,
            ef_construction: usize,
            ef_search: usize,
            level_multiplier: f64,
            level_seed: u64,
        }

        let data = HnswParamsData::deserialize(deserializer)?;
        Ok(HnswParams {
            max_connections: data.max_connections,
            max_connections_layer0: data.max_connections_layer0,
            ef_construction: data.ef_construction,
            ef_search: data.ef_search,
            level_multiplier: data.level_multiplier,
            level_seed: data.level_seed,
        })
    }
}

/// Compute Euclidean distance between two vectors
#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
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
    fn test_hnsw_creation() {
        let hnsw = HnswIndex::with_defaults(16, 4);
        assert!(hnsw.is_empty());
        assert_eq!(hnsw.len(), 0);
        assert!(!hnsw.is_trained());
    }

    #[test]
    fn test_deterministic_levels() {
        let hnsw = HnswIndex::with_defaults(16, 4);

        // Same doc_id should always get same level
        let level1 = hnsw.compute_level(42);
        let level2 = hnsw.compute_level(42);
        assert_eq!(level1, level2);

        // Different doc_ids may get different levels
        let _level3 = hnsw.compute_level(43);
        // Just verify it runs without panic
    }

    #[test]
    fn test_insert_single() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        // Train first
        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        // Insert one vector
        let vec = create_random_vectors(1, 16).pop().unwrap();
        hnsw.insert(1, &vec).unwrap();

        assert_eq!(hnsw.len(), 1);
        assert!(hnsw.entry_point.is_some());
    }

    #[test]
    fn test_insert_multiple() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        let vectors = create_random_vectors(100, 16);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(i as u64, vec).unwrap();
        }

        assert_eq!(hnsw.len(), 100);
        assert!(hnsw.layer_count() >= 1);
    }

    #[test]
    fn test_search_basic() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        // Insert vectors
        let vectors = create_random_vectors(100, 16);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(i as u64, vec).unwrap();
        }

        // Search for nearest to first vector
        let results = hnsw.search(&vectors[0], 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should be the query itself (or very close)
        assert_eq!(results[0].0, 0);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_delete() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        let vectors = create_random_vectors(10, 16);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(i as u64, vec).unwrap();
        }

        // Delete a node
        hnsw.delete(5);
        assert!(hnsw.is_deleted(5));
        assert_eq!(hnsw.deleted_count(), 1);

        // Search should not return deleted node
        let results = hnsw.search(&vectors[5], 10).unwrap();
        assert!(!results.iter().any(|(id, _)| *id == 5));
    }

    #[test]
    fn test_recall() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        // Insert vectors
        let vectors = create_random_vectors(200, 16);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(i as u64, vec).unwrap();
        }

        // Compute brute force ground truth for a query
        let query = &vectors[50];
        let mut brute_force: Vec<(u64, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64, euclidean_distance(query, v)))
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: HashSet<u64> = brute_force.iter().take(10).map(|(id, _)| *id).collect();

        // HNSW search
        let hnsw_results = hnsw.search(query, 10).unwrap();
        let hnsw_ids: HashSet<u64> = hnsw_results.iter().map(|(id, _)| *id).collect();

        // Compute recall
        let intersection = ground_truth.intersection(&hnsw_ids).count();
        let recall = intersection as f32 / ground_truth.len() as f32;

        // We expect reasonable recall (>= 70% for this simple test)
        assert!(
            recall >= 0.7,
            "Recall {} is too low (expected >= 0.7)",
            recall
        );
    }

    #[test]
    fn test_empty_search() {
        let hnsw = HnswIndex::with_defaults(16, 4);
        let query = vec![0.0; 16];
        let results = hnsw.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_layer_structure() {
        let mut hnsw = HnswIndex::with_defaults(16, 4);

        let training = create_random_vectors(500, 16);
        hnsw.train(&training).unwrap();

        // Insert many vectors to ensure multiple layers
        let vectors = create_random_vectors(500, 16);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(i as u64, vec).unwrap();
        }

        // Should have multiple layers
        assert!(hnsw.layer_count() >= 1);

        // Layer 0 should have all nodes
        assert_eq!(hnsw.layer_size(0), 500);

        // Higher layers should have fewer nodes
        for l in 1..hnsw.layer_count() {
            assert!(hnsw.layer_size(l) <= hnsw.layer_size(l - 1));
        }
    }
}
