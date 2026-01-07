use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::models::SimilarityMetric;

/// Index settings configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexSettings {
    pub vector_dimensions: usize,
    pub similarity_metric: SimilarityMetric,
    pub tokenizer_config: TokenizerConfig,
}

impl Default for IndexSettings {
    fn default() -> Self {
        Self {
            vector_dimensions: 384, // Default for many embedding models
            similarity_metric: SimilarityMetric::Cosine,
            tokenizer_config: TokenizerConfig::default(),
        }
    }
}

/// Tokenizer configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub lowercase: bool,
    pub remove_stopwords: bool,
    pub stem: bool,
    pub min_token_length: usize,
    pub max_token_length: usize,
    pub language: String,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            stem: true,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        }
    }
}

/// Node configuration for Squidex cluster
#[derive(Clone, Debug)]
pub struct NodeConfig {
    pub node_id: u64,
    pub bind_addr: String,
    pub peers: Vec<String>,
    pub data_dir: PathBuf,
    pub is_initial_leader: bool,
    pub worker_threads: usize,
    pub wal_batch_size: usize,
    pub wal_flush_interval_ms: u64,
    pub snapshot_lag_threshold: u64,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: 1,
            bind_addr: "127.0.0.1:5001".to_string(),
            peers: Vec::new(),
            data_dir: PathBuf::from("./data"),
            is_initial_leader: false,
            worker_threads: num_cpus::get(),
            wal_batch_size: 100,
            wal_flush_interval_ms: 50,
            snapshot_lag_threshold: 10_000,
        }
    }
}

/// Configuration profiles for different workloads
#[derive(Clone, Debug)]
pub enum PerformanceProfile {
    LowLatency,
    Balanced,
    HighThroughput,
    Durable,
}

impl PerformanceProfile {
    /// Get WAL batch size for this profile
    pub fn wal_batch_size(&self) -> usize {
        match self {
            PerformanceProfile::LowLatency => 10,
            PerformanceProfile::Balanced => 100,
            PerformanceProfile::HighThroughput => 1000,
            PerformanceProfile::Durable => 1,
        }
    }

    /// Get WAL flush interval for this profile
    pub fn wal_flush_interval_ms(&self) -> u64 {
        match self {
            PerformanceProfile::LowLatency => 10,
            PerformanceProfile::Balanced => 50,
            PerformanceProfile::HighThroughput => 200,
            PerformanceProfile::Durable => 0, // Sync each write
        }
    }

    /// Apply this profile to a NodeConfig
    pub fn apply_to(&self, config: &mut NodeConfig) {
        config.wal_batch_size = self.wal_batch_size();
        config.wal_flush_interval_ms = self.wal_flush_interval_ms();
    }
}

impl NodeConfig {
    /// Create a new node configuration
    pub fn new(
        node_id: u64,
        bind_addr: String,
        peers: Vec<String>,
        data_dir: PathBuf,
        is_initial_leader: bool,
    ) -> Self {
        Self {
            node_id,
            bind_addr,
            peers,
            data_dir,
            is_initial_leader,
            ..Default::default()
        }
    }

    /// Apply a performance profile to this configuration
    pub fn with_profile(mut self, profile: PerformanceProfile) -> Self {
        profile.apply_to(&mut self);
        self
    }

    /// Set the number of worker threads
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads;
        self
    }

    /// Set the snapshot lag threshold
    pub fn with_snapshot_lag_threshold(mut self, threshold: u64) -> Self {
        self.snapshot_lag_threshold = threshold;
        self
    }

    /// Get the WAL directory for this node
    pub fn wal_dir(&self) -> PathBuf {
        self.data_dir.join(format!("node{}", self.node_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs() {
        let index_settings = IndexSettings::default();
        assert_eq!(index_settings.vector_dimensions, 384);

        let tokenizer_config = TokenizerConfig::default();
        assert!(tokenizer_config.lowercase);
        assert!(tokenizer_config.remove_stopwords);

        let node_config = NodeConfig::default();
        assert_eq!(node_config.wal_batch_size, 100);
    }

    #[test]
    fn test_performance_profiles() {
        assert_eq!(PerformanceProfile::LowLatency.wal_batch_size(), 10);
        assert_eq!(PerformanceProfile::LowLatency.wal_flush_interval_ms(), 10);

        assert_eq!(PerformanceProfile::Durable.wal_batch_size(), 1);
        assert_eq!(PerformanceProfile::Durable.wal_flush_interval_ms(), 0);
    }

    #[test]
    fn test_node_config_builder() {
        let config = NodeConfig::new(
            1,
            "127.0.0.1:5001".to_string(),
            vec!["127.0.0.1:5002".to_string()],
            PathBuf::from("./data"),
            true,
        )
        .with_profile(PerformanceProfile::LowLatency)
        .with_worker_threads(8);

        assert_eq!(config.node_id, 1);
        assert_eq!(config.worker_threads, 8);
        assert_eq!(config.wal_batch_size, 10);
        assert_eq!(config.wal_dir(), PathBuf::from("./data/node1"));
    }
}
