use prometheus::{
    Counter, CounterVec, Gauge, Histogram, HistogramOpts, HistogramVec, Opts, Registry,
};
use std::sync::Arc;

/// Prometheus metrics for the search engine
#[derive(Clone)]
pub struct SearchMetrics {
    // Counters
    pub documents_indexed: Counter,
    pub documents_deleted: Counter,
    pub documents_updated: Counter,
    pub searches_total: CounterVec,
    pub search_errors: Counter,

    // Gauges
    pub total_documents: Gauge,
    pub index_size_bytes: Gauge,
    pub cluster_leader: Gauge,
    pub cluster_size: Gauge,

    // Histograms
    pub index_latency: Histogram,
    pub search_latency: HistogramVec,
    pub batch_size: Histogram,

    // Registry
    registry: Arc<Registry>,
}

impl SearchMetrics {
    /// Create a new SearchMetrics instance
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // Counters
        let documents_indexed = Counter::with_opts(Opts::new(
            "squidex_documents_indexed_total",
            "Total number of documents indexed",
        ))?;
        registry.register(Box::new(documents_indexed.clone()))?;

        let documents_deleted = Counter::with_opts(Opts::new(
            "squidex_documents_deleted_total",
            "Total number of documents deleted",
        ))?;
        registry.register(Box::new(documents_deleted.clone()))?;

        let documents_updated = Counter::with_opts(Opts::new(
            "squidex_documents_updated_total",
            "Total number of documents updated",
        ))?;
        registry.register(Box::new(documents_updated.clone()))?;

        let searches_total = CounterVec::new(
            Opts::new("squidex_searches_total", "Total number of searches by type"),
            &["type"],
        )?;
        registry.register(Box::new(searches_total.clone()))?;

        let search_errors = Counter::with_opts(Opts::new(
            "squidex_search_errors_total",
            "Total number of search errors",
        ))?;
        registry.register(Box::new(search_errors.clone()))?;

        // Gauges
        let total_documents = Gauge::with_opts(Opts::new(
            "squidex_total_documents",
            "Current number of documents in the index",
        ))?;
        registry.register(Box::new(total_documents.clone()))?;

        let index_size_bytes = Gauge::with_opts(Opts::new(
            "squidex_index_size_bytes",
            "Estimated index size in bytes",
        ))?;
        registry.register(Box::new(index_size_bytes.clone()))?;

        let cluster_leader = Gauge::with_opts(Opts::new(
            "squidex_cluster_leader",
            "1 if this node is the leader, 0 otherwise",
        ))?;
        registry.register(Box::new(cluster_leader.clone()))?;

        let cluster_size = Gauge::with_opts(Opts::new(
            "squidex_cluster_size",
            "Number of nodes in the cluster",
        ))?;
        registry.register(Box::new(cluster_size.clone()))?;

        // Histograms
        let index_latency = Histogram::with_opts(
            HistogramOpts::new("squidex_index_latency_seconds", "Index operation latency")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        )?;
        registry.register(Box::new(index_latency.clone()))?;

        let search_latency = HistogramVec::new(
            HistogramOpts::new("squidex_search_latency_seconds", "Search operation latency")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["type"],
        )?;
        registry.register(Box::new(search_latency.clone()))?;

        let batch_size = Histogram::with_opts(
            HistogramOpts::new(
                "squidex_batch_size",
                "Number of documents in batch operations",
            )
            .buckets(vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]),
        )?;
        registry.register(Box::new(batch_size.clone()))?;

        Ok(Self {
            documents_indexed,
            documents_deleted,
            documents_updated,
            searches_total,
            search_errors,
            total_documents,
            index_size_bytes,
            cluster_leader,
            cluster_size,
            index_latency,
            search_latency,
            batch_size,
            registry: Arc::new(registry),
        })
    }

    /// Get the Prometheus registry
    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    /// Record an index operation
    pub fn record_index(&self, duration_secs: f64) {
        self.documents_indexed.inc();
        self.index_latency.observe(duration_secs);
    }

    /// Record a delete operation
    pub fn record_delete(&self) {
        self.documents_deleted.inc();
    }

    /// Record an update operation
    pub fn record_update(&self, duration_secs: f64) {
        self.documents_updated.inc();
        self.index_latency.observe(duration_secs);
    }

    /// Record a search operation
    pub fn record_search(&self, search_type: &str, duration_secs: f64) {
        self.searches_total.with_label_values(&[search_type]).inc();
        self.search_latency
            .with_label_values(&[search_type])
            .observe(duration_secs);
    }

    /// Record a search error
    pub fn record_search_error(&self) {
        self.search_errors.inc();
    }

    /// Update total documents gauge
    pub fn set_total_documents(&self, count: u64) {
        self.total_documents.set(count as f64);
    }

    /// Update index size gauge
    pub fn set_index_size(&self, size_bytes: u64) {
        self.index_size_bytes.set(size_bytes as f64);
    }

    /// Update cluster leader status
    pub fn set_is_leader(&self, is_leader: bool) {
        self.cluster_leader.set(if is_leader { 1.0 } else { 0.0 });
    }

    /// Update cluster size
    pub fn set_cluster_size(&self, size: usize) {
        self.cluster_size.set(size as f64);
    }

    /// Record batch operation size
    pub fn record_batch(&self, size: usize) {
        self.batch_size.observe(size as f64);
    }
}

impl Default for SearchMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics")
    }
}
