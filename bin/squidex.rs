use anyhow::Result;
use clap::Parser;
use squidex::{IndexSettings, NodeConfig, PerformanceProfile, SearchStateMachine};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "squidex")]
#[command(about = "Distributed keyword & vector search engine", long_about = None)]
struct Args {
    /// Node ID (must be unique in cluster)
    #[arg(long, env = "SQUIDEX_NODE_ID")]
    node_id: u64,

    /// Bind address for this node
    #[arg(long, env = "SQUIDEX_BIND_ADDR", default_value = "127.0.0.1:5001")]
    bind_addr: String,

    /// Comma-separated list of peer addresses
    #[arg(long, env = "SQUIDEX_PEERS", value_delimiter = ',')]
    peers: Vec<String>,

    /// Data directory for WAL and snapshots
    #[arg(long, env = "SQUIDEX_DATA_DIR", default_value = "./data")]
    data_dir: PathBuf,

    /// Whether this node is the initial leader
    #[arg(long, env = "SQUIDEX_INITIAL_LEADER")]
    is_initial_leader: bool,

    /// Performance profile (low-latency, balanced, high-throughput, durable)
    #[arg(long, env = "SQUIDEX_PROFILE", default_value = "balanced")]
    profile: String,

    /// Vector embedding dimensions
    #[arg(long, env = "SQUIDEX_VECTOR_DIM", default_value = "384")]
    vector_dimensions: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!(
        "Starting Squidex v{} - Node {}",
        squidex::VERSION,
        args.node_id
    );

    // Parse performance profile
    let profile = match args.profile.to_lowercase().as_str() {
        "low-latency" | "low_latency" => PerformanceProfile::LowLatency,
        "balanced" => PerformanceProfile::Balanced,
        "high-throughput" | "high_throughput" => PerformanceProfile::HighThroughput,
        "durable" => PerformanceProfile::Durable,
        _ => {
            warn!(
                "Unknown profile '{}', using 'balanced'",
                args.profile
            );
            PerformanceProfile::Balanced
        }
    };

    // Create node configuration
    let node_config = NodeConfig::new(
        args.node_id,
        args.bind_addr.clone(),
        args.peers.clone(),
        args.data_dir.clone(),
        args.is_initial_leader,
    )
    .with_profile(profile);

    info!("Node configuration:");
    info!("  Node ID: {}", node_config.node_id);
    info!("  Bind address: {}", node_config.bind_addr);
    info!("  Peers: {:?}", node_config.peers);
    info!("  Data directory: {:?}", node_config.data_dir);
    info!("  WAL batch size: {}", node_config.wal_batch_size);
    info!(
        "  WAL flush interval: {}ms",
        node_config.wal_flush_interval_ms
    );
    info!("  Initial leader: {}", node_config.is_initial_leader);

    // Create index settings
    let mut index_settings = IndexSettings::default();
    index_settings.vector_dimensions = args.vector_dimensions;

    info!("Index settings:");
    info!("  Vector dimensions: {}", index_settings.vector_dimensions);
    info!("  Similarity metric: {:?}", index_settings.similarity_metric);

    // Create state machine
    let _state_machine = Arc::new(SearchStateMachine::new(index_settings));
    info!("Search state machine initialized");

    // TODO: Initialize Octopii node
    // let config = octopii::Config {
    //     node_id: node_config.node_id,
    //     bind_addr: node_config.bind_addr.parse()?,
    //     peers: node_config.peers.iter().map(|p| p.parse()).collect::<Result<Vec<_>, _>>()?,
    //     wal_dir: node_config.wal_dir(),
    //     worker_threads: node_config.worker_threads,
    //     wal_batch_size: node_config.wal_batch_size,
    //     wal_flush_interval_ms: node_config.wal_flush_interval_ms,
    //     is_initial_leader: node_config.is_initial_leader,
    //     snapshot_lag_threshold: node_config.snapshot_lag_threshold,
    // };
    //
    // let node = octopii::OctopiiNode::new(config, state_machine.clone()).await?;
    // info!("Octopii node initialized");

    // TODO: Start HTTP API server
    // let app_state = AppState {
    //     node: Arc::new(node),
    //     state_machine,
    // };
    //
    // let app = create_router(app_state);
    // let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    // info!("HTTP API server listening on 0.0.0.0:8080");
    //
    // axum::serve(listener, app).await?;

    info!("Squidex node is ready");

    // For now, just keep the process running
    // In full implementation, this would handle signals and graceful shutdown
    tokio::signal::ctrl_c().await?;
    info!("Received shutdown signal, exiting");

    Ok(())
}
