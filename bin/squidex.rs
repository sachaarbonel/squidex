use anyhow::Result;
use clap::Parser;
use squidex::consensus::proto::raft_service_server::RaftServiceServer;
use squidex::{
    IndexSettings, NodeConfig, PerformanceProfile, RaftServiceImpl, SearchStateMachine, SquidexNode,
};
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "squidex")]
#[command(about = "Distributed keyword & vector search engine", long_about = None)]
struct Args {
    /// Node ID (must be unique in cluster)
    #[arg(long, env = "SQUIDEX_NODE_ID")]
    node_id: u64,

    /// Bind address for Raft gRPC (inter-node communication)
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

    /// HTTP API port
    #[arg(long, env = "SQUIDEX_HTTP_PORT", default_value = "8080")]
    http_port: u16,
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
            warn!("Unknown profile '{}', using 'balanced'", args.profile);
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
    info!("  Initial leader: {}", node_config.is_initial_leader);

    // Create index settings
    let mut index_settings = IndexSettings::default();
    index_settings.vector_dimensions = args.vector_dimensions;

    info!("Index settings:");
    info!("  Vector dimensions: {}", index_settings.vector_dimensions);
    info!(
        "  Similarity metric: {:?}",
        index_settings.similarity_metric
    );

    // Create state machine
    let state_machine = Arc::new(SearchStateMachine::new(
        index_settings,
        node_config.data_dir.clone(),
    )?);
    info!("Search state machine initialized");

    // Create Squidex node (OpenRaft-based)
    let node =
        Arc::new(SquidexNode::new(args.node_id, node_config.clone(), state_machine.clone()).await?);
    info!("Raft node created");

    // Start Raft node
    node.start().await?;
    info!("Raft node started");

    // Start gRPC server for Raft communication
    let grpc_addr = args.bind_addr.parse()?;
    let raft_service = RaftServiceImpl::new(node.raft.clone());

    let grpc_server = Server::builder()
        .add_service(RaftServiceServer::new(raft_service))
        .serve(grpc_addr);

    tokio::spawn(async move {
        if let Err(e) = grpc_server.await {
            tracing::error!("gRPC server error: {}", e);
        }
    });
    info!("gRPC Raft server started on {}", args.bind_addr);

    // Initialize metrics
    let metrics = Arc::new(squidex::SearchMetrics::new()?);
    info!("Metrics initialized");

    // Start HTTP API server
    let app_state = squidex::AppState {
        node,
        state_machine,
        metrics,
    };

    let app = squidex::create_router(app_state);
    let http_addr = format!("0.0.0.0:{}", args.http_port);
    let listener = tokio::net::TcpListener::bind(&http_addr).await?;
    info!("HTTP API server listening on {}", http_addr);

    info!("Squidex node is ready");

    // Serve HTTP API
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install CTRL+C signal handler");
            info!("Received shutdown signal, gracefully shutting down");
        })
        .await?;

    Ok(())
}
