//! Consensus module - OpenRaft-based distributed consensus
//!
//! This module provides distributed consensus for the Squidex search engine
//! using OpenRaft with gRPC for inter-node communication.

pub mod grpc_service;
pub mod network;
pub mod node;
pub mod storage;
pub mod types;

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("squidex.raft");
}

pub use grpc_service::RaftServiceImpl;
pub use network::SquidexNetwork;
pub use node::{ClusterMembership, SquidexNode, SquidexRaft};
pub use types::{LogEntry, NodeId, Request, Response, TypeConfig};
