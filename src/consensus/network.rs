use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use openraft::error::{RPCError, RaftError, Unreachable};
use openraft::network::{RaftNetwork, RaftNetworkFactory};
use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    VoteRequest, VoteResponse,
};
use openraft::BasicNode;
use tonic::transport::Channel;

use super::types::{NodeId, TypeConfig};
use crate::consensus::proto;
use crate::consensus::proto::raft_service_client::RaftServiceClient;

/// Network implementation for Squidex Raft cluster
#[derive(Clone)]
pub struct SquidexNetwork {
    /// Node addresses
    pub peers: Arc<DashMap<NodeId, String>>,

    /// gRPC client cache
    clients: Arc<DashMap<NodeId, RaftServiceClient<Channel>>>,
}

impl SquidexNetwork {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(DashMap::new()),
            clients: Arc::new(DashMap::new()),
        }
    }

    pub fn with_peers(peers: Vec<(NodeId, String)>) -> Self {
        let network = Self::new();
        for (id, addr) in peers {
            network.peers.insert(id, addr);
        }
        network
    }

    /// Add or update a peer address
    pub fn add_peer(&self, node_id: NodeId, addr: String) {
        self.peers.insert(node_id, addr);
        // Invalidate cached client
        self.clients.remove(&node_id);
    }

    /// Remove a peer
    pub fn remove_peer(&self, node_id: NodeId) {
        self.peers.remove(&node_id);
        self.clients.remove(&node_id);
    }

    /// Get or create a gRPC client for a target node
    async fn get_client(&self, target: NodeId) -> Result<RaftServiceClient<Channel>, Unreachable> {
        // Check cache first
        if let Some(client) = self.clients.get(&target) {
            return Ok(client.clone());
        }

        // Get peer address
        let addr = self
            .peers
            .get(&target)
            .ok_or_else(|| {
                Unreachable::new(&std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("peer {} not found", target),
                ))
            })?
            .clone();

        // Create new client
        let endpoint = Channel::from_shared(format!("http://{}", addr))
            .map_err(|e| {
                Unreachable::new(&std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    e.to_string(),
                ))
            })?
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .tcp_keepalive(Some(Duration::from_secs(30)));

        let channel = endpoint.connect().await.map_err(|e| {
            Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                e.to_string(),
            ))
        })?;

        let client = RaftServiceClient::new(channel);

        // Cache it
        self.clients.insert(target, client.clone());

        Ok(client)
    }
}

impl Default for SquidexNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection to a specific Raft node
pub struct SquidexConnection {
    target: NodeId,
    #[allow(dead_code)]
    target_node: BasicNode,
    network: SquidexNetwork,
}

impl RaftNetworkFactory<TypeConfig> for SquidexNetwork {
    type Network = SquidexConnection;

    async fn new_client(&mut self, target: NodeId, node: &BasicNode) -> Self::Network {
        // Add peer to network if not already present
        self.peers.insert(target, node.addr.clone());

        SquidexConnection {
            target,
            target_node: node.clone(),
            network: self.clone(),
        }
    }
}

impl RaftNetwork<TypeConfig> for SquidexConnection {
    async fn append_entries(
        &mut self,
        rpc: AppendEntriesRequest<TypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<AppendEntriesResponse<NodeId>, RPCError<NodeId, BasicNode, RaftError<NodeId>>> {
        let mut client = self
            .network
            .get_client(self.target)
            .await
            .map_err(RPCError::Unreachable)?;

        // Convert to proto
        let request = tonic::Request::new(to_proto_append_entries(&rpc));

        let response = client.append_entries(request).await.map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))
        })?;

        // Convert from proto
        from_proto_append_entries_response(response.into_inner()).map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e,
            )))
        })
    }

    async fn install_snapshot(
        &mut self,
        rpc: InstallSnapshotRequest<TypeConfig>,
        _option: openraft::network::RPCOption,
    ) -> Result<
        InstallSnapshotResponse<NodeId>,
        RPCError<NodeId, BasicNode, RaftError<NodeId, openraft::error::InstallSnapshotError>>,
    > {
        let mut client = self
            .network
            .get_client(self.target)
            .await
            .map_err(RPCError::Unreachable)?;

        // Convert to proto
        let request = tonic::Request::new(to_proto_install_snapshot(&rpc));

        let response = client.install_snapshot(request).await.map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))
        })?;

        // Convert from proto
        from_proto_install_snapshot_response(response.into_inner()).map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e,
            )))
        })
    }

    async fn vote(
        &mut self,
        rpc: VoteRequest<NodeId>,
        _option: openraft::network::RPCOption,
    ) -> Result<VoteResponse<NodeId>, RPCError<NodeId, BasicNode, RaftError<NodeId>>> {
        let mut client = self
            .network
            .get_client(self.target)
            .await
            .map_err(RPCError::Unreachable)?;

        // Convert to proto
        let request = tonic::Request::new(to_proto_vote_request(&rpc));

        let response = client.vote(request).await.map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))
        })?;

        // Convert from proto
        from_proto_vote_response(response.into_inner()).map_err(|e| {
            RPCError::Unreachable(Unreachable::new(&std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e,
            )))
        })
    }
}

// Proto conversion helpers

fn to_proto_log_id(log_id: Option<openraft::LogId<NodeId>>) -> Option<proto::LogId> {
    log_id.map(|l| proto::LogId {
        term: l.leader_id.term,
        node_id: l.leader_id.node_id,
        index: l.index,
    })
}

fn from_proto_log_id(log_id: Option<proto::LogId>) -> Option<openraft::LogId<NodeId>> {
    log_id
        .map(|l| openraft::LogId::new(openraft::CommittedLeaderId::new(l.term, l.node_id), l.index))
}

fn to_proto_vote(vote: openraft::Vote<NodeId>) -> proto::VoteData {
    proto::VoteData {
        leader_id: vote.leader_id().node_id,
        committed: vote.is_committed(),
    }
}

fn from_proto_vote(vote: Option<proto::VoteData>) -> openraft::Vote<NodeId> {
    match vote {
        Some(v) => {
            if v.committed {
                openraft::Vote::new_committed(v.leader_id, v.leader_id)
            } else {
                openraft::Vote::new(v.leader_id, v.leader_id)
            }
        }
        None => openraft::Vote::new(0, 0),
    }
}

fn to_proto_append_entries(rpc: &AppendEntriesRequest<TypeConfig>) -> proto::AppendEntriesRequest {
    let entries: Vec<proto::Entry> = rpc
        .entries
        .iter()
        .map(|e| proto::Entry {
            log_id: to_proto_log_id(Some(e.log_id)),
            payload: bincode::serialize(&e.payload).unwrap_or_default(),
        })
        .collect();

    proto::AppendEntriesRequest {
        vote: Some(to_proto_vote(rpc.vote)),
        prev_log_id: to_proto_log_id(rpc.prev_log_id),
        entries,
        leader_commit: to_proto_log_id(rpc.leader_commit),
    }
}

fn from_proto_append_entries_response(
    resp: proto::AppendEntriesResponse,
) -> Result<AppendEntriesResponse<NodeId>, String> {
    // AppendEntriesResponse is an enum in OpenRaft 0.9
    if resp.success {
        Ok(AppendEntriesResponse::Success)
    } else if from_proto_log_id(resp.conflict).is_some() {
        Ok(AppendEntriesResponse::Conflict)
    } else {
        // Check if higher vote
        let vote = from_proto_vote(resp.vote);
        Ok(AppendEntriesResponse::HigherVote(vote))
    }
}

fn to_proto_vote_request(rpc: &VoteRequest<NodeId>) -> proto::VoteRequest {
    proto::VoteRequest {
        vote: Some(to_proto_vote(rpc.vote)),
        last_log_id: to_proto_log_id(rpc.last_log_id),
    }
}

fn from_proto_vote_response(resp: proto::VoteResponse) -> Result<VoteResponse<NodeId>, String> {
    Ok(VoteResponse {
        vote: from_proto_vote(resp.vote),
        vote_granted: resp.vote_granted,
        last_log_id: from_proto_log_id(resp.last_log_id),
    })
}

fn to_proto_install_snapshot(
    rpc: &InstallSnapshotRequest<TypeConfig>,
) -> proto::InstallSnapshotRequest {
    proto::InstallSnapshotRequest {
        vote: Some(to_proto_vote(rpc.vote)),
        meta: Some(proto::SnapshotMeta {
            last_log_id: to_proto_log_id(rpc.meta.last_log_id),
            last_membership: bincode::serialize(&rpc.meta.last_membership).unwrap_or_default(),
            snapshot_id: rpc.meta.snapshot_id.clone(),
        }),
        offset: rpc.offset,
        data: rpc.data.clone(),
        done: rpc.done,
    }
}

fn from_proto_install_snapshot_response(
    resp: proto::InstallSnapshotResponse,
) -> Result<InstallSnapshotResponse<NodeId>, String> {
    Ok(InstallSnapshotResponse {
        vote: from_proto_vote(resp.vote),
    })
}
