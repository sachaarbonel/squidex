use std::collections::BTreeSet;
use std::sync::Arc;

use openraft::{BasicNode, Config, Raft, ServerState};
use tracing::info;

use super::network::SquidexNetwork;
use super::storage::{SquidexLogStore, SquidexStateMachine};
use super::types::{LogEntry, NodeId, Request, Response, TypeConfig};
use crate::config::NodeConfig;
use crate::error::{Result, SquidexError};
use crate::state_machine::SearchStateMachine;

/// Alias for the Raft type with our configuration
pub type SquidexRaft = Raft<TypeConfig>;

/// Squidex Raft node for distributed consensus
pub struct SquidexNode {
    /// The Raft instance
    pub raft: Arc<SquidexRaft>,

    /// Node ID
    pub node_id: NodeId,

    /// Node configuration
    pub config: NodeConfig,

    /// Network layer
    pub network: Arc<SquidexNetwork>,

    /// Search state machine
    pub state_machine: Arc<SearchStateMachine>,
}

impl SquidexNode {
    /// Create a new Squidex node
    pub async fn new(
        node_id: NodeId,
        config: NodeConfig,
        state_machine: Arc<SearchStateMachine>,
    ) -> Result<Self> {
        // Configure Raft
        let raft_config = Config {
            cluster_name: "squidex".to_string(),
            heartbeat_interval: 500,  // 500ms
            election_timeout_min: 1500,
            election_timeout_max: 3000,
            max_in_snapshot_log_to_keep: 1000,
            ..Default::default()
        };

        let raft_config = Arc::new(raft_config.validate().map_err(|e| {
            SquidexError::Internal(format!("invalid raft config: {}", e))
        })?);

        // Create data directory
        let data_dir = config.data_dir.join(format!("node{}", node_id));
        std::fs::create_dir_all(&data_dir).map_err(SquidexError::Io)?;

        // Create log store
        let log_store = SquidexLogStore::new(data_dir.clone());

        // Create state machine store
        let sm_store = SquidexStateMachine::new(data_dir.clone(), state_machine.clone());

        // Create network
        let network = Arc::new(SquidexNetwork::new());

        // Add peers to network
        for (idx, peer) in config.peers.iter().enumerate() {
            let peer_id = idx as NodeId + 1;
            if peer_id != node_id {
                network.add_peer(peer_id, peer.clone());
            }
        }

        // Create Raft instance
        let raft = Raft::new(
            node_id,
            raft_config,
            network.as_ref().clone(),
            log_store,
            sm_store,
        )
        .await
        .map_err(|e| SquidexError::Internal(format!("failed to create raft: {}", e)))?;

        let raft = Arc::new(raft);

        Ok(Self {
            raft,
            node_id,
            config,
            network,
            state_machine,
        })
    }

    /// Start the node (initialize cluster if initial leader)
    pub async fn start(&self) -> Result<()> {
        if self.config.is_initial_leader {
            // Bootstrap single-node cluster
            let mut members = std::collections::BTreeMap::new();
            members.insert(
                self.node_id,
                BasicNode {
                    addr: self.config.bind_addr.clone(),
                },
            );

            match self.raft.initialize(members).await {
                Ok(_) => {
                    info!("Bootstrapped cluster as initial leader");
                }
                Err(e) => {
                    // Check if already initialized
                    let err_str = format!("{:?}", e);
                    if err_str.contains("NotAllowed") || err_str.contains("already initialized") {
                        info!("Cluster already initialized, joining as member");
                    } else {
                        return Err(SquidexError::Internal(format!(
                            "failed to initialize cluster: {:?}",
                            e
                        )));
                    }
                }
            }
        } else {
            // Non-leader nodes will be added via membership change
            info!("Node {} waiting to join cluster", self.node_id);
        }

        Ok(())
    }

    /// Propose a write operation (only leader accepts)
    pub async fn propose(&self, entry: LogEntry) -> Result<Response> {
        let request = Request::new(entry);

        let response = self
            .raft
            .client_write(request)
            .await
            .map_err(|e| SquidexError::Consensus(format!("{:?}", e)))?;

        Ok(response.data)
    }

    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        let metrics = self.raft.metrics().borrow().clone();
        matches!(metrics.state, ServerState::Leader)
    }

    /// Get the node ID
    pub fn id(&self) -> NodeId {
        self.node_id
    }

    /// Get current leader ID
    pub async fn leader_id(&self) -> Option<NodeId> {
        let metrics = self.raft.metrics().borrow().clone();
        metrics.current_leader
    }

    /// Get cluster membership info
    pub async fn membership(&self) -> ClusterMembership {
        let metrics = self.raft.metrics().borrow().clone();

        let voters: Vec<NodeId> = metrics
            .membership_config
            .membership()
            .voter_ids()
            .collect();

        let learners: Vec<NodeId> = metrics
            .membership_config
            .membership()
            .learner_ids()
            .collect();

        ClusterMembership { voters, learners }
    }

    /// Add a new node to the cluster (must be leader)
    pub async fn add_node(&self, node_id: NodeId, addr: String) -> Result<()> {
        if !self.is_leader().await {
            return Err(SquidexError::NotLeader);
        }

        // Add to network
        self.network.add_peer(node_id, addr.clone());

        // Add as learner first
        self.raft
            .add_learner(node_id, BasicNode { addr }, true)
            .await
            .map_err(|e| SquidexError::Consensus(format!("failed to add learner: {:?}", e)))?;

        // Promote to voter - collect current voters + new node
        let membership = self.membership().await;
        let mut new_voters: BTreeSet<NodeId> = membership.voters.into_iter().collect();
        new_voters.insert(node_id);

        self.raft
            .change_membership(new_voters, false)
            .await
            .map_err(|e| SquidexError::Consensus(format!("failed to change membership: {:?}", e)))?;

        info!("Added node {} to cluster", node_id);
        Ok(())
    }

    /// Remove a node from the cluster (must be leader)
    pub async fn remove_node(&self, node_id: NodeId) -> Result<()> {
        if !self.is_leader().await {
            return Err(SquidexError::NotLeader);
        }

        let membership = self.membership().await;
        let new_voters: BTreeSet<NodeId> = membership
            .voters
            .into_iter()
            .filter(|&id| id != node_id)
            .collect();

        self.raft
            .change_membership(new_voters, false)
            .await
            .map_err(|e| SquidexError::Consensus(format!("failed to change membership: {:?}", e)))?;

        // Remove from network
        self.network.remove_peer(node_id);

        info!("Removed node {} from cluster", node_id);
        Ok(())
    }

    /// Trigger a snapshot
    pub async fn trigger_snapshot(&self) -> Result<()> {
        self.raft
            .trigger()
            .snapshot()
            .await
            .map_err(|e| SquidexError::Internal(format!("failed to trigger snapshot: {:?}", e)))?;
        Ok(())
    }
}

/// Cluster membership information
#[derive(Clone, Debug)]
pub struct ClusterMembership {
    pub voters: Vec<NodeId>,
    pub learners: Vec<NodeId>,
}
