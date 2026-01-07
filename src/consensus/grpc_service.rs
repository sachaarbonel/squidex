use std::sync::Arc;

use openraft::raft::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, VoteRequest,
};
use openraft::{EntryPayload, LogId};
use tonic::{Request, Response, Status};

use super::node::SquidexRaft;
use super::proto::raft_service_server::RaftService;
use super::proto::{
    AppendEntriesRequest as ProtoAppendEntriesRequest,
    AppendEntriesResponse as ProtoAppendEntriesResponse,
    InstallSnapshotRequest as ProtoInstallSnapshotRequest,
    InstallSnapshotResponse as ProtoInstallSnapshotResponse, VoteRequest as ProtoVoteRequest,
    VoteResponse as ProtoVoteResponse,
};
use super::types::{NodeId, TypeConfig};

/// gRPC service implementation for Raft RPCs
pub struct RaftServiceImpl {
    pub raft: Arc<SquidexRaft>,
}

impl RaftServiceImpl {
    pub fn new(raft: Arc<SquidexRaft>) -> Self {
        Self { raft }
    }
}

#[tonic::async_trait]
impl RaftService for RaftServiceImpl {
    async fn append_entries(
        &self,
        request: Request<ProtoAppendEntriesRequest>,
    ) -> Result<Response<ProtoAppendEntriesResponse>, Status> {
        let req = request.into_inner();

        // Convert from proto
        let vote = from_proto_vote(req.vote);
        let prev_log_id = from_proto_log_id(req.prev_log_id);
        let leader_commit = from_proto_log_id(req.leader_commit);

        let entries: Result<Vec<openraft::Entry<TypeConfig>>, Status> = req
            .entries
            .iter()
            .map(|e| {
                let log_id = from_proto_log_id(e.log_id.clone())
                    .ok_or_else(|| Status::invalid_argument("missing log_id"))?;
                let payload: EntryPayload<TypeConfig> = bincode::deserialize(&e.payload)
                    .map_err(|e| Status::invalid_argument(format!("invalid payload: {}", e)))?;
                Ok(openraft::Entry { log_id, payload })
            })
            .collect();

        let entries = entries?;

        let rpc = AppendEntriesRequest {
            vote,
            prev_log_id,
            entries,
            leader_commit,
        };

        let response = self
            .raft
            .append_entries(rpc)
            .await
            .map_err(|e| Status::internal(format!("append_entries failed: {:?}", e)))?;

        // Convert response enum to proto
        let (success, conflict, resp_vote) = match response {
            AppendEntriesResponse::Success => (true, None, vote),
            AppendEntriesResponse::PartialSuccess(log_id) => (true, log_id, vote),
            AppendEntriesResponse::HigherVote(v) => (false, None, v),
            AppendEntriesResponse::Conflict => (false, None, vote),
        };

        Ok(Response::new(ProtoAppendEntriesResponse {
            vote: Some(to_proto_vote(resp_vote)),
            success,
            conflict: to_proto_log_id(conflict),
        }))
    }

    async fn vote(
        &self,
        request: Request<ProtoVoteRequest>,
    ) -> Result<Response<ProtoVoteResponse>, Status> {
        let req = request.into_inner();

        let vote = from_proto_vote(req.vote);
        let last_log_id = from_proto_log_id(req.last_log_id);

        let rpc = VoteRequest { vote, last_log_id };

        let response = self
            .raft
            .vote(rpc)
            .await
            .map_err(|e| Status::internal(format!("vote failed: {:?}", e)))?;

        Ok(Response::new(ProtoVoteResponse {
            vote: Some(to_proto_vote(response.vote)),
            vote_granted: response.vote_granted,
            last_log_id: to_proto_log_id(response.last_log_id),
        }))
    }

    async fn install_snapshot(
        &self,
        request: Request<ProtoInstallSnapshotRequest>,
    ) -> Result<Response<ProtoInstallSnapshotResponse>, Status> {
        let req = request.into_inner();

        let vote = from_proto_vote(req.vote);
        let meta = req
            .meta
            .ok_or_else(|| Status::invalid_argument("missing meta"))?;

        let last_membership = bincode::deserialize(&meta.last_membership)
            .map_err(|e| Status::invalid_argument(format!("invalid membership: {}", e)))?;

        let snapshot_meta = openraft::SnapshotMeta {
            last_log_id: from_proto_log_id(meta.last_log_id),
            last_membership,
            snapshot_id: meta.snapshot_id,
        };

        let rpc = InstallSnapshotRequest {
            vote,
            meta: snapshot_meta,
            offset: req.offset,
            data: req.data,
            done: req.done,
        };

        let response = self
            .raft
            .install_snapshot(rpc)
            .await
            .map_err(|e| Status::internal(format!("install_snapshot failed: {:?}", e)))?;

        Ok(Response::new(ProtoInstallSnapshotResponse {
            vote: Some(to_proto_vote(response.vote)),
        }))
    }
}

// Proto conversion helpers

fn to_proto_log_id(log_id: Option<LogId<NodeId>>) -> Option<super::proto::LogId> {
    log_id.map(|l| super::proto::LogId {
        term: l.leader_id.term,
        node_id: l.leader_id.node_id,
        index: l.index,
    })
}

fn from_proto_log_id(log_id: Option<super::proto::LogId>) -> Option<LogId<NodeId>> {
    log_id.map(|l| LogId::new(openraft::CommittedLeaderId::new(l.term, l.node_id), l.index))
}

fn to_proto_vote(vote: openraft::Vote<NodeId>) -> super::proto::VoteData {
    super::proto::VoteData {
        leader_id: vote.leader_id().node_id,
        committed: vote.is_committed(),
    }
}

fn from_proto_vote(vote: Option<super::proto::VoteData>) -> openraft::Vote<NodeId> {
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
