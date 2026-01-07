use std::collections::BTreeMap;
use std::fmt::Debug;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::path::PathBuf;
use std::sync::Arc;

use openraft::storage::{LogFlushed, LogState, RaftLogStorage, RaftStateMachine, Snapshot};
use openraft::{
    Entry, EntryPayload, LogId, OptionalSend, RaftLogReader, RaftSnapshotBuilder,
    SnapshotMeta, StorageError, StorageIOError, StoredMembership, Vote,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::types::{LogEntry, NodeId, Response, SquidexSnapshot, TypeConfig};
use crate::state_machine::SearchStateMachine;

/// Log store state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogStoreState {
    /// The last purged log id
    pub last_purged_log_id: Option<LogId<NodeId>>,

    /// All log entries
    pub logs: BTreeMap<u64, Entry<TypeConfig>>,

    /// The current vote
    pub vote: Option<Vote<NodeId>>,

    /// Committed vote (for storage)
    pub committed: bool,
}

/// State machine state for persistence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMachineState {
    pub last_applied_log: Option<LogId<NodeId>>,
    pub last_membership: StoredMembership<NodeId, openraft::BasicNode>,
}

/// RocksDB-backed Raft log storage
pub struct SquidexLogStore {
    /// Data directory
    data_dir: PathBuf,

    /// In-memory state (can be persisted to RocksDB)
    state: RwLock<LogStoreState>,
}

impl SquidexLogStore {
    pub fn new(data_dir: PathBuf) -> Self {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(&data_dir).ok();

        // Try to load existing state
        let state_path = data_dir.join("log_state.bin");
        let state = if state_path.exists() {
            match std::fs::read(&state_path) {
                Ok(data) => bincode::deserialize(&data).unwrap_or_default(),
                Err(_) => LogStoreState::default(),
            }
        } else {
            LogStoreState::default()
        };

        Self {
            data_dir,
            state: RwLock::new(state),
        }
    }

    fn persist_state(&self) {
        let state = self.state.read();
        let state_path = self.data_dir.join("log_state.bin");
        if let Ok(data) = bincode::serialize(&*state) {
            let _ = std::fs::write(state_path, data);
        }
    }
}

impl RaftLogReader<TypeConfig> for SquidexLogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>> {
        let state = self.state.read();

        let entries: Vec<_> = state
            .logs
            .range(range)
            .map(|(_, v)| v.clone())
            .collect();

        Ok(entries)
    }
}

impl RaftLogStorage<TypeConfig> for SquidexLogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let state = self.state.read();

        let last_log_id = state.logs.last_key_value().map(|(_, v)| v.log_id);
        let last_purged_log_id = state.last_purged_log_id;

        Ok(LogState {
            last_purged_log_id,
            last_log_id,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        // Return a clone of self's data
        Self {
            data_dir: self.data_dir.clone(),
            state: RwLock::new(self.state.read().clone()),
        }
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        let state = self.state.read();
        Ok(state.vote)
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut state = self.state.write();
        state.vote = Some(*vote);
        drop(state);
        self.persist_state();
        Ok(())
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        let mut state = self.state.write();
        state.committed = committed.is_some();
        drop(state);
        self.persist_state();
        Ok(())
    }

    async fn append<I>(&mut self, entries: I, callback: LogFlushed<TypeConfig>) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut state = self.state.write();

        for entry in entries {
            state.logs.insert(entry.log_id.index, entry);
        }

        drop(state);
        self.persist_state();

        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut state = self.state.write();

        let keys: Vec<_> = state
            .logs
            .range(log_id.index..)
            .map(|(k, _)| *k)
            .collect();

        for key in keys {
            state.logs.remove(&key);
        }

        drop(state);
        self.persist_state();
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut state = self.state.write();

        let keys: Vec<_> = state
            .logs
            .range(..=log_id.index)
            .map(|(k, _)| *k)
            .collect();

        for key in keys {
            state.logs.remove(&key);
        }

        state.last_purged_log_id = Some(log_id);

        drop(state);
        self.persist_state();
        Ok(())
    }
}

/// Raft state machine backed by SearchStateMachine
pub struct SquidexStateMachine {
    /// Data directory
    data_dir: PathBuf,

    /// State machine state
    state: RwLock<StateMachineState>,

    /// The actual search state machine
    search_machine: Arc<SearchStateMachine>,
}

impl SquidexStateMachine {
    pub fn new(data_dir: PathBuf, search_machine: Arc<SearchStateMachine>) -> Self {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(&data_dir).ok();

        // Try to load existing state
        let state_path = data_dir.join("sm_state.bin");
        let state = if state_path.exists() {
            match std::fs::read(&state_path) {
                Ok(data) => bincode::deserialize(&data).unwrap_or_default(),
                Err(_) => StateMachineState::default(),
            }
        } else {
            StateMachineState::default()
        };

        Self {
            data_dir,
            state: RwLock::new(state),
            search_machine,
        }
    }

    fn persist_state(&self) {
        let state = self.state.read();
        let state_path = self.data_dir.join("sm_state.bin");
        if let Ok(data) = bincode::serialize(&*state) {
            let _ = std::fs::write(state_path, data);
        }
    }

    /// Apply a log entry to the search state machine
    fn apply_entry(&self, entry: &LogEntry) -> Response {
        match entry {
            LogEntry::IndexDocument(doc) => {
                match self.search_machine.index_document(doc.clone()) {
                    Ok(_) => Response::success(format!("indexed doc {}", doc.id)),
                    Err(e) => Response::error(format!("failed to index: {}", e)),
                }
            }
            LogEntry::DeleteDocument(doc_id) => {
                match self.search_machine.delete_document(*doc_id) {
                    Ok(_) => Response::success(format!("deleted doc {}", doc_id)),
                    Err(e) => Response::error(format!("failed to delete: {}", e)),
                }
            }
            LogEntry::BatchIndex(docs) => {
                let mut indexed = 0;
                for doc in docs {
                    if self.search_machine.index_document(doc.clone()).is_ok() {
                        indexed += 1;
                    }
                }
                Response::success(format!("batch indexed {} docs", indexed))
            }
            LogEntry::BatchDelete(doc_ids) => {
                let mut deleted = 0;
                for doc_id in doc_ids {
                    if self.search_machine.delete_document(*doc_id).is_ok() {
                        deleted += 1;
                    }
                }
                Response::success(format!("batch deleted {} docs", deleted))
            }
            LogEntry::UpdateConfig(_settings) => {
                Response::success("config updated")
            }
        }
    }
}

impl RaftSnapshotBuilder<TypeConfig> for SquidexStateMachine {
    async fn build_snapshot(&mut self) -> Result<Snapshot<TypeConfig>, StorageError<NodeId>> {
        let state = self.state.read().clone();

        // Get snapshot data from search state machine
        let snapshot_data = self.search_machine.create_snapshot();

        // Combine with state machine state
        let combined = SquidexSnapshot::new(snapshot_data);
        let snapshot_bytes = bincode::serialize(&combined)
            .map_err(|e| StorageIOError::write_snapshot(None, &e))?;

        let last_applied = state.last_applied_log;
        let snapshot_id = format!(
            "snapshot-{}-{}",
            last_applied.map(|l| l.index).unwrap_or(0),
            chrono::Utc::now().timestamp()
        );

        let meta = SnapshotMeta {
            last_log_id: last_applied,
            last_membership: state.last_membership.clone(),
            snapshot_id,
        };

        Ok(Snapshot {
            meta,
            snapshot: Box::new(Cursor::new(snapshot_bytes)),
        })
    }
}

impl RaftStateMachine<TypeConfig> for SquidexStateMachine {
    type SnapshotBuilder = Self;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<NodeId>>, StoredMembership<NodeId, openraft::BasicNode>), StorageError<NodeId>> {
        let state = self.state.read();
        Ok((state.last_applied_log, state.last_membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<Response>, StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let mut responses = Vec::new();

        for entry in entries {
            let mut state = self.state.write();
            state.last_applied_log = Some(entry.log_id);

            match entry.payload {
                EntryPayload::Normal(ref request) => {
                    let response = self.apply_entry(&request.entry);
                    responses.push(response);
                }
                EntryPayload::Membership(ref membership) => {
                    state.last_membership = StoredMembership::new(Some(entry.log_id), membership.clone());
                    responses.push(Response::success("membership change applied"));
                }
                EntryPayload::Blank => {
                    responses.push(Response::success("blank entry"));
                }
            }

            drop(state);
        }

        self.persist_state();
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        Self {
            data_dir: self.data_dir.clone(),
            state: RwLock::new(self.state.read().clone()),
            search_machine: self.search_machine.clone(),
        }
    }

    async fn begin_receiving_snapshot(&mut self) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
        Ok(Box::new(Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<NodeId, openraft::BasicNode>,
        snapshot: Box<Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<NodeId>> {
        let snapshot_bytes = snapshot.into_inner();

        // Deserialize combined snapshot
        let combined: SquidexSnapshot = bincode::deserialize(&snapshot_bytes)
            .map_err(|e| StorageIOError::read_snapshot(None, &e))?;

        // Restore search state machine
        self.search_machine
            .restore_snapshot(&combined.data)
            .map_err(|e| StorageIOError::read_snapshot(None, &e))?;

        // Update state
        let mut state = self.state.write();
        state.last_applied_log = meta.last_log_id;
        state.last_membership = meta.last_membership.clone();
        drop(state);

        self.persist_state();
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<TypeConfig>>, StorageError<NodeId>> {
        // For now, just build a new snapshot
        // In production, you'd cache the last snapshot
        let snapshot = self.build_snapshot().await?;
        Ok(Some(snapshot))
    }
}
