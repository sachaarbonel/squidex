use std::fmt::Debug;
use std::io::Cursor;
use std::ops::RangeBounds;
use std::path::PathBuf;
use std::sync::Arc;

use fjall::{Database, Keyspace, KeyspaceCreateOptions, PersistMode};
use openraft::storage::{LogFlushed, LogState, RaftLogStorage, RaftStateMachine, Snapshot};
use openraft::{
    Entry, EntryPayload, LogId, OptionalSend, RaftLogReader, RaftSnapshotBuilder, SnapshotMeta,
    StorageError, StorageIOError, StoredMembership, Vote,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::types::{LogEntry, NodeId, Response, SquidexSnapshot, TypeConfig};
use crate::error::SquidexError;
use crate::state_machine::SearchStateMachine;
use crate::models::Command;

/// State machine state for persistence
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateMachineState {
    pub last_applied_log: Option<LogId<NodeId>>,
    pub last_membership: StoredMembership<NodeId, openraft::BasicNode>,
}

const LOG_KEYSPACE: &str = "raft_logs";
const META_KEYSPACE: &str = "raft_meta";
const META_LAST_PURGED: &[u8] = b"last_purged_log_id";
const META_VOTE: &[u8] = b"vote";
const META_COMMITTED: &[u8] = b"committed_log_id";

struct SquidexLogStoreInner {
    db: Database,
    logs: Keyspace,
    meta: Keyspace,
}

/// Fjall-backed Raft log storage.
#[derive(Clone)]
pub struct SquidexLogStore {
    inner: Arc<SquidexLogStoreInner>,
}

impl SquidexLogStore {
    pub fn new(data_dir: PathBuf) -> Result<Self, SquidexError> {
        let log_dir = data_dir.join("raft-log");
        std::fs::create_dir_all(&log_dir)?;

        let db = Database::builder(&log_dir)
            .open()
            .map_err(|e| SquidexError::Internal(format!("failed to open fjall db: {}", e)))?;

        let logs = db
            .keyspace(LOG_KEYSPACE, || KeyspaceCreateOptions::default())
            .map_err(|e| {
                SquidexError::Internal(format!("failed to open fjall logs keyspace: {}", e))
            })?;

        let meta = db
            .keyspace(META_KEYSPACE, || KeyspaceCreateOptions::default())
            .map_err(|e| {
                SquidexError::Internal(format!("failed to open fjall meta keyspace: {}", e))
            })?;

        Ok(Self {
            inner: Arc::new(SquidexLogStoreInner { db, logs, meta }),
        })
    }

    fn encode_log_key(index: u64) -> [u8; 8] {
        index.to_be_bytes()
    }

    fn decode_log_key(key: &[u8]) -> Option<u64> {
        if key.len() != 8 {
            return None;
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(key);
        Some(u64::from_be_bytes(buf))
    }

    fn deserialize_entry(
        bytes: &[u8],
        log_index: Option<u64>,
    ) -> Result<Entry<TypeConfig>, StorageError<NodeId>> {
        bincode::deserialize(bytes).map_err(|e| {
            let err = match log_index {
                Some(index) => StorageIOError::read_log_at_index(index, &e),
                None => StorageIOError::read_logs(&e),
            };
            err.into()
        })
    }

    fn last_log_id(&self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        let mut iter = self.inner.logs.iter().rev();
        let Some(item) = iter.next() else {
            return Ok(None);
        };

        let kv = item;
        let (key, value) = kv.into_inner().map_err(|e| StorageIOError::read_logs(&e))?;
        let log_index = Self::decode_log_key(key.as_ref());
        let entry = Self::deserialize_entry(value.as_ref(), log_index)?;
        Ok(Some(entry.log_id))
    }

    fn read_log_entries(
        &self,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>> {
        if let (Some(start), Some(end)) = (start, end) {
            if start > end {
                return Ok(Vec::new());
            }
        }

        let iter = match (start, end) {
            (Some(start), Some(end)) => {
                let start_key = Self::encode_log_key(start).to_vec();
                let end_key = Self::encode_log_key(end).to_vec();
                self.inner.logs.range(start_key..=end_key)
            }
            (Some(start), None) => {
                let start_key = Self::encode_log_key(start).to_vec();
                self.inner.logs.range(start_key..)
            }
            (None, Some(end)) => {
                let end_key = Self::encode_log_key(end).to_vec();
                self.inner.logs.range(..=end_key)
            }
            (None, None) => self.inner.logs.iter(),
        };

        let mut entries = Vec::new();
        for item in iter {
            let kv = item;
            let (key, value) = kv.into_inner().map_err(|e| StorageIOError::read_logs(&e))?;
            let log_index = Self::decode_log_key(key.as_ref());
            let entry = Self::deserialize_entry(value.as_ref(), log_index)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    fn remove_log_range(
        &self,
        start: Option<u64>,
        end: Option<u64>,
    ) -> Result<(), StorageError<NodeId>> {
        if let (Some(start), Some(end)) = (start, end) {
            if start > end {
                return Ok(());
            }
        }

        let iter = match (start, end) {
            (Some(start), Some(end)) => {
                let start_key = Self::encode_log_key(start).to_vec();
                let end_key = Self::encode_log_key(end).to_vec();
                self.inner.logs.range(start_key..=end_key)
            }
            (Some(start), None) => {
                let start_key = Self::encode_log_key(start).to_vec();
                self.inner.logs.range(start_key..)
            }
            (None, Some(end)) => {
                let end_key = Self::encode_log_key(end).to_vec();
                self.inner.logs.range(..=end_key)
            }
            (None, None) => self.inner.logs.iter(),
        };

        let mut keys = Vec::new();
        for item in iter {
            let kv = item;
            let key = kv.key().map_err(|e| StorageIOError::read_logs(&e))?;
            keys.push(key.as_ref().to_vec());
        }

        if keys.is_empty() {
            return Ok(());
        }

        let mut batch = self.inner.db.batch().durability(Some(PersistMode::SyncAll));
        for key in keys {
            batch.remove(&self.inner.logs, key);
        }

        batch.commit().map_err(|e| StorageIOError::write_logs(&e))?;
        Ok(())
    }
}

impl RaftLogReader<TypeConfig> for SquidexLogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + OptionalSend>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(value) => Some(*value),
            std::ops::Bound::Excluded(value) => match value.checked_add(1) {
                Some(value) => Some(value),
                None => return Ok(Vec::new()),
            },
            std::ops::Bound::Unbounded => None,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(value) => Some(*value),
            std::ops::Bound::Excluded(value) => match value.checked_sub(1) {
                Some(value) => Some(value),
                None => return Ok(Vec::new()),
            },
            std::ops::Bound::Unbounded => None,
        };

        self.read_log_entries(start, end)
    }
}

impl RaftLogStorage<TypeConfig> for SquidexLogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let last_purged_log_id = match self
            .inner
            .meta
            .get(META_LAST_PURGED)
            .map_err(|e| StorageIOError::read_logs(&e))?
        {
            Some(bytes) => {
                Some(bincode::deserialize(&bytes).map_err(|e| StorageIOError::read_logs(&e))?)
            }
            None => None,
        };
        let last_log_id = self.last_log_id()?;

        Ok(LogState {
            last_purged_log_id,
            last_log_id,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        let value = self
            .inner
            .meta
            .get(META_VOTE)
            .map_err(|e| StorageIOError::read_vote(&e))?;

        match value {
            Some(bytes) => Ok(Some(
                bincode::deserialize(&bytes).map_err(|e| StorageIOError::read_vote(&e))?,
            )),
            None => Ok(None),
        }
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        let bytes = bincode::serialize(vote).map_err(|e| StorageIOError::write_vote(&e))?;

        let mut batch = self.inner.db.batch().durability(Some(PersistMode::SyncAll));
        batch.insert(&self.inner.meta, META_VOTE, bytes);
        batch.commit().map_err(|e| StorageIOError::write_vote(&e))?;
        Ok(())
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        let bytes = bincode::serialize(&committed).map_err(|e| StorageIOError::write_logs(&e))?;

        let mut batch = self.inner.db.batch().durability(Some(PersistMode::SyncAll));
        batch.insert(&self.inner.meta, META_COMMITTED, bytes);
        batch.commit().map_err(|e| StorageIOError::write_logs(&e))?;
        Ok(())
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<TypeConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + OptionalSend,
        I::IntoIter: OptionalSend,
    {
        let result: Result<(), StorageError<NodeId>> =
            (|| -> Result<(), StorageIOError<NodeId>> {
                let mut batch = self.inner.db.batch().durability(Some(PersistMode::SyncAll));

                for entry in entries {
                    let key = Self::encode_log_key(entry.log_id.index);
                    let value = bincode::serialize(&entry)
                        .map_err(|e| StorageIOError::write_log_entry(entry.log_id, &e))?;
                    batch.insert(&self.inner.logs, key, value);
                }

                batch
                    .commit()
                    .map_err(|e| StorageIOError::<NodeId>::write_logs(&e))?;
                Ok(())
            })()
            .map_err(Into::into);

        let io_result = result
            .as_ref()
            .map(|_| ())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
        callback.log_io_completed(io_result);
        result
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.remove_log_range(Some(log_id.index), None)
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut keys = Vec::new();
        let iter = self
            .inner
            .logs
            .range(..=Self::encode_log_key(log_id.index).to_vec());

        for item in iter {
            let kv = item;
            let key = kv.key().map_err(|e| StorageIOError::read_logs(&e))?;
            keys.push(key.as_ref().to_vec());
        }

        let mut batch = self.inner.db.batch().durability(Some(PersistMode::SyncAll));
        for key in keys {
            batch.remove(&self.inner.logs, key);
        }

        let meta_bytes = bincode::serialize(&log_id).map_err(|e| StorageIOError::write_logs(&e))?;
        batch.insert(&self.inner.meta, META_LAST_PURGED, meta_bytes);

        batch.commit().map_err(|e| StorageIOError::write_logs(&e))?;
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
    fn apply_entry(&self, log_index: u64, entry: &LogEntry) -> Response {
        let cmd = match entry {
            LogEntry::IndexDocument(doc) => Command::IndexDocument(doc.clone()),
            LogEntry::DeleteDocument(doc_id) => Command::DeleteDocument(*doc_id),
            LogEntry::BatchIndex(docs) => Command::BatchIndex(docs.clone()),
            LogEntry::BatchDelete(doc_ids) => Command::BatchDelete(doc_ids.clone()),
            LogEntry::UpdateConfig(settings) => Command::UpdateSettings(settings.clone()),
        };

        match self.search_machine.apply_parsed_command(log_index, cmd) {
            Ok(_) => Response::success("ok"),
            Err(e) => Response::error(format!("apply failed: {}", e)),
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
        let snapshot_bytes =
            bincode::serialize(&combined).map_err(|e| StorageIOError::write_snapshot(None, &e))?;

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
    ) -> Result<
        (
            Option<LogId<NodeId>>,
            StoredMembership<NodeId, openraft::BasicNode>,
        ),
        StorageError<NodeId>,
    > {
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
                    let response = self.apply_entry(entry.log_id.index, &request.entry);
                    responses.push(response);
                }
                EntryPayload::Membership(ref membership) => {
                    state.last_membership =
                        StoredMembership::new(Some(entry.log_id), membership.clone());
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

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<Cursor<Vec<u8>>>, StorageError<NodeId>> {
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
