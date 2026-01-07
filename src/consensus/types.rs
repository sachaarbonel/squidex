use openraft::BasicNode;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Cursor;

use crate::config::IndexSettings;
use crate::models::{Document, DocumentId};

/// Node ID type
pub type NodeId = u64;

/// OpenRaft type configuration for Squidex
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[derive(Serialize, Deserialize)]
pub struct TypeConfig;

impl fmt::Display for TypeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeConfig")
    }
}

impl openraft::RaftTypeConfig for TypeConfig {
    type D = Request;
    type R = Response;
    type Node = BasicNode;
    type NodeId = NodeId;
    type Entry = openraft::Entry<TypeConfig>;
    type SnapshotData = Cursor<Vec<u8>>;
    type AsyncRuntime = openraft::TokioRuntime;
    type Responder = openraft::impls::OneshotResponder<TypeConfig>;
}

/// Raft log entry commands
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogEntry {
    /// Index a document
    IndexDocument(Document),

    /// Delete a document
    DeleteDocument(DocumentId),

    /// Batch index
    BatchIndex(Vec<Document>),

    /// Batch delete
    BatchDelete(Vec<DocumentId>),

    /// Update configuration
    UpdateConfig(IndexSettings),
}

/// Request data for Raft (what gets proposed)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    pub entry: LogEntry,
}

impl Request {
    pub fn new(entry: LogEntry) -> Self {
        Self { entry }
    }
}

/// Response from applying a Raft entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    pub success: bool,
    pub message: Option<String>,
}

impl Response {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: Some(message.into()),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: Some(message.into()),
        }
    }
}

/// Snapshot data for state machine
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SquidexSnapshot {
    pub data: Vec<u8>,
}

impl SquidexSnapshot {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}
