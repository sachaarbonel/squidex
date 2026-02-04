//! Event types for capturing operations in the system
//!
//! This module defines the event model used for invariant-based testing.
//! Events capture all operations with timing information for verification.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for an operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub u64);

impl OperationId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Type of operation in the system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Index a document
    Index { doc_id: u64, content: String },
    /// Delete a document
    Delete { doc_id: u64 },
    /// Search operation
    Search { query: String },
    /// Get document by ID
    Get { doc_id: u64 },
    /// Batch index
    BatchIndex { doc_ids: Vec<u64> },
    /// Batch delete
    BatchDelete { doc_ids: Vec<u64> },
}

/// Result of an operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationResult {
    /// Successful index
    IndexSuccess { doc_id: u64 },
    /// Successful delete
    DeleteSuccess { doc_id: u64 },
    /// Search results
    SearchSuccess { doc_ids: Vec<u64> },
    /// Get result
    GetSuccess { doc_id: u64, found: bool },
    /// Batch operation success
    BatchSuccess { count: usize },
    /// Operation failed
    Error { message: String },
}

/// Timestamp wrapper for consistent time handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

impl Timestamp {
    pub fn now() -> Self {
        Self(Utc::now().timestamp_nanos_opt().unwrap_or(0))
    }

    pub fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    pub fn as_nanos(&self) -> i64 {
        self.0
    }

    pub fn to_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_nanos(self.0)
    }
}

/// A recorded event in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique operation ID
    pub op_id: OperationId,

    /// Type of operation
    pub op_type: OperationType,

    /// When the operation was invoked
    pub invoke_time: Timestamp,

    /// When the operation returned (None if still pending)
    pub return_time: Option<Timestamp>,

    /// Result of the operation (None if still pending)
    pub result: Option<OperationResult>,

    /// Raft log index (for distributed systems)
    pub raft_index: Option<u64>,

    /// Node that executed the operation
    pub node_id: Option<u64>,
}

impl Event {
    /// Create a new event at invocation time
    pub fn invoke(op_id: OperationId, op_type: OperationType) -> Self {
        Self {
            op_id,
            op_type,
            invoke_time: Timestamp::now(),
            return_time: None,
            result: None,
            raft_index: None,
            node_id: None,
        }
    }

    /// Mark event as completed
    pub fn complete(&mut self, result: OperationResult) {
        self.return_time = Some(Timestamp::now());
        self.result = Some(result);
    }

    /// Add Raft context
    pub fn with_raft_index(mut self, index: u64) -> Self {
        self.raft_index = Some(index);
        self
    }

    /// Add node context
    pub fn with_node_id(mut self, node_id: u64) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Check if event is completed
    pub fn is_complete(&self) -> bool {
        self.return_time.is_some() && self.result.is_some()
    }

    /// Duration of the operation in nanoseconds
    pub fn duration_nanos(&self) -> Option<i64> {
        self.return_time.map(|rt| rt.0 - self.invoke_time.0)
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Op({:?}) {:?} @ {} -> ",
            self.op_id, self.op_type, self.invoke_time.0
        )?;
        match &self.result {
            Some(result) => write!(f, "{:?} @ {:?}", result, self.return_time),
            None => write!(f, "<pending>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation_and_completion() {
        let op_id = OperationId::new(1);
        let mut event = Event::invoke(
            op_id,
            OperationType::Index {
                doc_id: 42,
                content: "test content".to_string(),
            },
        );

        assert!(!event.is_complete());
        assert!(event.result.is_none());
        assert!(event.return_time.is_none());

        event.complete(OperationResult::IndexSuccess { doc_id: 42 });

        assert!(event.is_complete());
        assert!(event.result.is_some());
        assert!(event.return_time.is_some());
    }

    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp::from_nanos(100);
        let t2 = Timestamp::from_nanos(200);

        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, Timestamp::from_nanos(100));
    }

    #[test]
    fn test_event_with_raft_context() {
        let op_id = OperationId::new(1);
        let event = Event::invoke(op_id, OperationType::Delete { doc_id: 1 })
            .with_raft_index(42)
            .with_node_id(3);

        assert_eq!(event.raft_index, Some(42));
        assert_eq!(event.node_id, Some(3));
    }
}
