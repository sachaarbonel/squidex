//! Event history and log capture for testing
//!
//! This module provides thread-safe event logging for capturing
//! operation histories that can be verified against invariants.

use super::events::{Event, OperationId, OperationResult, OperationType, Timestamp};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe event log for capturing operation history
#[derive(Clone)]
pub struct EventLog {
    inner: Arc<RwLock<EventLogInner>>,
    next_op_id: Arc<AtomicU64>,
}

struct EventLogInner {
    events: Vec<Event>,
    pending: HashMap<OperationId, usize>, // Maps op_id to index in events
}

impl EventLog {
    /// Create a new empty event log
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(EventLogInner {
                events: Vec::new(),
                pending: HashMap::new(),
            })),
            next_op_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Record a new operation invocation
    pub fn record_invoke(&self, op_type: OperationType) -> OperationId {
        let op_id = OperationId::new(self.next_op_id.fetch_add(1, Ordering::SeqCst));
        let event = Event::invoke(op_id, op_type);

        let mut inner = self.inner.write();
        let index = inner.events.len();
        inner.events.push(event);
        inner.pending.insert(op_id, index);

        op_id
    }

    /// Record operation completion
    pub fn record_return(&self, op_id: OperationId, result: OperationResult) {
        let mut inner = self.inner.write();

        if let Some(&index) = inner.pending.get(&op_id) {
            if let Some(event) = inner.events.get_mut(index) {
                event.complete(result);
            }
            inner.pending.remove(&op_id);
        }
    }

    /// Update event with Raft index
    pub fn update_raft_index(&self, op_id: OperationId, raft_index: u64) {
        let mut inner = self.inner.write();

        if let Some(&index) = inner.pending.get(&op_id) {
            if let Some(event) = inner.events.get_mut(index) {
                event.raft_index = Some(raft_index);
            }
        } else {
            // Check completed events
            for event in inner.events.iter_mut() {
                if event.op_id == op_id {
                    event.raft_index = Some(raft_index);
                    break;
                }
            }
        }
    }

    /// Get all events
    pub fn events(&self) -> Vec<Event> {
        self.inner.read().events.clone()
    }

    /// Get completed events only
    pub fn completed_events(&self) -> Vec<Event> {
        self.inner
            .read()
            .events
            .iter()
            .filter(|e| e.is_complete())
            .cloned()
            .collect()
    }

    /// Get pending events
    pub fn pending_events(&self) -> Vec<Event> {
        self.inner
            .read()
            .events
            .iter()
            .filter(|e| !e.is_complete())
            .cloned()
            .collect()
    }

    /// Number of recorded events
    pub fn len(&self) -> usize {
        self.inner.read().events.len()
    }

    /// Check if log is empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().events.is_empty()
    }

    /// Clear all events
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.events.clear();
        inner.pending.clear();
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let events = self.events();
        serde_json::to_string_pretty(&events)
    }

    /// Import from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let events: Vec<Event> = serde_json::from_str(json)?;
        let log = Self::new();

        // Collect pending events and max op_id first
        let mut max_op_id = 0u64;
        let mut pending_entries = Vec::new();
        for (idx, event) in events.iter().enumerate() {
            max_op_id = max_op_id.max(event.op_id.0);
            if !event.is_complete() {
                pending_entries.push((event.op_id, idx));
            }
        }

        // Now update the inner state
        let mut inner = log.inner.write();
        inner.events = events;
        for (op_id, idx) in pending_entries {
            inner.pending.insert(op_id, idx);
        }
        drop(inner);

        // Update next_op_id to be greater than any existing
        log.next_op_id.store(max_op_id + 1, Ordering::SeqCst);

        Ok(log)
    }

    /// Get events in a time range
    pub fn events_in_range(&self, start: Timestamp, end: Timestamp) -> Vec<Event> {
        self.inner
            .read()
            .events
            .iter()
            .filter(|e| e.invoke_time >= start && e.invoke_time <= end)
            .cloned()
            .collect()
    }

    /// Get events by operation type
    pub fn events_by_type(&self, filter: impl Fn(&OperationType) -> bool) -> Vec<Event> {
        self.inner
            .read()
            .events
            .iter()
            .filter(|e| filter(&e.op_type))
            .cloned()
            .collect()
    }

    /// Get index operations
    pub fn index_operations(&self) -> Vec<Event> {
        self.events_by_type(|op| matches!(op, OperationType::Index { .. }))
    }

    /// Get delete operations
    pub fn delete_operations(&self) -> Vec<Event> {
        self.events_by_type(|op| matches!(op, OperationType::Delete { .. }))
    }

    /// Get search operations
    pub fn search_operations(&self) -> Vec<Event> {
        self.events_by_type(|op| matches!(op, OperationType::Search { .. }))
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_log_basic() {
        let log = EventLog::new();

        let op_id = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });

        assert_eq!(log.len(), 1);
        assert_eq!(log.pending_events().len(), 1);
        assert_eq!(log.completed_events().len(), 0);

        log.record_return(op_id, OperationResult::IndexSuccess { doc_id: 1 });

        assert_eq!(log.pending_events().len(), 0);
        assert_eq!(log.completed_events().len(), 1);
    }

    #[test]
    fn test_event_log_multiple_operations() {
        let log = EventLog::new();

        // Index three documents
        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "first".to_string(),
        });
        let op2 = log.record_invoke(OperationType::Index {
            doc_id: 2,
            content: "second".to_string(),
        });
        let op3 = log.record_invoke(OperationType::Index {
            doc_id: 3,
            content: "third".to_string(),
        });

        assert_eq!(log.len(), 3);
        assert_eq!(log.pending_events().len(), 3);

        // Complete them out of order
        log.record_return(op2, OperationResult::IndexSuccess { doc_id: 2 });
        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });
        log.record_return(op3, OperationResult::IndexSuccess { doc_id: 3 });

        assert_eq!(log.pending_events().len(), 0);
        assert_eq!(log.completed_events().len(), 3);
    }

    #[test]
    fn test_event_log_json_roundtrip() {
        let log = EventLog::new();

        let op_id = log.record_invoke(OperationType::Delete { doc_id: 42 });
        log.record_return(op_id, OperationResult::DeleteSuccess { doc_id: 42 });

        let json = log.to_json().unwrap();
        let restored = EventLog::from_json(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.completed_events().len(), 1);

        let events = restored.events();
        assert_eq!(events[0].op_id, OperationId::new(1));
        if let OperationType::Delete { doc_id } = &events[0].op_type {
            assert_eq!(*doc_id, 42);
        } else {
            panic!("Expected Delete operation");
        }
    }

    #[test]
    fn test_event_log_raft_index_update() {
        let log = EventLog::new();

        let op_id = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });

        log.update_raft_index(op_id, 100);
        log.record_return(op_id, OperationResult::IndexSuccess { doc_id: 1 });

        let events = log.completed_events();
        assert_eq!(events[0].raft_index, Some(100));
    }

    #[test]
    fn test_event_log_type_filters() {
        let log = EventLog::new();

        let op1 = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        let op2 = log.record_invoke(OperationType::Delete { doc_id: 2 });
        let op3 = log.record_invoke(OperationType::Search {
            query: "test".to_string(),
        });

        log.record_return(op1, OperationResult::IndexSuccess { doc_id: 1 });
        log.record_return(op2, OperationResult::DeleteSuccess { doc_id: 2 });
        log.record_return(
            op3,
            OperationResult::SearchSuccess {
                doc_ids: vec![1, 2],
            },
        );

        assert_eq!(log.index_operations().len(), 1);
        assert_eq!(log.delete_operations().len(), 1);
        assert_eq!(log.search_operations().len(), 1);
    }

    #[test]
    fn test_event_log_clear() {
        let log = EventLog::new();

        let op_id = log.record_invoke(OperationType::Index {
            doc_id: 1,
            content: "test".to_string(),
        });
        log.record_return(op_id, OperationResult::IndexSuccess { doc_id: 1 });

        assert_eq!(log.len(), 1);

        log.clear();

        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }
}
