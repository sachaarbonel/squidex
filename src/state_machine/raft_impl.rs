use bytes::Bytes;
use octopii::StateMachineTrait;

use crate::state_machine::SearchStateMachine;

/// Raft StateMachineTrait implementation for SearchStateMachine
impl StateMachineTrait for SearchStateMachine {
    /// Apply a committed command - MUST be deterministic
    fn apply(&self, command: &[u8]) -> std::result::Result<Bytes, String> {
        self.apply_command(command)
            .map_err(|e| format!("Failed to apply command: {}", e))
    }

    /// Create a complete snapshot of the search index
    fn snapshot(&self) -> Vec<u8> {
        self.create_snapshot()
    }

    /// Restore from a snapshot
    fn restore(&self, data: &[u8]) -> std::result::Result<(), String> {
        self.restore_snapshot(data)
            .map_err(|e| format!("Failed to restore snapshot: {}", e))
    }

    /// Compact the index - remove tombstones, optimize structures
    fn compact(&self) -> std::result::Result<(), String> {
        self.optimize_index()
            .map_err(|e| format!("Failed to compact index: {}", e))
    }
}
