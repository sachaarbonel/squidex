use thiserror::Error;

/// Main error type for Squidex operations
#[derive(Error, Debug)]
pub enum SquidexError {
    #[error("Document not found: {0}")]
    DocumentNotFound(u64),

    #[error("Invalid embedding dimensions: expected {expected}, got {actual}")]
    InvalidEmbeddingDimensions { expected: usize, actual: usize },

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Consensus error: {0}")]
    Consensus(String),

    #[error("Not leader - cannot process write request")]
    NotLeader,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Snapshot error: incompatible version {actual}, expected <= {expected}")]
    IncompatibleSnapshot { expected: u32, actual: u32 },

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Search error: {0}")]
    SearchError(String),
}

/// Result type alias for Squidex operations
pub type Result<T> = std::result::Result<T, SquidexError>;

impl SquidexError {
    /// Convert to a string suitable for Raft state machine responses
    pub fn to_state_machine_error(&self) -> String {
        self.to_string()
    }

    /// Check if this error indicates a transient failure that could be retried
    pub fn is_retriable(&self) -> bool {
        matches!(self, SquidexError::NotLeader | SquidexError::Consensus(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SquidexError::DocumentNotFound(42);
        assert_eq!(err.to_string(), "Document not found: 42");
    }

    #[test]
    fn test_retriable_errors() {
        assert!(SquidexError::NotLeader.is_retriable());
        assert!(SquidexError::Consensus("test".to_string()).is_retriable());
        assert!(!SquidexError::DocumentNotFound(1).is_retriable());
    }
}
