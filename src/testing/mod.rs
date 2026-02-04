//! Testing infrastructure for correctness verification
//!
//! This module provides:
//! - Event capture and history tracking
//! - Invariant checking framework
//! - Property-based testing support
//! - Integration with Squidex state machine
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use squidex::testing::prelude::*;
//! use squidex::state_machine::InstrumentedStateMachine;
//!
//! // Wrap your state machine for event capture
//! let instrumented = InstrumentedStateMachine::new(machine);
//!
//! // Perform operations (automatically captured)
//! instrumented.index_document(doc, raft_index)?;
//! instrumented.get_document(doc_id)?;
//!
//! // Check invariants
//! let violations = instrumented.check_invariants(&default_invariants());
//! assert!(violations.is_empty());
//! ```
//!
//! # Invariants
//!
//! The following invariants are provided:
//!
//! - **IndexedDocumentRetrievable**: Indexed documents can be retrieved
//! - **DeletedDocumentNotRetrievable**: Deleted documents are not retrievable
//! - **MonotonicReads**: Once observed, documents don't disappear (without delete)
//! - **UniqueDocumentIds**: No duplicate ID allocation
//! - **SearchReflectsIndex**: Search results match indexed state
//! - **RaftOrderingRespected**: Operations applied in Raft order

pub mod events;
pub mod history;
pub mod invariants;

pub use events::{Event, OperationId, OperationResult, OperationType, Timestamp};
pub use history::EventLog;
pub use invariants::{
    check_all_invariants, default_invariants, DeletedDocumentNotRetrievable,
    IndexedDocumentRetrievable, Invariant, MonotonicReads, RaftOrderingRespected,
    SearchReflectsIndex, UniqueDocumentIds, Violation,
};

/// Prelude for easy imports
pub mod prelude {
    pub use super::events::*;
    pub use super::history::EventLog;
    pub use super::invariants::{
        check_all_invariants, default_invariants, Invariant, Violation,
    };
}
