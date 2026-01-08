//! Segment-based inverted index for full-text search
//!
//! This module implements immutable segment files with a mutable buffer
//! for recent writes, following the SPEC.md design decisions.
//!
//! # Architecture
//!
//! - `MutableBuffer`: In-memory buffer for recent writes
//! - `SegmentReader`: Immutable segment reader backed by mmapped files
//! - `SegmentIndex`: Combines mutable buffer + immutable segments
//! - `SegmentManifest`: Tracks live segments with atomic updates

mod buffer;
mod docno_map;
mod docvalues;
mod index;
mod manifest;
mod merge;
mod postings;
mod reader;
mod statistics;
mod store;
mod term_dict;
mod types;
mod writer;

pub use buffer::*;
pub use docno_map::*;
pub use docvalues::*;
pub use index::*;
pub use manifest::*;
pub use merge::*;
pub use postings::*;
pub use reader::*;
pub use statistics::*;
pub use store::*;
pub use term_dict::*;
pub use types::*;
pub use writer::*;
