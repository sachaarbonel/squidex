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

mod types;
mod statistics;
mod buffer;
mod postings;
mod term_dict;
mod docvalues;
mod docno_map;
mod reader;
mod writer;
mod manifest;
mod index;
mod merge;

pub use types::*;
pub use statistics::*;
pub use buffer::*;
pub use postings::*;
pub use term_dict::*;
pub use docvalues::*;
pub use docno_map::*;
pub use reader::*;
pub use writer::*;
pub use manifest::*;
pub use index::*;
pub use merge::*;
