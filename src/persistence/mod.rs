//! Persistence primitives: append-only blob log and Fjall-backed document store.

mod blob_log;
mod doc_store;

pub use blob_log::{BlobLog, BlobPointer};
pub use doc_store::{DocPointer, DocStore};
