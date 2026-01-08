//! Segment manifest for tracking live segments
//!
//! Per SPEC Decision 5.2.4 — Manifest Atomicity:
//! 1. Write new segment files → fsync
//! 2. Write segments.manifest.tmp → fsync
//! 3. Atomic rename to segments.manifest → fsync directory
//! 4. Only then advance the shard index_applied_index watermark

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::reader::{SegmentMeta, SegmentReader};
use super::types::{RaftIndex, SegmentId};

/// Manifest entry for a segment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Segment metadata
    pub meta: SegmentMeta,
    /// Checksum of segment files
    pub checksum: u64,
}

/// The segment manifest tracks all live segments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentManifest {
    /// Manifest version (for format upgrades)
    pub version: u32,
    /// Generation number (incremented on each update)
    pub generation: u64,
    /// Next segment ID to allocate
    pub next_segment_id: SegmentId,
    /// Live segments
    pub segments: Vec<ManifestEntry>,
    /// Current index_applied_index
    pub index_applied_index: RaftIndex,
    /// Timestamp of last update
    pub updated_at: u64,
}

impl SegmentManifest {
    /// Current manifest format version
    pub const VERSION: u32 = 1;

    /// Create a new empty manifest
    pub fn new() -> Self {
        Self {
            version: Self::VERSION,
            generation: 0,
            next_segment_id: SegmentId::new(0),
            segments: Vec::new(),
            index_applied_index: 0,
            updated_at: 0,
        }
    }

    /// Allocate a new segment ID
    pub fn allocate_segment_id(&mut self) -> SegmentId {
        let id = self.next_segment_id;
        self.next_segment_id = id.next();
        id
    }

    /// Add a new segment to the manifest
    pub fn add_segment(&mut self, meta: SegmentMeta) {
        let entry = ManifestEntry {
            meta,
            checksum: 0, // TODO: compute checksum
        };
        self.segments.push(entry);
        self.generation += 1;
        self.updated_at = current_timestamp();
    }

    /// Remove a segment from the manifest (after merge)
    pub fn remove_segment(&mut self, segment_id: SegmentId) -> Option<ManifestEntry> {
        if let Some(pos) = self.segments.iter().position(|e| e.meta.id == segment_id) {
            self.generation += 1;
            self.updated_at = current_timestamp();
            Some(self.segments.remove(pos))
        } else {
            None
        }
    }

    /// Update the index_applied_index
    pub fn update_index_applied(&mut self, index: RaftIndex) {
        if index > self.index_applied_index {
            self.index_applied_index = index;
            self.updated_at = current_timestamp();
        }
    }

    /// Get total document count across all segments
    pub fn total_doc_count(&self) -> u64 {
        self.segments.iter().map(|e| e.meta.doc_count as u64).sum()
    }

    /// Get total live document count across all segments
    pub fn total_live_doc_count(&self) -> u64 {
        self.segments.iter().map(|e| e.meta.live_doc_count as u64).sum()
    }

    /// Get total size in bytes
    pub fn total_size_bytes(&self) -> u64 {
        self.segments.iter().map(|e| e.meta.size_bytes).sum()
    }

    /// Get segments count
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Check if manifest is empty
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get segment metadata by ID
    pub fn get_segment(&self, segment_id: SegmentId) -> Option<&ManifestEntry> {
        self.segments.iter().find(|e| e.meta.id == segment_id)
    }

    /// Iterate over segment entries
    pub fn iter(&self) -> impl Iterator<Item = &ManifestEntry> {
        self.segments.iter()
    }

    /// Serialize the manifest to JSON
    pub fn to_json(&self) -> io::Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize manifest from JSON
    pub fn from_json(data: &[u8]) -> io::Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Serialize the manifest to bincode (more compact)
    pub fn to_bincode(&self) -> io::Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize manifest from bincode
    pub fn from_bincode(data: &[u8]) -> io::Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Default for SegmentManifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe manifest holder with atomic updates
pub struct ManifestHolder {
    inner: arc_swap::ArcSwap<SegmentManifest>,
}

impl ManifestHolder {
    pub fn new(manifest: SegmentManifest) -> Self {
        Self {
            inner: arc_swap::ArcSwap::from_pointee(manifest),
        }
    }

    /// Get the current manifest
    pub fn load(&self) -> arc_swap::Guard<Arc<SegmentManifest>> {
        self.inner.load()
    }

    /// Get a clone of the current manifest
    pub fn snapshot(&self) -> SegmentManifest {
        (**self.inner.load()).clone()
    }

    /// Atomically replace the manifest
    pub fn store(&self, manifest: SegmentManifest) {
        self.inner.store(Arc::new(manifest));
    }

    /// Update the manifest using a closure
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut SegmentManifest),
    {
        let mut manifest = self.snapshot();
        f(&mut manifest);
        self.store(manifest);
    }

    /// Get generation number
    pub fn generation(&self) -> u64 {
        self.inner.load().generation
    }
}

impl Default for ManifestHolder {
    fn default() -> Self {
        Self::new(SegmentManifest::new())
    }
}

// Implement Serialize for SegmentMeta
impl Serialize for SegmentMeta {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SegmentMeta", 7)?;
        state.serialize_field("id", &self.id.0)?;
        state.serialize_field("min_raft_index", &self.min_raft_index)?;
        state.serialize_field("max_raft_index", &self.max_raft_index)?;
        state.serialize_field("doc_count", &self.doc_count)?;
        state.serialize_field("live_doc_count", &self.live_doc_count)?;
        state.serialize_field("size_bytes", &self.size_bytes)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SegmentMeta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SegmentMetaHelper {
            id: u64,
            min_raft_index: RaftIndex,
            max_raft_index: RaftIndex,
            doc_count: u32,
            live_doc_count: u32,
            size_bytes: u64,
            created_at: u64,
        }

        let helper = SegmentMetaHelper::deserialize(deserializer)?;
        Ok(SegmentMeta {
            id: SegmentId::new(helper.id),
            min_raft_index: helper.min_raft_index,
            max_raft_index: helper.max_raft_index,
            doc_count: helper.doc_count,
            live_doc_count: helper.live_doc_count,
            size_bytes: helper.size_bytes,
            created_at: helper.created_at,
        })
    }
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_basic() {
        let mut manifest = SegmentManifest::new();

        assert_eq!(manifest.segment_count(), 0);
        assert!(manifest.is_empty());

        let id = manifest.allocate_segment_id();
        assert_eq!(id, SegmentId::new(0));

        let meta = SegmentMeta {
            id,
            min_raft_index: 1,
            max_raft_index: 100,
            doc_count: 1000,
            live_doc_count: 950,
            size_bytes: 1024 * 1024,
            created_at: current_timestamp(),
        };

        manifest.add_segment(meta);

        assert_eq!(manifest.segment_count(), 1);
        assert!(!manifest.is_empty());
        assert_eq!(manifest.total_doc_count(), 1000);
        assert_eq!(manifest.total_live_doc_count(), 950);
    }

    #[test]
    fn test_manifest_serialization() {
        let mut manifest = SegmentManifest::new();

        let id = manifest.allocate_segment_id();
        manifest.add_segment(SegmentMeta {
            id,
            min_raft_index: 1,
            max_raft_index: 100,
            doc_count: 1000,
            live_doc_count: 950,
            size_bytes: 1024 * 1024,
            created_at: 0,
        });

        // Test JSON serialization
        let json = manifest.to_json().unwrap();
        let restored = SegmentManifest::from_json(&json).unwrap();
        assert_eq!(restored.segment_count(), 1);
        assert_eq!(restored.total_doc_count(), 1000);

        // Test bincode serialization
        let bincode = manifest.to_bincode().unwrap();
        let restored = SegmentManifest::from_bincode(&bincode).unwrap();
        assert_eq!(restored.segment_count(), 1);
        assert_eq!(restored.total_doc_count(), 1000);
    }

    #[test]
    fn test_manifest_remove_segment() {
        let mut manifest = SegmentManifest::new();

        let id1 = manifest.allocate_segment_id();
        let id2 = manifest.allocate_segment_id();

        manifest.add_segment(SegmentMeta {
            id: id1,
            min_raft_index: 1,
            max_raft_index: 50,
            doc_count: 500,
            live_doc_count: 500,
            size_bytes: 512 * 1024,
            created_at: 0,
        });

        manifest.add_segment(SegmentMeta {
            id: id2,
            min_raft_index: 51,
            max_raft_index: 100,
            doc_count: 500,
            live_doc_count: 500,
            size_bytes: 512 * 1024,
            created_at: 0,
        });

        assert_eq!(manifest.segment_count(), 2);

        let removed = manifest.remove_segment(id1);
        assert!(removed.is_some());
        assert_eq!(manifest.segment_count(), 1);
        assert!(manifest.get_segment(id1).is_none());
        assert!(manifest.get_segment(id2).is_some());
    }

    #[test]
    fn test_manifest_holder() {
        let holder = ManifestHolder::new(SegmentManifest::new());

        assert_eq!(holder.generation(), 0);
        assert!(holder.load().is_empty());

        holder.update(|m| {
            let id = m.allocate_segment_id();
            m.add_segment(SegmentMeta {
                id,
                min_raft_index: 1,
                max_raft_index: 100,
                doc_count: 1000,
                live_doc_count: 1000,
                size_bytes: 1024,
                created_at: 0,
            });
        });

        assert_eq!(holder.generation(), 1);
        assert_eq!(holder.load().segment_count(), 1);
    }

    #[test]
    fn test_index_applied_update() {
        let mut manifest = SegmentManifest::new();

        assert_eq!(manifest.index_applied_index, 0);

        manifest.update_index_applied(100);
        assert_eq!(manifest.index_applied_index, 100);

        // Should not go backwards
        manifest.update_index_applied(50);
        assert_eq!(manifest.index_applied_index, 100);

        manifest.update_index_applied(150);
        assert_eq!(manifest.index_applied_index, 150);
    }
}
