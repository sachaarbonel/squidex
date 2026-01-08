//! Dense document number mapping
//!
//! Internal DocId mapping is mandatory for performance:
//! - Each segment allocates a dense `docno: u32` in `[0..max_doc)` space.
//! - Store per-segment arrays: `docno -> (doc_id, version)` (packed, mmappable).
//! - `delete_bitset` per segment.

use std::io::{self, Read, Write};

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use super::postings::{decode_vbyte, encode_vbyte};
use super::types::{DocNo, DocNoEntry, DocumentId, Version};

/// Dense document number mapping for a segment
///
/// Maps internal docno (u32) to external (doc_id, version) pairs.
/// This allows efficient iteration over posting lists while still
/// being able to resolve the external document ID when needed.
#[derive(Clone, Debug)]
pub struct DocNoMap {
    /// Dense array: docno -> (doc_id, version)
    entries: Vec<DocNoEntry>,
    /// Delete bitset: which docnos are deleted
    deleted: RoaringBitmap,
}

impl DocNoMap {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            deleted: RoaringBitmap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            deleted: RoaringBitmap::new(),
        }
    }

    /// Add a new document and return its docno
    pub fn add(&mut self, doc_id: DocumentId, version: Version) -> DocNo {
        let docno = DocNo::new(self.entries.len() as u32);
        self.entries.push(DocNoEntry::new(doc_id, version));
        docno
    }

    /// Get the entry for a docno
    pub fn get(&self, docno: DocNo) -> Option<&DocNoEntry> {
        self.entries.get(docno.as_usize())
    }

    /// Get the document ID for a docno
    pub fn get_doc_id(&self, docno: DocNo) -> Option<DocumentId> {
        self.entries.get(docno.as_usize()).map(|e| e.doc_id)
    }

    /// Get the version for a docno
    pub fn get_version(&self, docno: DocNo) -> Option<Version> {
        self.entries.get(docno.as_usize()).map(|e| e.version)
    }

    /// Mark a docno as deleted
    pub fn delete(&mut self, docno: DocNo) {
        self.deleted.insert(docno.as_u32());
    }

    /// Check if a docno is deleted
    pub fn is_deleted(&self, docno: DocNo) -> bool {
        self.deleted.contains(docno.as_u32())
    }

    /// Check if a docno is live (exists and not deleted)
    pub fn is_live(&self, docno: DocNo) -> bool {
        docno.as_usize() < self.entries.len() && !self.deleted.contains(docno.as_u32())
    }

    /// Get the number of documents (including deleted)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of live documents (excluding deleted)
    pub fn live_count(&self) -> usize {
        self.entries.len() - self.deleted.len() as usize
    }

    /// Get the number of deleted documents
    pub fn deleted_count(&self) -> usize {
        self.deleted.len() as usize
    }

    /// Get the delete ratio (for merge policy decisions)
    pub fn delete_ratio(&self) -> f64 {
        if self.entries.is_empty() {
            0.0
        } else {
            self.deleted.len() as f64 / self.entries.len() as f64
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the delete bitset
    pub fn deleted_bitset(&self) -> &RoaringBitmap {
        &self.deleted
    }

    /// Get all entries (for serialization/iteration)
    pub fn entries(&self) -> &[DocNoEntry] {
        &self.entries
    }

    /// Iterate over live documents
    pub fn live_docs(&self) -> impl Iterator<Item = (DocNo, &DocNoEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter(move |(i, _)| !self.deleted.contains(*i as u32))
            .map(|(i, entry)| (DocNo::new(i as u32), entry))
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Write entry count
        encode_vbyte(self.entries.len() as u32, &mut output);

        // Write entries (doc_id and version as u64s)
        for entry in &self.entries {
            output.extend_from_slice(&entry.doc_id.to_le_bytes());
            output.extend_from_slice(&entry.version.0.to_le_bytes());
        }

        // Write delete bitset
        let mut delete_bytes = Vec::new();
        self.deleted.serialize_into(&mut delete_bytes).unwrap();
        encode_vbyte(delete_bytes.len() as u32, &mut output);
        output.extend(delete_bytes);

        output
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;

        // Read entry count
        let count = decode_vbyte(data, &mut pos)? as usize;

        // Read entries
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if pos + 16 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Not enough data for docno entry",
                ));
            }

            let mut doc_id_bytes = [0u8; 8];
            doc_id_bytes.copy_from_slice(&data[pos..pos + 8]);
            pos += 8;
            let doc_id = u64::from_le_bytes(doc_id_bytes);

            let mut version_bytes = [0u8; 8];
            version_bytes.copy_from_slice(&data[pos..pos + 8]);
            pos += 8;
            let version = Version::new(u64::from_le_bytes(version_bytes));

            entries.push(DocNoEntry::new(doc_id, version));
        }

        // Read delete bitset
        let delete_len = decode_vbyte(data, &mut pos)? as usize;
        let deleted = RoaringBitmap::deserialize_from(&data[pos..pos + delete_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Self { entries, deleted })
    }

    /// Create from existing entries and deleted set
    pub fn from_parts(entries: Vec<DocNoEntry>, deleted: RoaringBitmap) -> Self {
        Self { entries, deleted }
    }

    /// Merge with another DocNoMap (for segment merging)
    /// Returns a new DocNoMap with remapped docnos and a mapping from old to new docnos
    pub fn merge_with(&self, other: &DocNoMap) -> (DocNoMap, Vec<DocNo>, Vec<DocNo>) {
        let mut merged = DocNoMap::with_capacity(self.live_count() + other.live_count());
        let mut self_remap = vec![DocNo::MAX; self.len()];
        let mut other_remap = vec![DocNo::MAX; other.len()];

        // Add live documents from self
        for (old_docno, entry) in self.live_docs() {
            let new_docno = merged.add(entry.doc_id, entry.version);
            self_remap[old_docno.as_usize()] = new_docno;
        }

        // Add live documents from other
        for (old_docno, entry) in other.live_docs() {
            let new_docno = merged.add(entry.doc_id, entry.version);
            other_remap[old_docno.as_usize()] = new_docno;
        }

        (merged, self_remap, other_remap)
    }
}

impl Default for DocNoMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for DocNoMap that tracks external doc_id -> docno mapping
#[derive(Debug)]
pub struct DocNoMapBuilder {
    map: DocNoMap,
    /// Lookup from external doc_id to internal docno
    doc_id_to_docno: std::collections::HashMap<DocumentId, DocNo>,
}

impl DocNoMapBuilder {
    pub fn new() -> Self {
        Self {
            map: DocNoMap::new(),
            doc_id_to_docno: std::collections::HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: DocNoMap::with_capacity(capacity),
            doc_id_to_docno: std::collections::HashMap::with_capacity(capacity),
        }
    }

    /// Add a document and return its docno
    pub fn add(&mut self, doc_id: DocumentId, version: Version) -> DocNo {
        let docno = self.map.add(doc_id, version);
        self.doc_id_to_docno.insert(doc_id, docno);
        docno
    }

    /// Look up docno by external doc_id
    pub fn get_docno(&self, doc_id: DocumentId) -> Option<DocNo> {
        self.doc_id_to_docno.get(&doc_id).copied()
    }

    /// Mark a document as deleted by external doc_id
    pub fn delete_by_doc_id(&mut self, doc_id: DocumentId) -> bool {
        if let Some(docno) = self.doc_id_to_docno.get(&doc_id) {
            self.map.delete(*docno);
            true
        } else {
            false
        }
    }

    /// Build the final DocNoMap (consuming the builder)
    pub fn build(self) -> DocNoMap {
        self.map
    }

    /// Get reference to the underlying map
    pub fn map(&self) -> &DocNoMap {
        &self.map
    }
}

impl Default for DocNoMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docno_map_basic() {
        let mut map = DocNoMap::new();

        let docno1 = map.add(100, Version::new(1));
        let docno2 = map.add(200, Version::new(2));
        let docno3 = map.add(300, Version::new(1));

        assert_eq!(docno1, DocNo::new(0));
        assert_eq!(docno2, DocNo::new(1));
        assert_eq!(docno3, DocNo::new(2));

        assert_eq!(map.get_doc_id(docno1), Some(100));
        assert_eq!(map.get_doc_id(docno2), Some(200));
        assert_eq!(map.get_version(docno2), Some(Version::new(2)));

        assert_eq!(map.len(), 3);
        assert_eq!(map.live_count(), 3);
    }

    #[test]
    fn test_docno_map_delete() {
        let mut map = DocNoMap::new();

        map.add(100, Version::new(1));
        let docno2 = map.add(200, Version::new(1));
        map.add(300, Version::new(1));

        assert!(!map.is_deleted(docno2));
        assert!(map.is_live(docno2));

        map.delete(docno2);

        assert!(map.is_deleted(docno2));
        assert!(!map.is_live(docno2));
        assert_eq!(map.len(), 3);
        assert_eq!(map.live_count(), 2);
        assert_eq!(map.deleted_count(), 1);
    }

    #[test]
    fn test_docno_map_serialization() {
        let mut map = DocNoMap::new();

        map.add(100, Version::new(1));
        map.add(200, Version::new(2));
        let docno3 = map.add(300, Version::new(3));
        map.delete(docno3);

        let data = map.serialize();
        let restored = DocNoMap::deserialize(&data).unwrap();

        assert_eq!(restored.len(), 3);
        assert_eq!(restored.get_doc_id(DocNo::new(0)), Some(100));
        assert_eq!(restored.get_doc_id(DocNo::new(1)), Some(200));
        assert_eq!(restored.get_version(DocNo::new(1)), Some(Version::new(2)));
        assert!(restored.is_deleted(DocNo::new(2)));
    }

    #[test]
    fn test_docno_map_merge() {
        let mut map1 = DocNoMap::new();
        map1.add(100, Version::new(1));
        let docno = map1.add(200, Version::new(1));
        map1.delete(docno); // Delete doc 200
        map1.add(300, Version::new(1));

        let mut map2 = DocNoMap::new();
        map2.add(400, Version::new(1));
        map2.add(500, Version::new(1));

        let (merged, remap1, remap2) = map1.merge_with(&map2);

        // Only live docs should be in merged
        assert_eq!(merged.len(), 4); // 2 from map1 + 2 from map2
        assert_eq!(merged.live_count(), 4);

        // Check remapping
        assert_eq!(remap1[0], DocNo::new(0)); // doc 100
        assert_eq!(remap1[1], DocNo::MAX); // doc 200 was deleted
        assert_eq!(remap1[2], DocNo::new(1)); // doc 300

        assert_eq!(remap2[0], DocNo::new(2)); // doc 400
        assert_eq!(remap2[1], DocNo::new(3)); // doc 500
    }

    #[test]
    fn test_docno_map_builder() {
        let mut builder = DocNoMapBuilder::new();

        let docno1 = builder.add(100, Version::new(1));
        let docno2 = builder.add(200, Version::new(1));

        assert_eq!(builder.get_docno(100), Some(docno1));
        assert_eq!(builder.get_docno(200), Some(docno2));
        assert_eq!(builder.get_docno(999), None);

        builder.delete_by_doc_id(100);

        let map = builder.build();
        assert!(map.is_deleted(docno1));
        assert!(!map.is_deleted(docno2));
    }

    #[test]
    fn test_delete_ratio() {
        let mut map = DocNoMap::new();

        for i in 0..10 {
            map.add(i as u64, Version::new(1));
        }

        assert_eq!(map.delete_ratio(), 0.0);

        map.delete(DocNo::new(0));
        map.delete(DocNo::new(1));

        assert!((map.delete_ratio() - 0.2).abs() < 0.001);
    }
}
