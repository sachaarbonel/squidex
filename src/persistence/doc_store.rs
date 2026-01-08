use std::path::PathBuf;

use fjall::{Database, Keyspace, KeyspaceCreateOptions};
use serde::{Deserialize, Serialize};

use crate::error::SquidexError;
use crate::models::Document;
use crate::persistence::{BlobLog, BlobPointer};
use crate::Result;

/// Pointer plus metadata stored in Fjall for a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocPointer {
    pub blob: BlobPointer,
    pub raft_index: u64,
}

impl DocPointer {
    pub fn new(blob: BlobPointer, raft_index: u64) -> Self {
        Self { blob, raft_index }
    }
}

/// Fjall-backed document store with append-only blob log.
pub struct DocStore {
    base_dir: PathBuf,
    db: Database,
    docptr: Keyspace,
    tombstone: Keyspace,
    index_meta: Keyspace,
    blob_log: BlobLog,
}

const DOCPTR_CF: &str = "docptr";
const TOMBSTONE_CF: &str = "tombstone";
const INDEX_META_CF: &str = "index_meta";
const INDEX_APPLIED_KEY: &[u8] = b"index_applied_index";

impl DocStore {
    pub fn open(base_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&base_dir).map_err(SquidexError::Io)?;
        let db = Database::builder(&base_dir).open().map_err(|e| {
            SquidexError::Internal(format!("failed to open fjall doc store: {}", e))
        })?;

        let docptr = db
            .keyspace(DOCPTR_CF, || KeyspaceCreateOptions::default())
            .map_err(|e| SquidexError::Internal(format!("failed to open docptr cf: {}", e)))?;
        let tombstone = db
            .keyspace(TOMBSTONE_CF, || KeyspaceCreateOptions::default())
            .map_err(|e| SquidexError::Internal(format!("failed to open tombstone cf: {}", e)))?;
        let index_meta = db
            .keyspace(INDEX_META_CF, || KeyspaceCreateOptions::default())
            .map_err(|e| SquidexError::Internal(format!("failed to open index_meta cf: {}", e)))?;

        let blob_path = base_dir.join("doc.blob");
        let blob_log = BlobLog::open(blob_path)?;

        Ok(Self {
            base_dir,
            db,
            docptr,
            tombstone,
            index_meta,
            blob_log,
        })
    }

    fn encode_doc_id(doc_id: u64) -> [u8; 8] {
        doc_id.to_be_bytes()
    }

    /// Persist a document: append to blob log and store pointer.
    pub fn put_document(&self, doc: &Document, raft_index: u64) -> Result<DocPointer> {
        let payload = bincode::serialize(doc).map_err(SquidexError::Serialization)?;
        let blob_ptr = self.blob_log.append(&payload)?;
        let ptr = DocPointer::new(blob_ptr, raft_index);
        let key = Self::encode_doc_id(doc.id);
        let val = bincode::serialize(&ptr).map_err(SquidexError::Serialization)?;
        self.docptr
            .insert(key, val)
            .map_err(|e| SquidexError::Internal(e.to_string()))?;

        // Clear tombstone if present
        let _ = self.tombstone.remove(key);
        Ok(ptr)
    }

    /// Mark a document as deleted (tombstone), without removing blob immediately.
    pub fn tombstone(&self, doc_id: u64, raft_index: u64) -> Result<()> {
        let key = Self::encode_doc_id(doc_id);
        let val = raft_index.to_be_bytes();
        self.tombstone
            .insert(key, val)
            .map_err(|e| SquidexError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn is_tombstoned(&self, doc_id: u64) -> Result<bool> {
        let key = Self::encode_doc_id(doc_id);
        Ok(self
            .tombstone
            .get(key)
            .map_err(|e| SquidexError::Internal(e.to_string()))?
            .is_some())
    }

    pub fn list_tombstones(&self) -> Result<Vec<(u64, u64)>> {
        let mut out = Vec::new();
        for kv in self.tombstone.iter() {
            let key = kv
                .key()
                .map_err(|e| SquidexError::Internal(e.to_string()))?;
            let key_bytes = key.as_ref().to_vec();
            if key_bytes.len() != 8 {
                continue;
            }
            let mut doc_id_bytes = [0u8; 8];
            doc_id_bytes.copy_from_slice(&key_bytes);
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            if let Some(val) = self
                .tombstone
                .get(&key_bytes)
                .map_err(|e| SquidexError::Internal(e.to_string()))?
            {
                if val.len() == 8 {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(val.as_ref());
                    let raft_index = u64::from_be_bytes(buf);
                    out.push((doc_id, raft_index));
                }
            }
        }
        Ok(out)
    }

    pub fn exists(&self, doc_id: u64) -> Result<bool> {
        let key = Self::encode_doc_id(doc_id);
        Ok(self
            .docptr
            .get(key)
            .map_err(|e| SquidexError::Internal(e.to_string()))?
            .is_some())
    }

    pub fn get_document(&self, doc_id: u64) -> Result<Option<Document>> {
        if self.is_tombstoned(doc_id)? {
            return Ok(None);
        }
        let key = Self::encode_doc_id(doc_id);
        let Some(val) = self
            .docptr
            .get(key)
            .map_err(|e| SquidexError::Internal(e.to_string()))?
        else {
            return Ok(None);
        };
        let ptr: DocPointer = bincode::deserialize(&val).map_err(SquidexError::Serialization)?;
        let payload = self.blob_log.read(ptr.blob)?;
        let doc: Document = bincode::deserialize(&payload).map_err(SquidexError::Serialization)?;
        Ok(Some(doc))
    }

    /// Iterate all document pointers (including those that may be tombstoned).
    pub fn list_doc_pointers(&self) -> Result<Vec<(u64, DocPointer)>> {
        let mut out = Vec::new();
        for kv in self.docptr.iter() {
            let key = kv
                .key()
                .map_err(|e| SquidexError::Internal(e.to_string()))?;
            let key_bytes = key.as_ref().to_vec();
            if key_bytes.len() != 8 {
                continue;
            }
            let mut doc_id_bytes = [0u8; 8];
            doc_id_bytes.copy_from_slice(&key_bytes);
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            if let Some(val) = self
                .docptr
                .get(&key_bytes)
                .map_err(|e| SquidexError::Internal(e.to_string()))?
            {
                let ptr: DocPointer =
                    bincode::deserialize(val.as_ref()).map_err(SquidexError::Serialization)?;
                out.push((doc_id, ptr));
            }
        }
        Ok(out)
    }

    /// Highest raft index that has been fully applied to indexes.
    pub fn set_index_applied_index(&self, index: u64) -> Result<()> {
        let val = index.to_be_bytes();
        self.index_meta
            .insert(INDEX_APPLIED_KEY, val)
            .map_err(|e| SquidexError::Internal(e.to_string()))?;
        Ok(())
    }

    pub fn get_index_applied_index(&self) -> Result<u64> {
        let Some(val) = self
            .index_meta
            .get(INDEX_APPLIED_KEY)
            .map_err(|e| SquidexError::Internal(e.to_string()))?
        else {
            return Ok(0);
        };
        if val.len() != 8 {
            return Ok(0);
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(val.as_ref());
        Ok(u64::from_be_bytes(buf))
    }

    pub fn blob_path(&self) -> &PathBuf {
        self.blob_log.path()
    }

    /// Wipe all logical content (doc pointers, tombstones, index meta) and reset the blob log.
    pub fn clear_all(&self) -> Result<()> {
        // Remove all docptr entries
        for kv in self.docptr.iter() {
            if let Ok(key) = kv.key() {
                let _ = self.docptr.remove(key);
            }
        }

        for kv in self.tombstone.iter() {
            if let Ok(key) = kv.key() {
                let _ = self.tombstone.remove(key);
            }
        }

        let _ = self.index_meta.remove(INDEX_APPLIED_KEY);

        // Reset blob log
        self.blob_log.reset()?;

        Ok(())
    }
}
