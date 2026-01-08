use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use crc32fast::Hasher;

use crate::error::SquidexError;
use crate::Result;

/// Pointer to a blob record inside the blob log.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BlobPointer {
    pub offset: u64,
    pub len: u32,
    pub crc32: u32,
}

impl BlobPointer {
    pub fn new(offset: u64, len: u32, crc32: u32) -> Self {
        Self { offset, len, crc32 }
    }
}

/// Append-only blob log for document bodies.
///
/// Record format:
/// - u32 length (little endian)
/// - u32 crc32 of payload
/// - raw payload bytes
pub struct BlobLog {
    path: PathBuf,
    file: Mutex<File>,
}

impl BlobLog {
    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(SquidexError::Io)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
            .map_err(SquidexError::Io)?;

        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }

    /// Append a payload and return its pointer.
    pub fn append(&self, payload: &[u8]) -> Result<BlobPointer> {
        let mut file = self.file.lock().unwrap();
        let offset = file.seek(SeekFrom::End(0)).map_err(SquidexError::Io)?;

        let len = payload.len() as u32;
        let mut hasher = Hasher::new();
        hasher.update(payload);
        let crc32 = hasher.finalize();

        file.write_all(&len.to_le_bytes())
            .map_err(SquidexError::Io)?;
        file.write_all(&crc32.to_le_bytes())
            .map_err(SquidexError::Io)?;
        file.write_all(payload).map_err(SquidexError::Io)?;
        // Rely on OS buffers; durability is provided by Raft replication.

        Ok(BlobPointer::new(offset, len, crc32))
    }

    /// Read a payload given its pointer, validating checksum.
    pub fn read(&self, ptr: BlobPointer) -> Result<Vec<u8>> {
        let mut file = self.file.lock().unwrap();
        file.seek(SeekFrom::Start(ptr.offset))
            .map_err(SquidexError::Io)?;

        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf).map_err(SquidexError::Io)?;
        let len = u32::from_le_bytes(len_buf);

        let mut crc_buf = [0u8; 4];
        file.read_exact(&mut crc_buf).map_err(SquidexError::Io)?;
        let stored_crc = u32::from_le_bytes(crc_buf);

        if len != ptr.len {
            return Err(SquidexError::Internal(format!(
                "Blob length mismatch: expected {}, found {}",
                ptr.len, len
            )));
        }

        let mut payload = vec![0u8; len as usize];
        file.read_exact(&mut payload).map_err(SquidexError::Io)?;

        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let crc = hasher.finalize();
        if crc != stored_crc || crc != ptr.crc32 {
            return Err(SquidexError::Internal(
                "Blob checksum mismatch (corrupt record)".to_string(),
            ));
        }

        Ok(payload)
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Truncate the log (used during snapshot restore).
    pub fn reset(&self) -> Result<()> {
        let mut file = self.file.lock().unwrap();
        file.set_len(0).map_err(SquidexError::Io)?;
        file.seek(SeekFrom::Start(0)).map_err(SquidexError::Io)?;
        Ok(())
    }
}
