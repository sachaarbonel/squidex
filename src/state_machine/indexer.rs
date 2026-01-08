use std::sync::{atomic::AtomicU64, atomic::Ordering, Arc};
use std::thread;

use crossbeam::channel::{Receiver, Sender};
use parking_lot::{Condvar, Mutex, RwLock};

use crate::config::IndexSettings;
use crate::models::{Document, DocumentId};
use crate::persistence::DocStore;
use crate::segment::{SegmentIndex, Version};
use crate::tokenizer::Tokenizer;
use crate::vector::HnswIndex;
use crate::Result;

#[derive(Clone, Debug)]
pub enum IndexOp {
    Upsert { doc: Document, raft_index: u64 },
    Delete { doc_id: DocumentId, raft_index: u64 },
}

pub struct IndexerHandles {
    pub tx: Sender<IndexOp>,
    pub join: thread::JoinHandle<()>,
}

pub fn spawn_indexer(
    tx: Sender<IndexOp>,
    rx: Receiver<IndexOp>,
    text_index: Arc<SegmentIndex>,
    hnsw_index: Arc<RwLock<HnswIndex>>,
    doc_store: Arc<DocStore>,
    index_applied: Arc<AtomicU64>,
    index_cv: Arc<(Mutex<()>, Condvar)>,
    settings: IndexSettings,
) -> IndexerHandles {
    let handle = thread::spawn(move || {
        // Dedicated tokenizer for the indexer thread
        let tokenizer = Tokenizer::new(&settings.tokenizer_config);
        while let Ok(op) = rx.recv() {
            match op {
                IndexOp::Upsert { doc, raft_index } => {
                    // Text indexing
                    let term_freqs = tokenizer.compute_term_frequencies(&doc.content);
                    let doc_len = tokenizer.tokenize(&doc.content).len() as u32;
                    // Version derived from updated_at for now
                    let version = Version::new(doc.updated_at);
                    let _ =
                        text_index.index_document(doc.id, version, term_freqs, doc_len, raft_index);

                    // Vector indexing
                    {
                        let mut hnsw = hnsw_index.write();
                        let _ = hnsw.insert(doc.id, &doc.embedding);
                    }

                    let _ = doc_store.set_index_applied_index(raft_index);
                    index_applied.store(raft_index, Ordering::SeqCst);
                    let (lock, cv) = &*index_cv;
                    let _g = lock.lock();
                    cv.notify_all();
                }
                IndexOp::Delete { doc_id, raft_index } => {
                    // Text: mark delete in buffer only; tombstone filtering happens at query time
                    let _ = text_index.delete_document(doc_id, raft_index);

                    // Vector: remove
                    {
                        let mut hnsw = hnsw_index.write();
                        hnsw.remove(doc_id);
                    }

                    let _ = doc_store.set_index_applied_index(raft_index);
                    index_applied.store(raft_index, Ordering::SeqCst);
                    let (lock, cv) = &*index_cv;
                    let _g = lock.lock();
                    cv.notify_all();
                }
            }
        }
    });

    IndexerHandles { tx, join: handle }
}
