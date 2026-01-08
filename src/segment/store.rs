use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::segment::manifest::SegmentManifest;
use crate::segment::reader::{SegmentMeta, SegmentReader, SegmentReaderBuilder};
use crate::segment::writer::SegmentWriteResult;

const MANIFEST_FILE: &str = "segments.manifest";

/// Persistent storage for segment files and manifest.
pub struct SegmentStore {
    base_dir: PathBuf,
}

impl SegmentStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> io::Result<Self> {
        fs::create_dir_all(&base_dir)?;
        Ok(Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        })
    }

    fn segment_dir(&self, id: u64) -> PathBuf {
        self.base_dir.join(format!("segment_{}", id))
    }

    pub fn write_segment(&self, result: &SegmentWriteResult) -> io::Result<()> {
        let dir = self.segment_dir(result.reader.id().0);
        fs::create_dir_all(&dir)?;

        fs::write(dir.join("postings.bin"), &result.postings_data)?;
        fs::write(dir.join("fst.bin"), &result.fst_data)?;
        fs::write(
            dir.join("term_meta.bin"),
            bincode::serialize(&result.term_metadata)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        )?;
        fs::write(dir.join("docno_map.bin"), &result.docno_map_data)?;
        fs::write(dir.join("stats.bin"), &result.stats_data)?;
        Ok(())
    }

    pub fn read_segment(&self, meta: SegmentMeta) -> io::Result<Arc<SegmentReader>> {
        let dir = self.segment_dir(meta.id.0);
        let postings = fs::read(dir.join("postings.bin"))?;
        let fst_data = fs::read(dir.join("fst.bin"))?;
        let term_meta_bytes = fs::read(dir.join("term_meta.bin"))?;
        let term_meta: Vec<crate::segment::types::PostingListMeta> =
            bincode::deserialize(&term_meta_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let docno_map = fs::read(dir.join("docno_map.bin"))?;
        let stats_bytes = fs::read(dir.join("stats.bin"))?;
        let stats: crate::segment::statistics::SegmentStatistics =
            bincode::deserialize(&stats_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let builder = SegmentReaderBuilder::new()
            .with_meta(meta)
            .with_terms(fst_data, term_meta)
            .with_postings(postings)
            .with_docno_map(
                crate::segment::docno_map::DocNoMap::deserialize(&docno_map)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            )
            .with_stats(stats);

        Ok(Arc::new(builder.build()?))
    }

    pub fn save_manifest(&self, manifest: &SegmentManifest) -> io::Result<()> {
        let bytes = manifest.to_bincode()?;
        fs::write(self.base_dir.join(MANIFEST_FILE), bytes)?;
        Ok(())
    }

    pub fn load_manifest(&self) -> io::Result<SegmentManifest> {
        let path = self.base_dir.join(MANIFEST_FILE);
        if !path.exists() {
            return Ok(SegmentManifest::new());
        }
        let bytes = fs::read(path)?;
        SegmentManifest::from_bincode(&bytes)
    }
}
