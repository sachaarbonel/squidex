//! Segment writer for creating new immutable segments
//!
//! when buffer hits thresholds, write a new immutable segment:
//! - postings + term dictionary + docvalues column files
//! - vector store (handled separately)

use std::collections::HashMap;
use std::io::{self, Write};

use crc32fast::Hasher;

use super::buffer::MutableBuffer;
use super::docno_map::DocNoMap;
use super::docvalues::{BooleanColumn, DocValuesReader, KeywordColumn, NumericColumn};
use super::postings::{PostingsReader, PostingsWriter};
use super::reader::{SegmentMeta, SegmentReader};
use super::statistics::SegmentStatistics;
use super::term_dict::{TermDictionary, TermDictionaryBuilder};
use super::types::{DocNo, Posting, PostingListMeta, SegmentId, Version};

/// Result of writing a segment
pub struct SegmentWriteResult {
    /// The created segment reader
    pub reader: SegmentReader,
    /// Postings data
    pub postings_data: Vec<u8>,
    /// Term dictionary FST data
    pub fst_data: Vec<u8>,
    /// Term metadata
    pub term_metadata: Vec<PostingListMeta>,
    /// DocNo map serialized
    pub docno_map_data: Vec<u8>,
    /// Statistics serialized
    pub stats_data: Vec<u8>,
    /// DocValues serialized
    pub docvalues_data: Vec<u8>,
}

impl SegmentWriteResult {
    /// Compute a checksum over all persisted segment artifacts.
    ///
    /// Algorithm: crc32fast (CRC32).
    /// Coverage: postings, term dictionary FST, term metadata (bincode), docno map, stats, docvalues.
    /// The manifest checksum MUST match this value for both flush and merge paths.
    pub fn checksum(&self) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(&self.postings_data);
        hasher.update(&self.fst_data);
        let term_meta_bytes = bincode::serialize(&self.term_metadata).unwrap_or_default();
        hasher.update(&term_meta_bytes);
        hasher.update(&self.docno_map_data);
        hasher.update(&self.stats_data);
        hasher.update(&self.docvalues_data);
        hasher.finalize() as u64
    }
}

/// Writer for creating new segments from a mutable buffer
pub struct SegmentWriter {
    segment_id: SegmentId,
}

impl SegmentWriter {
    pub fn new(segment_id: SegmentId) -> Self {
        Self { segment_id }
    }

    /// Write a segment from a mutable buffer
    pub fn write_from_buffer(&self, buffer: &MutableBuffer) -> io::Result<SegmentWriteResult> {
        // Sort terms for FST
        let mut terms: Vec<_> = buffer.all_postings().keys().cloned().collect();
        terms.sort();

        // Write postings and build term dictionary
        let mut postings_writer = PostingsWriter::new();
        let mut term_builder = TermDictionaryBuilder::with_capacity(terms.len());
        let mut total_postings = 0u64;

        for term in &terms {
            if let Some(postings) = buffer.get_postings(term) {
                postings_writer.start_posting_list();

                // Filter out deleted documents and compute stats
                let mut doc_frequency = 0u32;
                let mut total_term_frequency = 0u64;

                for posting in postings {
                    if !buffer.is_deleted(posting.docno) {
                        postings_writer.add_posting(posting.clone());
                        doc_frequency += 1;
                        total_term_frequency += posting.term_frequency as u64;
                    }
                }

                if doc_frequency > 0 {
                    let meta =
                        postings_writer.finish_posting_list(doc_frequency, total_term_frequency);
                    term_builder.add(term.clone(), meta);
                    total_postings += doc_frequency as u64;
                }
            }
        }

        let postings_data = postings_writer.into_data();
        let term_dict = term_builder.build()?;

        // Build DocNo map
        let mut docno_map = DocNoMap::new();
        for entry in buffer.docno_map() {
            docno_map.add(entry.doc_id, entry.version);
        }
        // Apply deletes
        for (i, &deleted) in buffer.deleted_flags().iter().enumerate() {
            if deleted {
                docno_map.delete(DocNo::new(i as u32));
            }
        }

        // Clone statistics
        let stats = SegmentStatistics::from_doc_lengths(buffer.stats().doc_lengths().to_vec());

        // Create segment metadata
        let (min_raft, max_raft) = buffer.raft_index_range();
        let meta = SegmentMeta {
            id: self.segment_id,
            min_raft_index: min_raft.unwrap_or(0),
            max_raft_index: max_raft.unwrap_or(0),
            doc_count: buffer.doc_count(),
            live_doc_count: buffer.live_doc_count(),
            size_bytes: postings_data.len() as u64,
            created_at: current_timestamp(),
        };

        // Serialize components for optional persistence
        let fst_data = term_dict.fst_bytes().to_vec();
        let term_metadata = term_dict.metadata().to_vec();
        let docno_map_data = docno_map.serialize();
        let stats_data = bincode::serialize(&stats)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let docvalues =
            DocValuesReader::from_rows(buffer.all_docvalues(), buffer.doc_count() as usize);
        let docvalues_data = docvalues.serialize();

        // Create the segment reader
        let reader = SegmentReader::from_memory(
            meta,
            term_dict,
            PostingsReader::new(postings_data.clone()),
            docvalues,
            stats,
            docno_map,
        );

        Ok(SegmentWriteResult {
            reader,
            postings_data,
            fst_data,
            term_metadata,
            docno_map_data,
            stats_data,
            docvalues_data,
        })
    }

    /// Merge multiple segments into a new segment
    pub fn merge_segments(&self, segments: &[&SegmentReader]) -> io::Result<SegmentWriteResult> {
        if segments.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "No segments to merge",
            ));
        }

        // Collect all unique terms from all segments
        let mut all_terms: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for segment in segments {
            for (term, _) in segment.terms().iter_terms() {
                all_terms.insert(term);
            }
        }

        // Build merged docno map and track remapping
        let mut merged_docno_map = DocNoMap::new();
        let mut segment_docno_remaps: Vec<std::collections::HashMap<DocNo, DocNo>> = Vec::new();

        for segment in segments {
            let mut remap = std::collections::HashMap::new();
            for (old_docno, entry) in segment.docno_map().live_docs() {
                let new_docno = merged_docno_map.add(entry.doc_id, entry.version);
                remap.insert(old_docno, new_docno);
            }
            segment_docno_remaps.push(remap);
        }

        // Build merged statistics
        let mut merged_stats = SegmentStatistics::new();
        for segment in segments {
            for (old_docno, _) in segment.docno_map().live_docs() {
                if let Some(doc_len) = segment.get_doc_length(old_docno) {
                    merged_stats.add_document(doc_len);
                }
            }
        }

        // Merge posting lists for each term
        let mut postings_writer = PostingsWriter::new();
        let mut term_builder = TermDictionaryBuilder::with_capacity(all_terms.len());

        for term in &all_terms {
            postings_writer.start_posting_list();
            let mut doc_frequency = 0u32;
            let mut total_term_frequency = 0u64;

            // Collect and merge postings from all segments
            let mut merged_postings: Vec<Posting> = Vec::new();

            for (seg_idx, segment) in segments.iter().enumerate() {
                if let Ok(Some(mut iter)) = segment.get_postings(term) {
                    while let Some((old_docno, tf)) = iter.next() {
                        // Only include live documents
                        if let Some(&new_docno) = segment_docno_remaps[seg_idx].get(&old_docno) {
                            merged_postings.push(Posting::new(new_docno, tf));
                        }
                    }
                }
            }

            // Sort by new docno
            merged_postings.sort_by_key(|p| p.docno);

            // Write postings
            for posting in merged_postings {
                postings_writer.add_posting(posting.clone());
                doc_frequency += 1;
                total_term_frequency += posting.term_frequency as u64;
            }

            if doc_frequency > 0 {
                let meta = postings_writer.finish_posting_list(doc_frequency, total_term_frequency);
                term_builder.add(term.clone(), meta);
            }
        }

        let postings_data = postings_writer.into_data();
        let term_dict = term_builder.build()?;

        // Calculate merged metadata
        let min_raft = segments
            .iter()
            .map(|s| s.meta().min_raft_index)
            .min()
            .unwrap_or(0);
        let max_raft = segments
            .iter()
            .map(|s| s.meta().max_raft_index)
            .max()
            .unwrap_or(0);

        let meta = SegmentMeta {
            id: self.segment_id,
            min_raft_index: min_raft,
            max_raft_index: max_raft,
            doc_count: merged_docno_map.len() as u32,
            live_doc_count: merged_docno_map.live_count() as u32,
            size_bytes: postings_data.len() as u64,
            created_at: current_timestamp(),
        };

        let fst_data = term_dict.fst_bytes().to_vec();
        let term_metadata = term_dict.metadata().to_vec();
        let docno_map_data = merged_docno_map.serialize();
        let stats_data = bincode::serialize(&merged_stats)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let merged_doc_count = merged_docno_map.len();
        let mut doc_sources: Vec<Option<(usize, DocNo)>> = vec![None; merged_doc_count];
        for (seg_idx, remap) in segment_docno_remaps.iter().enumerate() {
            for (old_docno, &new_docno) in remap.iter() {
                let idx = new_docno.as_usize();
                if idx < doc_sources.len() {
                    doc_sources[idx] = Some((seg_idx, *old_docno));
                }
            }
        }

        let mut merged_docvalues = DocValuesReader::new();

        let mut numeric_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut boolean_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut keyword_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();

        for segment in segments {
            for name in segment.docvalues().numeric_columns().keys() {
                numeric_names.insert(name.clone());
            }
            for name in segment.docvalues().boolean_columns().keys() {
                boolean_names.insert(name.clone());
            }
            for name in segment.docvalues().keyword_columns().keys() {
                keyword_names.insert(name.clone());
            }
        }

        for name in numeric_names {
            let mut column = NumericColumn::with_capacity(merged_doc_count);
            for docno_idx in 0..merged_doc_count {
                let value = doc_sources[docno_idx].and_then(|(seg_idx, old_docno)| {
                    segments[seg_idx]
                        .docvalues()
                        .get_numeric(&name)
                        .and_then(|c| c.get(old_docno))
                });
                column.add(value);
            }
            merged_docvalues.add_numeric(name, column);
        }

        for name in boolean_names {
            let mut column = BooleanColumn::new();
            for docno_idx in 0..merged_doc_count {
                let value = doc_sources[docno_idx].and_then(|(seg_idx, old_docno)| {
                    segments[seg_idx]
                        .docvalues()
                        .get_boolean(&name)
                        .and_then(|c| c.get(old_docno))
                });
                column.add(value);
            }
            merged_docvalues.add_boolean(name, column);
        }

        for name in keyword_names {
            let mut column = KeywordColumn::new();
            for docno_idx in 0..merged_doc_count {
                let value = doc_sources[docno_idx].and_then(|(seg_idx, old_docno)| {
                    segments[seg_idx]
                        .docvalues()
                        .get_keyword(&name)
                        .and_then(|c| c.get(old_docno))
                });
                column.add(value);
            }
            merged_docvalues.add_keyword(name, column);
        }

        let docvalues_data = merged_docvalues.serialize();

        let reader = SegmentReader::from_memory(
            meta,
            term_dict,
            PostingsReader::new(postings_data.clone()),
            merged_docvalues,
            merged_stats,
            merged_docno_map,
        );

        Ok(SegmentWriteResult {
            reader,
            postings_data,
            fst_data,
            term_metadata,
            docno_map_data,
            stats_data,
            docvalues_data,
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
    use std::collections::HashMap;

    #[test]
    fn test_write_from_buffer() {
        let mut buffer = MutableBuffer::new();

        // Add some documents
        let mut term_freqs1 = HashMap::new();
        term_freqs1.insert("hello".to_string(), 2);
        term_freqs1.insert("world".to_string(), 1);
        buffer.index_document(100, Version::new(1), term_freqs1, 50, None, 1);

        let mut term_freqs2 = HashMap::new();
        term_freqs2.insert("hello".to_string(), 1);
        term_freqs2.insert("rust".to_string(), 3);
        buffer.index_document(200, Version::new(1), term_freqs2, 75, None, 2);

        let mut term_freqs3 = HashMap::new();
        term_freqs3.insert("world".to_string(), 2);
        term_freqs3.insert("rust".to_string(), 1);
        buffer.index_document(300, Version::new(1), term_freqs3, 60, None, 3);

        // Write segment
        let writer = SegmentWriter::new(SegmentId::new(1));
        let result = writer.write_from_buffer(&buffer).unwrap();

        // Verify
        assert_eq!(result.reader.doc_count(), 3);
        assert_eq!(result.reader.live_doc_count(), 3);
        assert_eq!(result.reader.term_count(), 3); // hello, world, rust

        // Check postings
        assert_eq!(result.reader.doc_frequency("hello"), 2);
        assert_eq!(result.reader.doc_frequency("world"), 2);
        assert_eq!(result.reader.doc_frequency("rust"), 2);
    }

    #[test]
    fn test_write_with_deletes() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs1 = HashMap::new();
        term_freqs1.insert("test".to_string(), 1);
        buffer.index_document(100, Version::new(1), term_freqs1, 50, None, 1);

        let mut term_freqs2 = HashMap::new();
        term_freqs2.insert("test".to_string(), 2);
        buffer.index_document(200, Version::new(1), term_freqs2, 75, None, 2);

        // Delete one document
        buffer.delete_document(100, 3);

        let writer = SegmentWriter::new(SegmentId::new(1));
        let result = writer.write_from_buffer(&buffer).unwrap();

        // Document count includes deleted, live_doc_count excludes them
        assert_eq!(result.reader.doc_count(), 2);
        assert_eq!(result.reader.live_doc_count(), 1);

        // Only one document should be in postings for "test"
        assert_eq!(result.reader.doc_frequency("test"), 1);
    }

    #[test]
    fn test_segment_write_result_data() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs = HashMap::new();
        term_freqs.insert("hello".to_string(), 1);
        buffer.index_document(100, Version::new(1), term_freqs, 50, None, 1);

        let writer = SegmentWriter::new(SegmentId::new(1));
        let result = writer.write_from_buffer(&buffer).unwrap();

        // Verify data was generated
        assert!(!result.postings_data.is_empty());
        assert!(!result.fst_data.is_empty());
        assert!(!result.docno_map_data.is_empty());
        assert!(!result.stats_data.is_empty());
        assert!(!result.docvalues_data.is_empty());
    }

    #[test]
    fn test_write_docvalues() {
        let mut buffer = MutableBuffer::new();

        let mut term_freqs = HashMap::new();
        term_freqs.insert("hello".to_string(), 1);

        let mut docvalues = super::super::types::DocValueRow::new();
        docvalues.numerics = vec![Some(42)];
        docvalues.booleans = vec![Some(true)];
        docvalues.keywords = vec![Some("alpha".to_string())];

        buffer.index_document(100, Version::new(1), term_freqs, 50, Some(docvalues), 1);

        let writer = SegmentWriter::new(SegmentId::new(1));
        let result = writer.write_from_buffer(&buffer).unwrap();

        let dv = result.reader.docvalues();
        assert_eq!(
            dv.get_numeric("numeric_0").unwrap().get(DocNo::new(0)),
            Some(42)
        );
        assert_eq!(
            dv.get_boolean("boolean_0").unwrap().get(DocNo::new(0)),
            Some(true)
        );
        assert_eq!(
            dv.get_keyword("keyword_0").unwrap().get(DocNo::new(0)),
            Some("alpha")
        );
    }

    #[test]
    fn test_merge_segments() {
        // Create first segment
        let mut buffer1 = MutableBuffer::new();
        let mut tf1 = HashMap::new();
        tf1.insert("rust".to_string(), 3);
        tf1.insert("programming".to_string(), 2);
        buffer1.index_document(100, Version::new(1), tf1, 50, None, 1);

        let mut tf2 = HashMap::new();
        tf2.insert("rust".to_string(), 1);
        buffer1.index_document(200, Version::new(1), tf2, 30, None, 2);

        let writer1 = SegmentWriter::new(SegmentId::new(1));
        let result1 = writer1.write_from_buffer(&buffer1).unwrap();

        // Create second segment
        let mut buffer2 = MutableBuffer::new();
        let mut tf3 = HashMap::new();
        tf3.insert("rust".to_string(), 2);
        tf3.insert("language".to_string(), 4);
        buffer2.index_document(300, Version::new(1), tf3, 60, None, 3);

        let mut tf4 = HashMap::new();
        tf4.insert("programming".to_string(), 1);
        tf4.insert("language".to_string(), 2);
        buffer2.index_document(400, Version::new(1), tf4, 40, None, 4);

        let writer2 = SegmentWriter::new(SegmentId::new(2));
        let result2 = writer2.write_from_buffer(&buffer2).unwrap();

        // Merge segments
        let merge_writer = SegmentWriter::new(SegmentId::new(3));
        let segments = vec![&result1.reader, &result2.reader];
        let merged = merge_writer.merge_segments(&segments).unwrap();

        // Verify merged segment
        assert_eq!(merged.reader.live_doc_count(), 4);
        assert_eq!(merged.reader.doc_frequency("rust"), 3); // docs 100, 200, 300
        assert_eq!(merged.reader.doc_frequency("programming"), 2); // docs 100, 400
        assert_eq!(merged.reader.doc_frequency("language"), 2); // docs 300, 400

        // Verify we can iterate postings
        let mut rust_iter = merged.reader.get_postings("rust").unwrap().unwrap();
        let rust_postings: Vec<_> = rust_iter.collect();
        assert_eq!(rust_postings.len(), 3);
    }

    #[test]
    fn test_merge_with_deletes() {
        // Create segment with deleted doc
        let mut buffer1 = MutableBuffer::new();
        let mut tf1 = HashMap::new();
        tf1.insert("hello".to_string(), 1);
        buffer1.index_document(100, Version::new(1), tf1, 50, None, 1);

        let mut tf2 = HashMap::new();
        tf2.insert("hello".to_string(), 2);
        buffer1.index_document(200, Version::new(1), tf2, 50, None, 2);

        buffer1.delete_document(100, 3); // Delete doc 100

        let writer1 = SegmentWriter::new(SegmentId::new(1));
        let result1 = writer1.write_from_buffer(&buffer1).unwrap();

        // Create second segment
        let mut buffer2 = MutableBuffer::new();
        let mut tf3 = HashMap::new();
        tf3.insert("hello".to_string(), 3);
        buffer2.index_document(300, Version::new(1), tf3, 50, None, 4);

        let writer2 = SegmentWriter::new(SegmentId::new(2));
        let result2 = writer2.write_from_buffer(&buffer2).unwrap();

        // Merge segments
        let merge_writer = SegmentWriter::new(SegmentId::new(3));
        let segments = vec![&result1.reader, &result2.reader];
        let merged = merge_writer.merge_segments(&segments).unwrap();

        // Only live docs should be in merged segment
        assert_eq!(merged.reader.live_doc_count(), 2); // 200 and 300
        assert_eq!(merged.reader.doc_frequency("hello"), 2); // Only docs 200 and 300
    }
}
