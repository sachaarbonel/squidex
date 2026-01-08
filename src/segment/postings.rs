//! Postings format with block-based compression
//!
//! Per SPEC: store postings in fixed-size blocks (128/256 docs) with:
//! - docID deltas: bitpacked
//! - TF: bitpacked
//! - positions: separate stream
//! - skip data per block
//! - impact metadata per block for WAND/MaxScore

use std::io::{self, Read, Write};

use super::types::{DocNo, Posting, PostingBlock, PostingListMeta, BLOCK_SIZE};

/// Variable-byte encoding for integers (commonly used in search engines)
pub fn encode_vbyte(value: u32, output: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            output.push(byte | 0x80); // Set high bit to indicate last byte
            break;
        } else {
            output.push(byte);
        }
    }
}

/// Decode a variable-byte encoded integer
pub fn decode_vbyte(input: &[u8], pos: &mut usize) -> io::Result<u32> {
    let mut result: u32 = 0;
    let mut shift = 0;

    loop {
        if *pos >= input.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected end of vbyte",
            ));
        }

        let byte = input[*pos];
        *pos += 1;

        result |= ((byte & 0x7F) as u32) << shift;

        if byte & 0x80 != 0 {
            return Ok(result);
        }

        shift += 7;
        if shift > 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "VByte value too large",
            ));
        }
    }
}

/// Simple bitpacking for a block of integers
/// Uses the minimum number of bits needed to represent the max value
pub fn bitpack_encode(values: &[u32], output: &mut Vec<u8>) {
    if values.is_empty() {
        output.push(0); // 0 bits needed
        return;
    }

    let max_val = *values.iter().max().unwrap();
    let bits_needed = if max_val == 0 {
        1
    } else {
        32 - max_val.leading_zeros()
    } as u8;

    output.push(bits_needed);

    // Pack values into bytes
    let mut current_byte: u64 = 0;
    let mut bits_in_current = 0;

    for &value in values {
        current_byte |= (value as u64) << bits_in_current;
        bits_in_current += bits_needed as u32;

        while bits_in_current >= 8 {
            output.push(current_byte as u8);
            current_byte >>= 8;
            bits_in_current -= 8;
        }
    }

    // Flush remaining bits
    if bits_in_current > 0 {
        output.push(current_byte as u8);
    }
}

/// Decode bitpacked integers
pub fn bitpack_decode(input: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<u32>> {
    if *pos >= input.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Unexpected end of bitpack",
        ));
    }

    let bits_needed = input[*pos] as u32;
    *pos += 1;

    if bits_needed == 0 {
        return Ok(vec![0; count]);
    }

    let total_bits = count as u32 * bits_needed;
    let bytes_needed = ((total_bits + 7) / 8) as usize;

    if *pos + bytes_needed > input.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Not enough bytes for bitpack",
        ));
    }

    let mut values = Vec::with_capacity(count);
    let mut current: u64 = 0;
    let mut bits_available = 0;
    let mask = (1u64 << bits_needed) - 1;

    for _ in 0..count {
        while bits_available < bits_needed {
            if *pos < input.len() {
                current |= (input[*pos] as u64) << bits_available;
                *pos += 1;
            }
            bits_available += 8;
        }

        values.push((current & mask) as u32);
        current >>= bits_needed;
        bits_available -= bits_needed;
    }

    Ok(values)
}

/// Skip data for fast block skipping during query processing
#[derive(Clone, Debug, Default)]
pub struct SkipEntry {
    /// Maximum docno in this block
    pub max_docno: DocNo,
    /// Offset to this block in the postings data
    pub block_offset: u64,
    /// Maximum score contribution (for WAND/MaxScore)
    pub max_score: f32,
}

/// Writer for posting lists
pub struct PostingsWriter {
    /// Accumulated blocks for current posting list
    block_data: Vec<u8>,
    /// Skip entries for the current term
    skip_entries: Vec<SkipEntry>,
    /// Current block being built
    current_block: PostingBlock,
    /// Final output data
    data: Vec<u8>,
}

impl PostingsWriter {
    pub fn new() -> Self {
        Self {
            block_data: Vec::new(),
            skip_entries: Vec::new(),
            current_block: PostingBlock::new(),
            data: Vec::new(),
        }
    }

    /// Start writing a new posting list
    pub fn start_posting_list(&mut self) {
        self.block_data.clear();
        self.skip_entries.clear();
        self.current_block = PostingBlock::new();
    }

    /// Add a posting to the current list
    pub fn add_posting(&mut self, posting: Posting) {
        self.current_block.push(posting);

        if self.current_block.is_full() {
            self.flush_block();
        }
    }

    /// Finish writing a posting list and return metadata
    pub fn finish_posting_list(&mut self, doc_frequency: u32, total_term_frequency: u64) -> PostingListMeta {
        // Flush any remaining postings
        if !self.current_block.is_empty() {
            self.flush_block();
        }

        let offset = self.data.len() as u64;

        // Write block count first
        encode_vbyte(self.skip_entries.len() as u32, &mut self.data);

        // Write skip entries (for efficient skip-to)
        for skip in &self.skip_entries {
            encode_vbyte(skip.max_docno.0, &mut self.data);
            self.data.extend_from_slice(&skip.block_offset.to_le_bytes());
            self.data.extend_from_slice(&skip.max_score.to_le_bytes());
        }

        // Write all blocks
        self.data.extend_from_slice(&self.block_data);

        let length = self.data.len() as u64 - offset;

        PostingListMeta {
            offset,
            length,
            doc_frequency,
            total_term_frequency,
        }
    }

    /// Get the current data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Take the data (consuming the writer)
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    fn flush_block(&mut self) {
        if self.current_block.is_empty() {
            return;
        }

        let block_start = self.block_data.len() as u64;

        // Record skip entry with max TF for WAND scoring
        self.skip_entries.push(SkipEntry {
            max_docno: self.current_block.max_docno,
            block_offset: block_start,
            max_score: self.current_block.max_tf as f32, // Store max_tf as float for WAND
        });

        // Write block header (count)
        encode_vbyte(self.current_block.len() as u32, &mut self.block_data);

        // Delta-encode and write docnos
        let mut deltas = Vec::with_capacity(self.current_block.len());
        let mut prev = 0u32;
        for docno in &self.current_block.docnos {
            deltas.push(docno.0 - prev);
            prev = docno.0;
        }
        bitpack_encode(&deltas, &mut self.block_data);

        // Write term frequencies
        bitpack_encode(&self.current_block.term_frequencies, &mut self.block_data);

        // Write max_tf (for per-block WAND bounds)
        self.block_data.extend_from_slice(&self.current_block.max_tf.to_le_bytes());

        // Clear block for next batch
        self.current_block = PostingBlock::new();
    }
}

impl Default for PostingsWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader for posting lists
pub struct PostingsReader {
    /// The postings data (typically mmapped)
    data: Vec<u8>,
}

impl PostingsReader {
    /// Create a reader from data
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Create a reader from a byte slice (copies the data)
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            data: bytes.to_vec(),
        }
    }

    /// Get an iterator over postings for a term
    pub fn get_postings(&self, meta: &PostingListMeta) -> io::Result<PostingIterator> {
        let start = meta.offset as usize;
        let end = (meta.offset + meta.length) as usize;

        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Posting list extends beyond data",
            ));
        }

        Ok(PostingIterator::new(&self.data[start..end]))
    }

    /// Get the underlying data
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// Iterator over postings in a posting list
pub struct PostingIterator<'a> {
    data: &'a [u8],
    /// Position in data where blocks start (after skip entries)
    blocks_start: usize,
    /// Current position in data
    pos: usize,
    /// Total number of blocks
    block_count: usize,
    /// Skip entries for efficient seeking
    skip_entries: Vec<SkipEntry>,
    /// Current block index
    current_block_idx: usize,
    /// Current block contents
    current_block: Vec<(DocNo, u32)>,
    /// Position within current block
    block_pos: usize,
    /// Last docno seen (for delta decoding)
    last_docno: u32,
}

impl<'a> PostingIterator<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut pos = 0;
        let block_count = decode_vbyte(data, &mut pos).unwrap_or(0) as usize;

        // Read skip entries
        let mut skip_entries = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let max_docno = decode_vbyte(data, &mut pos).unwrap_or(0);

            let block_offset = if pos + 8 <= data.len() {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data[pos..pos + 8]);
                pos += 8;
                u64::from_le_bytes(bytes)
            } else {
                0
            };

            let max_score = if pos + 4 <= data.len() {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(&data[pos..pos + 4]);
                pos += 4;
                f32::from_le_bytes(bytes)
            } else {
                0.0
            };

            skip_entries.push(SkipEntry {
                max_docno: DocNo(max_docno),
                block_offset,
                max_score,
            });
        }

        let blocks_start = pos;

        Self {
            data,
            blocks_start,
            pos,
            block_count,
            skip_entries,
            current_block_idx: 0,
            current_block: Vec::new(),
            block_pos: 0,
            last_docno: 0,
        }
    }

    /// Skip to the first posting with docno >= target
    pub fn skip_to(&mut self, target: DocNo) -> Option<(DocNo, u32)> {
        // Use skip entries to find the right block
        while self.current_block_idx < self.skip_entries.len() {
            let skip = &self.skip_entries[self.current_block_idx];
            if skip.max_docno >= target {
                // This block might contain our target
                break;
            }
            // Skip this block entirely
            self.current_block_idx += 1;
            if self.current_block_idx < self.skip_entries.len() {
                // Jump to next block's position
                self.pos = self.blocks_start + self.skip_entries[self.current_block_idx].block_offset as usize;
                self.last_docno = if self.current_block_idx > 0 {
                    self.skip_entries[self.current_block_idx - 1].max_docno.0
                } else {
                    0
                };
            }
            self.current_block.clear();
            self.block_pos = 0;
        }

        // Linear scan within the current/next blocks
        while let Some((docno, tf)) = self.next() {
            if docno >= target {
                return Some((docno, tf));
            }
        }
        None
    }

    /// Get maximum score for remaining blocks (for WAND optimization)
    pub fn max_remaining_score(&self) -> f32 {
        self.skip_entries[self.current_block_idx..]
            .iter()
            .map(|s| s.max_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    fn load_next_block(&mut self) -> bool {
        if self.current_block_idx >= self.block_count || self.pos >= self.data.len() {
            return false;
        }

        let count = match decode_vbyte(self.data, &mut self.pos) {
            Ok(c) => c as usize,
            Err(_) => return false,
        };

        if count == 0 {
            return false;
        }

        // Decode docno deltas
        let deltas = match bitpack_decode(self.data, &mut self.pos, count) {
            Ok(d) => d,
            Err(_) => return false,
        };

        // Decode term frequencies
        let tfs = match bitpack_decode(self.data, &mut self.pos, count) {
            Ok(t) => t,
            Err(_) => return false,
        };

        // Skip max_tf (4 bytes)
        if self.pos + 4 <= self.data.len() {
            self.pos += 4;
        }

        // Reconstruct docnos from deltas
        self.current_block.clear();
        let mut docno = self.last_docno;
        for i in 0..count {
            docno += deltas[i];
            self.current_block.push((DocNo(docno), tfs[i]));
        }

        if !self.current_block.is_empty() {
            self.last_docno = self.current_block.last().unwrap().0 .0;
        }

        self.block_pos = 0;
        self.current_block_idx += 1;
        true
    }
}

impl<'a> Iterator for PostingIterator<'a> {
    type Item = (DocNo, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_pos >= self.current_block.len() {
            if !self.load_next_block() {
                return None;
            }
        }

        let result = self.current_block.get(self.block_pos).copied();
        self.block_pos += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vbyte_encoding() {
        let mut output = Vec::new();

        encode_vbyte(0, &mut output);
        encode_vbyte(127, &mut output);
        encode_vbyte(128, &mut output);
        encode_vbyte(16383, &mut output);
        encode_vbyte(1_000_000, &mut output);

        let mut pos = 0;
        assert_eq!(decode_vbyte(&output, &mut pos).unwrap(), 0);
        assert_eq!(decode_vbyte(&output, &mut pos).unwrap(), 127);
        assert_eq!(decode_vbyte(&output, &mut pos).unwrap(), 128);
        assert_eq!(decode_vbyte(&output, &mut pos).unwrap(), 16383);
        assert_eq!(decode_vbyte(&output, &mut pos).unwrap(), 1_000_000);
    }

    #[test]
    fn test_bitpack_encoding() {
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut output = Vec::new();

        bitpack_encode(&values, &mut output);

        let mut pos = 0;
        let decoded = bitpack_decode(&output, &mut pos, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_bitpack_large_values() {
        let values = vec![1000, 2000, 3000, 4000];
        let mut output = Vec::new();

        bitpack_encode(&values, &mut output);

        let mut pos = 0;
        let decoded = bitpack_decode(&output, &mut pos, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_postings_writer_reader() {
        let mut writer = PostingsWriter::new();
        writer.start_posting_list();

        // Add some postings
        for i in 0..10 {
            writer.add_posting(Posting::new(DocNo(i * 10), (i + 1) as u32));
        }

        let meta = writer.finish_posting_list(10, 55);

        // Read back
        let reader = PostingsReader::new(writer.into_data());
        let iter = reader.get_postings(&meta).unwrap();

        let postings: Vec<_> = iter.collect();
        assert_eq!(postings.len(), 10);
        assert_eq!(postings[0], (DocNo(0), 1));
        assert_eq!(postings[5], (DocNo(50), 6));
        assert_eq!(postings[9], (DocNo(90), 10));
    }

    #[test]
    fn test_postings_skip() {
        let mut writer = PostingsWriter::new();
        writer.start_posting_list();

        // Add enough postings to create multiple blocks
        for i in 0..300 {
            writer.add_posting(Posting::new(DocNo(i * 2), 1));
        }

        let meta = writer.finish_posting_list(300, 300);

        let reader = PostingsReader::new(writer.into_data());
        let mut iter = reader.get_postings(&meta).unwrap();

        // Skip to docno 400
        let result = iter.skip_to(DocNo(400));
        assert!(result.is_some());
        let (docno, _) = result.unwrap();
        assert!(docno >= DocNo(400));
    }
}
