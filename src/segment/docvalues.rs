//! DocValues for columnar storage
//!
//! typed column stores with:
//! - numeric: bitpacked blocks + optional delta coding + per-block min/max
//! - boolean: roaring bitmap or dense bitset
//! - keyword/string: dictionary encode to ordinals + bitpacked ordinals
//! - nulls: explicit bitmap

use std::collections::HashMap;
use std::io;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use super::postings::{bitpack_decode, bitpack_encode, decode_vbyte, encode_vbyte};
use super::types::{DocNo, DocValueRow};

/// Column types for doc values
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ColumnType {
    Numeric,
    Boolean,
    Keyword,
}

/// Numeric column with bitpacked values
#[derive(Clone, Debug)]
pub struct NumericColumn {
    /// Values indexed by docno (None for missing values)
    values: Vec<Option<i64>>,
    /// Per-block min/max for range queries
    block_bounds: Vec<(i64, i64)>,
    /// Null bitmap
    nulls: RoaringBitmap,
    /// Global min value
    min_value: Option<i64>,
    /// Global max value
    max_value: Option<i64>,
}

impl NumericColumn {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            block_bounds: Vec::new(),
            nulls: RoaringBitmap::new(),
            min_value: None,
            max_value: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            block_bounds: Vec::new(),
            nulls: RoaringBitmap::new(),
            min_value: None,
            max_value: None,
        }
    }

    /// Add a value for the next docno
    pub fn add(&mut self, value: Option<i64>) {
        let docno = self.values.len() as u32;

        if let Some(v) = value {
            self.min_value = Some(self.min_value.map_or(v, |m| m.min(v)));
            self.max_value = Some(self.max_value.map_or(v, |m| m.max(v)));
        } else {
            self.nulls.insert(docno);
        }

        self.values.push(value);
    }

    /// Get value for a docno
    pub fn get(&self, docno: DocNo) -> Option<i64> {
        self.values.get(docno.as_usize()).copied().flatten()
    }

    /// Check if a docno has a null value
    pub fn is_null(&self, docno: DocNo) -> bool {
        self.nulls.contains(docno.as_u32())
    }

    /// Get documents with value in range [min, max]
    pub fn range_query(&self, min: i64, max: i64) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for (docno, value) in self.values.iter().enumerate() {
            if let Some(v) = value {
                if *v >= min && *v <= max {
                    result.insert(docno as u32);
                }
            }
        }
        result
    }

    /// Get documents equal to value
    pub fn equals_query(&self, value: i64) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for (docno, v) in self.values.iter().enumerate() {
            if *v == Some(value) {
                result.insert(docno as u32);
            }
        }
        result
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Write count
        encode_vbyte(self.values.len() as u32, &mut output);

        // Write min and max
        if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
            output.push(1); // has values
            output.extend_from_slice(&min.to_le_bytes());
            output.extend_from_slice(&max.to_le_bytes());
        } else {
            output.push(0); // no values
        }

        // Write null bitmap
        let mut null_bytes = Vec::new();
        self.nulls.serialize_into(&mut null_bytes).unwrap();
        encode_vbyte(null_bytes.len() as u32, &mut output);
        output.extend(null_bytes);

        // Write values (delta encoded from min if available)
        if let Some(min) = self.min_value {
            let offsets: Vec<u32> = self
                .values
                .iter()
                .map(|v| v.map(|x| (x - min) as u32).unwrap_or(0))
                .collect();
            bitpack_encode(&offsets, &mut output);
        }

        output
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;

        let count = decode_vbyte(data, &mut pos)? as usize;

        let has_values = data.get(pos).copied().unwrap_or(0) == 1;
        pos += 1;

        let (min_value, max_value) = if has_values {
            let mut min_bytes = [0u8; 8];
            min_bytes.copy_from_slice(&data[pos..pos + 8]);
            pos += 8;
            let mut max_bytes = [0u8; 8];
            max_bytes.copy_from_slice(&data[pos..pos + 8]);
            pos += 8;
            (
                Some(i64::from_le_bytes(min_bytes)),
                Some(i64::from_le_bytes(max_bytes)),
            )
        } else {
            (None, None)
        };

        let null_len = decode_vbyte(data, &mut pos)? as usize;
        let nulls = RoaringBitmap::deserialize_from(&data[pos..pos + null_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        pos += null_len;

        let mut values = Vec::with_capacity(count);
        if let Some(min) = min_value {
            let offsets = bitpack_decode(data, &mut pos, count)?;
            for (docno, offset) in offsets.into_iter().enumerate() {
                if nulls.contains(docno as u32) {
                    values.push(None);
                } else {
                    values.push(Some(min + offset as i64));
                }
            }
        } else {
            values.resize(count, None);
        }

        Ok(Self {
            values,
            block_bounds: Vec::new(),
            nulls,
            min_value,
            max_value,
        })
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for NumericColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Boolean column using roaring bitmap
#[derive(Clone, Debug)]
pub struct BooleanColumn {
    /// Bitmap of true values
    true_values: RoaringBitmap,
    /// Bitmap of null values
    nulls: RoaringBitmap,
    /// Total document count
    doc_count: u32,
}

impl BooleanColumn {
    pub fn new() -> Self {
        Self {
            true_values: RoaringBitmap::new(),
            nulls: RoaringBitmap::new(),
            doc_count: 0,
        }
    }

    /// Add a value for the next docno
    pub fn add(&mut self, value: Option<bool>) {
        let docno = self.doc_count;
        match value {
            Some(true) => {
                self.true_values.insert(docno);
            }
            Some(false) => {
                // False is implicit (not in true_values, not in nulls)
            }
            None => {
                self.nulls.insert(docno);
            }
        }
        self.doc_count += 1;
    }

    /// Get value for a docno
    pub fn get(&self, docno: DocNo) -> Option<bool> {
        if self.nulls.contains(docno.as_u32()) {
            None
        } else {
            Some(self.true_values.contains(docno.as_u32()))
        }
    }

    /// Get documents with true value
    pub fn true_docs(&self) -> &RoaringBitmap {
        &self.true_values
    }

    /// Get documents with false value
    pub fn false_docs(&self) -> RoaringBitmap {
        let all: RoaringBitmap = (0..self.doc_count).collect();
        &(&all - &self.nulls) - &self.true_values
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::new();

        encode_vbyte(self.doc_count, &mut output);

        let mut true_bytes = Vec::new();
        self.true_values.serialize_into(&mut true_bytes).unwrap();
        encode_vbyte(true_bytes.len() as u32, &mut output);
        output.extend(true_bytes);

        let mut null_bytes = Vec::new();
        self.nulls.serialize_into(&mut null_bytes).unwrap();
        encode_vbyte(null_bytes.len() as u32, &mut output);
        output.extend(null_bytes);

        output
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;

        let doc_count = decode_vbyte(data, &mut pos)?;

        let true_len = decode_vbyte(data, &mut pos)? as usize;
        let true_values = RoaringBitmap::deserialize_from(&data[pos..pos + true_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        pos += true_len;

        let null_len = decode_vbyte(data, &mut pos)? as usize;
        let nulls = RoaringBitmap::deserialize_from(&data[pos..pos + null_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Self {
            true_values,
            nulls,
            doc_count,
        })
    }

    pub fn len(&self) -> usize {
        self.doc_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
}

impl Default for BooleanColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Keyword column with dictionary encoding
#[derive(Clone, Debug)]
pub struct KeywordColumn {
    /// Dictionary: ordinal -> keyword
    dictionary: Vec<String>,
    /// Reverse lookup: keyword -> ordinal
    keyword_to_ordinal: HashMap<String, u32>,
    /// Ordinals indexed by docno
    ordinals: Vec<Option<u32>>,
    /// Null bitmap
    nulls: RoaringBitmap,
}

impl KeywordColumn {
    pub fn new() -> Self {
        Self {
            dictionary: Vec::new(),
            keyword_to_ordinal: HashMap::new(),
            ordinals: Vec::new(),
            nulls: RoaringBitmap::new(),
        }
    }

    /// Add a value for the next docno
    pub fn add(&mut self, value: Option<&str>) {
        let docno = self.ordinals.len() as u32;

        match value {
            Some(keyword) => {
                let ordinal = if let Some(&ord) = self.keyword_to_ordinal.get(keyword) {
                    ord
                } else {
                    let ord = self.dictionary.len() as u32;
                    self.dictionary.push(keyword.to_string());
                    self.keyword_to_ordinal.insert(keyword.to_string(), ord);
                    ord
                };
                self.ordinals.push(Some(ordinal));
            }
            None => {
                self.nulls.insert(docno);
                self.ordinals.push(None);
            }
        }
    }

    /// Get value for a docno
    pub fn get(&self, docno: DocNo) -> Option<&str> {
        self.ordinals
            .get(docno.as_usize())
            .and_then(|ord| ord.as_ref())
            .and_then(|&ord| self.dictionary.get(ord as usize))
            .map(|s| s.as_str())
    }

    /// Get documents with exact keyword match
    pub fn equals_query(&self, keyword: &str) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();

        if let Some(&ordinal) = self.keyword_to_ordinal.get(keyword) {
            for (docno, ord) in self.ordinals.iter().enumerate() {
                if *ord == Some(ordinal) {
                    result.insert(docno as u32);
                }
            }
        }

        result
    }

    /// Get all unique keywords
    pub fn unique_keywords(&self) -> &[String] {
        &self.dictionary
    }

    /// Serialize to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::new();

        // Write dictionary
        encode_vbyte(self.dictionary.len() as u32, &mut output);
        for keyword in &self.dictionary {
            encode_vbyte(keyword.len() as u32, &mut output);
            output.extend(keyword.as_bytes());
        }

        // Write document count
        encode_vbyte(self.ordinals.len() as u32, &mut output);

        // Write null bitmap
        let mut null_bytes = Vec::new();
        self.nulls.serialize_into(&mut null_bytes).unwrap();
        encode_vbyte(null_bytes.len() as u32, &mut output);
        output.extend(null_bytes);

        // Write ordinals (using 0 for nulls, actual values are ordinal + 1)
        let encoded_ordinals: Vec<u32> = self
            .ordinals
            .iter()
            .map(|ord| ord.map(|o| o + 1).unwrap_or(0))
            .collect();
        bitpack_encode(&encoded_ordinals, &mut output);

        output
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;

        // Read dictionary
        let dict_len = decode_vbyte(data, &mut pos)? as usize;
        let mut dictionary = Vec::with_capacity(dict_len);
        let mut keyword_to_ordinal = HashMap::with_capacity(dict_len);

        for i in 0..dict_len {
            let keyword_len = decode_vbyte(data, &mut pos)? as usize;
            let keyword = std::str::from_utf8(&data[pos..pos + keyword_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += keyword_len;
            keyword_to_ordinal.insert(keyword.clone(), i as u32);
            dictionary.push(keyword);
        }

        // Read document count
        let doc_count = decode_vbyte(data, &mut pos)? as usize;

        // Read null bitmap
        let null_len = decode_vbyte(data, &mut pos)? as usize;
        let nulls = RoaringBitmap::deserialize_from(&data[pos..pos + null_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        pos += null_len;

        // Read ordinals
        let encoded_ordinals = bitpack_decode(data, &mut pos, doc_count)?;
        let ordinals: Vec<Option<u32>> = encoded_ordinals
            .into_iter()
            .enumerate()
            .map(|(docno, v)| {
                if nulls.contains(docno as u32) || v == 0 {
                    None
                } else {
                    Some(v - 1)
                }
            })
            .collect();

        Ok(Self {
            dictionary,
            keyword_to_ordinal,
            ordinals,
            nulls,
        })
    }

    pub fn len(&self) -> usize {
        self.ordinals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordinals.is_empty()
    }
}

impl Default for KeywordColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined doc values reader for a segment
#[derive(Clone, Debug, Default)]
pub struct DocValuesReader {
    numeric_columns: HashMap<String, NumericColumn>,
    boolean_columns: HashMap<String, BooleanColumn>,
    keyword_columns: HashMap<String, KeywordColumn>,
}

impl DocValuesReader {
    pub fn new() -> Self {
        Self {
            numeric_columns: HashMap::new(),
            boolean_columns: HashMap::new(),
            keyword_columns: HashMap::new(),
        }
    }

    /// Build docvalues from row-oriented data.
    pub fn from_rows(rows: &HashMap<DocNo, DocValueRow>, doc_count: usize) -> Self {
        if rows.is_empty() || doc_count == 0 {
            return Self::new();
        }

        let mut max_numeric = 0usize;
        let mut max_boolean = 0usize;
        let mut max_keyword = 0usize;

        for row in rows.values() {
            max_numeric = max_numeric.max(row.numerics.len());
            max_boolean = max_boolean.max(row.booleans.len());
            max_keyword = max_keyword.max(row.keywords.len());
        }

        let mut numeric_columns = Vec::with_capacity(max_numeric);
        for _ in 0..max_numeric {
            numeric_columns.push(NumericColumn::with_capacity(doc_count));
        }

        let mut boolean_columns = Vec::with_capacity(max_boolean);
        for _ in 0..max_boolean {
            boolean_columns.push(BooleanColumn::new());
        }

        let mut keyword_columns = Vec::with_capacity(max_keyword);
        for _ in 0..max_keyword {
            keyword_columns.push(KeywordColumn::new());
        }

        for docno_idx in 0..doc_count {
            let docno = DocNo::new(docno_idx as u32);
            let row = rows.get(&docno);

            for (idx, column) in numeric_columns.iter_mut().enumerate() {
                let value = row
                    .and_then(|r| r.numerics.get(idx).cloned())
                    .unwrap_or(None);
                column.add(value);
            }

            for (idx, column) in boolean_columns.iter_mut().enumerate() {
                let value = row
                    .and_then(|r| r.booleans.get(idx).cloned())
                    .unwrap_or(None);
                column.add(value);
            }

            for (idx, column) in keyword_columns.iter_mut().enumerate() {
                let value = row
                    .and_then(|r| r.keywords.get(idx).cloned())
                    .unwrap_or(None);
                match value.as_deref() {
                    Some(keyword) => column.add(Some(keyword)),
                    None => column.add(None),
                }
            }
        }

        let mut reader = Self::new();
        for (idx, column) in numeric_columns.into_iter().enumerate() {
            reader.add_numeric(format!("numeric_{}", idx), column);
        }
        for (idx, column) in boolean_columns.into_iter().enumerate() {
            reader.add_boolean(format!("boolean_{}", idx), column);
        }
        for (idx, column) in keyword_columns.into_iter().enumerate() {
            reader.add_keyword(format!("keyword_{}", idx), column);
        }

        reader
    }

    pub fn add_numeric(&mut self, name: String, column: NumericColumn) {
        self.numeric_columns.insert(name, column);
    }

    pub fn add_boolean(&mut self, name: String, column: BooleanColumn) {
        self.boolean_columns.insert(name, column);
    }

    pub fn add_keyword(&mut self, name: String, column: KeywordColumn) {
        self.keyword_columns.insert(name, column);
    }

    pub fn get_numeric(&self, name: &str) -> Option<&NumericColumn> {
        self.numeric_columns.get(name)
    }

    pub fn get_boolean(&self, name: &str) -> Option<&BooleanColumn> {
        self.boolean_columns.get(name)
    }

    pub fn get_keyword(&self, name: &str) -> Option<&KeywordColumn> {
        self.keyword_columns.get(name)
    }

    pub fn numeric_columns(&self) -> &HashMap<String, NumericColumn> {
        &self.numeric_columns
    }

    pub fn boolean_columns(&self) -> &HashMap<String, BooleanColumn> {
        &self.boolean_columns
    }

    pub fn keyword_columns(&self) -> &HashMap<String, KeywordColumn> {
        &self.keyword_columns
    }

    /// Serialize all columns into a single byte stream.
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::new();

        let total_columns =
            self.numeric_columns.len() + self.boolean_columns.len() + self.keyword_columns.len();
        encode_vbyte(total_columns as u32, &mut output);

        let mut numeric_names: Vec<_> = self.numeric_columns.keys().collect();
        numeric_names.sort();
        for name in numeric_names {
            let column = &self.numeric_columns[name];
            output.push(ColumnType::Numeric as u8);
            encode_vbyte(name.len() as u32, &mut output);
            output.extend(name.as_bytes());
            let data = column.serialize();
            encode_vbyte(data.len() as u32, &mut output);
            output.extend(data);
        }

        let mut boolean_names: Vec<_> = self.boolean_columns.keys().collect();
        boolean_names.sort();
        for name in boolean_names {
            let column = &self.boolean_columns[name];
            output.push(ColumnType::Boolean as u8);
            encode_vbyte(name.len() as u32, &mut output);
            output.extend(name.as_bytes());
            let data = column.serialize();
            encode_vbyte(data.len() as u32, &mut output);
            output.extend(data);
        }

        let mut keyword_names: Vec<_> = self.keyword_columns.keys().collect();
        keyword_names.sort();
        for name in keyword_names {
            let column = &self.keyword_columns[name];
            output.push(ColumnType::Keyword as u8);
            encode_vbyte(name.len() as u32, &mut output);
            output.extend(name.as_bytes());
            let data = column.serialize();
            encode_vbyte(data.len() as u32, &mut output);
            output.extend(data);
        }

        output
    }

    /// Deserialize columns from a byte stream.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Ok(Self::new());
        }

        let mut pos = 0;
        let column_count = decode_vbyte(data, &mut pos)? as usize;
        let mut reader = Self::new();

        for _ in 0..column_count {
            let col_type = data.get(pos).copied().ok_or_else(|| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "Missing column type")
            })?;
            pos += 1;

            let name_len = decode_vbyte(data, &mut pos)? as usize;
            if pos + name_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Truncated column name",
                ));
            }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += name_len;

            let data_len = decode_vbyte(data, &mut pos)? as usize;
            if pos + data_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Truncated column data",
                ));
            }
            let column_data = &data[pos..pos + data_len];
            pos += data_len;

            match col_type {
                x if x == ColumnType::Numeric as u8 => {
                    let column = NumericColumn::deserialize(column_data)?;
                    reader.add_numeric(name, column);
                }
                x if x == ColumnType::Boolean as u8 => {
                    let column = BooleanColumn::deserialize(column_data)?;
                    reader.add_boolean(name, column);
                }
                x if x == ColumnType::Keyword as u8 => {
                    let column = KeywordColumn::deserialize(column_data)?;
                    reader.add_keyword(name, column);
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unknown column type",
                    ))
                }
            }
        }

        Ok(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numeric_column() {
        let mut col = NumericColumn::new();
        col.add(Some(100));
        col.add(Some(200));
        col.add(None);
        col.add(Some(150));

        assert_eq!(col.get(DocNo(0)), Some(100));
        assert_eq!(col.get(DocNo(1)), Some(200));
        assert_eq!(col.get(DocNo(2)), None);
        assert_eq!(col.get(DocNo(3)), Some(150));

        assert!(!col.is_null(DocNo(0)));
        assert!(col.is_null(DocNo(2)));

        // Test range query
        let result = col.range_query(100, 160);
        assert!(result.contains(0));
        assert!(!result.contains(1));
        assert!(!result.contains(2));
        assert!(result.contains(3));
    }

    #[test]
    fn test_numeric_column_serialization() {
        let mut col = NumericColumn::new();
        col.add(Some(100));
        col.add(Some(200));
        col.add(None);
        col.add(Some(150));

        let data = col.serialize();
        let restored = NumericColumn::deserialize(&data).unwrap();

        assert_eq!(restored.get(DocNo(0)), Some(100));
        assert_eq!(restored.get(DocNo(1)), Some(200));
        assert_eq!(restored.get(DocNo(2)), None);
        assert_eq!(restored.get(DocNo(3)), Some(150));
    }

    #[test]
    fn test_boolean_column() {
        let mut col = BooleanColumn::new();
        col.add(Some(true));
        col.add(Some(false));
        col.add(None);
        col.add(Some(true));

        assert_eq!(col.get(DocNo(0)), Some(true));
        assert_eq!(col.get(DocNo(1)), Some(false));
        assert_eq!(col.get(DocNo(2)), None);
        assert_eq!(col.get(DocNo(3)), Some(true));

        assert!(col.true_docs().contains(0));
        assert!(!col.true_docs().contains(1));
        assert!(col.true_docs().contains(3));
    }

    #[test]
    fn test_boolean_column_serialization() {
        let mut col = BooleanColumn::new();
        col.add(Some(true));
        col.add(Some(false));
        col.add(None);
        col.add(Some(true));

        let data = col.serialize();
        let restored = BooleanColumn::deserialize(&data).unwrap();

        assert_eq!(restored.get(DocNo(0)), Some(true));
        assert_eq!(restored.get(DocNo(1)), Some(false));
        assert_eq!(restored.get(DocNo(2)), None);
        assert_eq!(restored.get(DocNo(3)), Some(true));
    }

    #[test]
    fn test_keyword_column() {
        let mut col = KeywordColumn::new();
        col.add(Some("apple"));
        col.add(Some("banana"));
        col.add(None);
        col.add(Some("apple")); // Reuse "apple"

        assert_eq!(col.get(DocNo(0)), Some("apple"));
        assert_eq!(col.get(DocNo(1)), Some("banana"));
        assert_eq!(col.get(DocNo(2)), None);
        assert_eq!(col.get(DocNo(3)), Some("apple"));

        // Test equals query
        let result = col.equals_query("apple");
        assert!(result.contains(0));
        assert!(!result.contains(1));
        assert!(!result.contains(2));
        assert!(result.contains(3));

        // Check dictionary size
        assert_eq!(col.unique_keywords().len(), 2);
    }

    #[test]
    fn test_keyword_column_serialization() {
        let mut col = KeywordColumn::new();
        col.add(Some("apple"));
        col.add(Some("banana"));
        col.add(None);
        col.add(Some("apple"));

        let data = col.serialize();
        let restored = KeywordColumn::deserialize(&data).unwrap();

        assert_eq!(restored.get(DocNo(0)), Some("apple"));
        assert_eq!(restored.get(DocNo(1)), Some("banana"));
        assert_eq!(restored.get(DocNo(2)), None);
        assert_eq!(restored.get(DocNo(3)), Some("apple"));
    }

    #[test]
    fn test_docvalues_reader_from_rows_and_serialization() {
        use std::collections::HashMap;

        let mut rows: HashMap<DocNo, DocValueRow> = HashMap::new();

        let mut row0 = DocValueRow::new();
        row0.numerics = vec![Some(10), None];
        row0.booleans = vec![Some(true)];
        row0.keywords = vec![Some("alpha".to_string()), None];
        rows.insert(DocNo(0), row0);

        let mut row1 = DocValueRow::new();
        row1.numerics = vec![Some(20)];
        row1.booleans = vec![None];
        row1.keywords = vec![Some("beta".to_string()), Some("gamma".to_string())];
        rows.insert(DocNo(1), row1);

        let reader = DocValuesReader::from_rows(&rows, 2);

        let num0 = reader.get_numeric("numeric_0").unwrap();
        let num1 = reader.get_numeric("numeric_1").unwrap();
        assert_eq!(num0.get(DocNo(0)), Some(10));
        assert_eq!(num0.get(DocNo(1)), Some(20));
        assert_eq!(num1.get(DocNo(0)), None);
        assert_eq!(num1.get(DocNo(1)), None);

        let bool0 = reader.get_boolean("boolean_0").unwrap();
        assert_eq!(bool0.get(DocNo(0)), Some(true));
        assert_eq!(bool0.get(DocNo(1)), None);

        let kw0 = reader.get_keyword("keyword_0").unwrap();
        let kw1 = reader.get_keyword("keyword_1").unwrap();
        assert_eq!(kw0.get(DocNo(0)), Some("alpha"));
        assert_eq!(kw0.get(DocNo(1)), Some("beta"));
        assert_eq!(kw1.get(DocNo(0)), None);
        assert_eq!(kw1.get(DocNo(1)), Some("gamma"));

        let data = reader.serialize();
        let restored = DocValuesReader::deserialize(&data).unwrap();
        assert_eq!(
            restored.get_numeric("numeric_0").unwrap().get(DocNo(0)),
            Some(10)
        );
        assert_eq!(
            restored.get_keyword("keyword_1").unwrap().get(DocNo(1)),
            Some("gamma")
        );
    }
}
