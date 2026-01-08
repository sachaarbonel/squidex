//! Term dictionary using FST (Finite State Transducer)
//!
//! Use an FST term dictionary with per-term offsets into postings.
//! FST provides O(|key|) lookups and efficient prefix/range queries.

use std::io::{self, Write};

use fst::{Map, MapBuilder, Streamer};

use super::types::PostingListMeta;

/// Term dictionary backed by FST
///
/// Maps terms to postings metadata (offset, length, doc frequency).
/// The FST stores a u64 value which indexes into a metadata array.
pub struct TermDictionary {
    /// FST mapping term -> index in metadata array
    fst: Map<Vec<u8>>,
    /// Metadata for each term (parallel to FST output values)
    metadata: Vec<PostingListMeta>,
}

impl TermDictionary {
    /// Create a term dictionary from FST data and metadata
    pub fn new(fst_data: Vec<u8>, metadata: Vec<PostingListMeta>) -> io::Result<Self> {
        let fst = Map::new(fst_data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { fst, metadata })
    }

    /// Look up a term and return its postings metadata
    pub fn get(&self, term: &str) -> Option<&PostingListMeta> {
        self.fst
            .get(term.as_bytes())
            .map(|idx| &self.metadata[idx as usize])
    }

    /// Check if a term exists
    pub fn contains(&self, term: &str) -> bool {
        self.fst.contains_key(term.as_bytes())
    }

    /// Get the number of terms
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Iterate over all terms with prefix
    pub fn prefix_search<'a>(&'a self, prefix: &str) -> Vec<(String, &'a PostingListMeta)> {
        let prefix_bytes = prefix.as_bytes();
        let mut results = Vec::new();

        // Use FST stream with range query
        let mut stream = self.fst.stream();
        while let Some((key, idx)) = stream.next() {
            if key.starts_with(prefix_bytes) {
                if let Ok(term) = std::str::from_utf8(key) {
                    results.push((term.to_string(), &self.metadata[idx as usize]));
                }
            } else if key > prefix_bytes && !key.starts_with(prefix_bytes) {
                // We've passed the prefix range, can stop
                if results.is_empty() || !results.last().map(|(t, _)| t.starts_with(prefix)).unwrap_or(false) {
                    continue; // Haven't found matching terms yet
                }
                break;
            }
        }

        results
    }

    /// Get the raw FST data (for serialization)
    pub fn fst_bytes(&self) -> &[u8] {
        self.fst.as_fst().as_bytes()
    }

    /// Get the metadata array (for serialization)
    pub fn metadata(&self) -> &[PostingListMeta] {
        &self.metadata
    }

    /// Iterate over all terms in the dictionary
    pub fn iter_terms(&self) -> Vec<(String, &PostingListMeta)> {
        let mut results = Vec::new();
        let mut stream = self.fst.stream();
        while let Some((key, idx)) = stream.next() {
            if let Ok(term) = std::str::from_utf8(key) {
                results.push((term.to_string(), &self.metadata[idx as usize]));
            }
        }
        results
    }
}

/// Builder for term dictionaries
pub struct TermDictionaryBuilder {
    /// Terms and their metadata, must be added in sorted order
    terms: Vec<(String, PostingListMeta)>,
}

impl TermDictionaryBuilder {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            terms: Vec::with_capacity(capacity),
        }
    }

    /// Add a term with its postings metadata
    /// Terms MUST be added in lexicographic order!
    pub fn add(&mut self, term: String, meta: PostingListMeta) {
        self.terms.push((term, meta));
    }

    /// Build the term dictionary
    pub fn build(mut self) -> io::Result<TermDictionary> {
        // Sort terms (FST requires sorted input)
        self.terms.sort_by(|a, b| a.0.cmp(&b.0));

        // Build FST
        let mut fst_builder = MapBuilder::memory();
        let mut metadata = Vec::with_capacity(self.terms.len());

        for (idx, (term, meta)) in self.terms.into_iter().enumerate() {
            fst_builder
                .insert(term.as_bytes(), idx as u64)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            metadata.push(meta);
        }

        let fst_data = fst_builder
            .into_inner()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        TermDictionary::new(fst_data, metadata)
    }
}

impl Default for TermDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory term dictionary for mutable buffers
/// Uses a simple HashMap for fast lookups during indexing
#[derive(Debug, Clone)]
pub struct MutableTermDict {
    terms: std::collections::HashMap<String, PostingListMeta>,
}

impl MutableTermDict {
    pub fn new() -> Self {
        Self {
            terms: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, term: String, meta: PostingListMeta) {
        self.terms.insert(term, meta);
    }

    pub fn get(&self, term: &str) -> Option<&PostingListMeta> {
        self.terms.get(term)
    }

    pub fn contains(&self, term: &str) -> bool {
        self.terms.contains_key(term)
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &PostingListMeta)> {
        self.terms.iter()
    }

    /// Convert to an immutable term dictionary
    pub fn freeze(self) -> io::Result<TermDictionary> {
        let mut builder = TermDictionaryBuilder::with_capacity(self.terms.len());
        for (term, meta) in self.terms {
            builder.add(term, meta);
        }
        builder.build()
    }
}

impl Default for MutableTermDict {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_dictionary_builder() {
        let mut builder = TermDictionaryBuilder::new();

        builder.add(
            "apple".to_string(),
            PostingListMeta {
                offset: 0,
                length: 100,
                doc_frequency: 10,
                total_term_frequency: 50,
            },
        );

        builder.add(
            "banana".to_string(),
            PostingListMeta {
                offset: 100,
                length: 200,
                doc_frequency: 20,
                total_term_frequency: 100,
            },
        );

        builder.add(
            "cherry".to_string(),
            PostingListMeta {
                offset: 300,
                length: 150,
                doc_frequency: 15,
                total_term_frequency: 75,
            },
        );

        let dict = builder.build().unwrap();

        assert_eq!(dict.len(), 3);
        assert!(dict.contains("apple"));
        assert!(dict.contains("banana"));
        assert!(dict.contains("cherry"));
        assert!(!dict.contains("date"));

        let apple_meta = dict.get("apple").unwrap();
        assert_eq!(apple_meta.offset, 0);
        assert_eq!(apple_meta.doc_frequency, 10);
    }

    #[test]
    fn test_prefix_search() {
        let mut builder = TermDictionaryBuilder::new();

        builder.add(
            "test".to_string(),
            PostingListMeta {
                offset: 0,
                length: 100,
                doc_frequency: 10,
                total_term_frequency: 50,
            },
        );

        builder.add(
            "testing".to_string(),
            PostingListMeta {
                offset: 100,
                length: 100,
                doc_frequency: 5,
                total_term_frequency: 25,
            },
        );

        builder.add(
            "tester".to_string(),
            PostingListMeta {
                offset: 200,
                length: 100,
                doc_frequency: 3,
                total_term_frequency: 15,
            },
        );

        builder.add(
            "other".to_string(),
            PostingListMeta {
                offset: 300,
                length: 100,
                doc_frequency: 2,
                total_term_frequency: 10,
            },
        );

        let dict = builder.build().unwrap();

        let results = dict.prefix_search("test");
        assert_eq!(results.len(), 3);
        assert!(results.iter().any(|(t, _)| t == "test"));
        assert!(results.iter().any(|(t, _)| t == "testing"));
        assert!(results.iter().any(|(t, _)| t == "tester"));
    }

    #[test]
    fn test_mutable_term_dict() {
        let mut dict = MutableTermDict::new();

        dict.insert(
            "hello".to_string(),
            PostingListMeta {
                offset: 0,
                length: 100,
                doc_frequency: 5,
                total_term_frequency: 20,
            },
        );

        dict.insert(
            "world".to_string(),
            PostingListMeta {
                offset: 100,
                length: 150,
                doc_frequency: 8,
                total_term_frequency: 40,
            },
        );

        assert_eq!(dict.len(), 2);
        assert!(dict.contains("hello"));
        assert!(dict.contains("world"));
        assert!(!dict.contains("foo"));

        let frozen = dict.freeze().unwrap();
        assert_eq!(frozen.len(), 2);
        assert!(frozen.contains("hello"));
        assert!(frozen.contains("world"));
    }
}
