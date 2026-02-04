//! Field type definitions
//!
//! Defines how different data types are indexed and queried.

use serde::{Deserialize, Serialize};

/// Field data type
///
/// Determines how a field is indexed, stored, and queried.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    /// Full-text searchable field
    ///
    /// Text fields are analyzed (tokenized, lowercased, stemmed) before indexing.
    /// They support full-text queries like `match` and `match_phrase`.
    Text {
        /// Analyzer to use for indexing
        #[serde(default = "default_analyzer")]
        analyzer: String,
        /// Analyzer to use for search (if different from index analyzer)
        #[serde(default)]
        search_analyzer: Option<String>,
        /// Store term positions for phrase queries
        #[serde(default = "default_true")]
        index_positions: bool,
    },

    /// Exact match keyword field
    ///
    /// Keyword fields are not analyzed - the entire value is indexed as a single term.
    /// They support exact match queries, prefix queries, and aggregations.
    Keyword {
        /// Ignore strings longer than this
        #[serde(default = "default_ignore_above")]
        ignore_above: usize,
        /// Normalize before indexing (e.g., lowercase)
        #[serde(default)]
        normalizer: Option<String>,
    },

    /// 64-bit signed integer
    ///
    /// Supports range queries, sorting, and aggregations.
    Long,

    /// 64-bit floating point
    ///
    /// Supports range queries, sorting, and aggregations.
    Double,

    /// Boolean value
    ///
    /// Supports term queries and aggregations.
    Boolean,

    /// Date/time field
    ///
    /// Stored as Unix timestamp (milliseconds). Supports range queries and date math.
    Date {
        /// Date format for parsing string values
        #[serde(default = "default_date_format")]
        format: String,
    },

    /// Dense vector field for similarity search
    ///
    /// Supports kNN queries.
    Vector {
        /// Number of dimensions
        dimensions: usize,
        /// Similarity metric
        #[serde(default)]
        similarity: VectorSimilarity,
    },

    /// Nested object with its own field mappings
    Object {
        /// Whether to enable dynamic mapping for this object
        #[serde(default)]
        dynamic: bool,
    },
}

fn default_analyzer() -> String {
    "standard".to_string()
}

fn default_true() -> bool {
    true
}

fn default_ignore_above() -> usize {
    256
}

fn default_date_format() -> String {
    "strict_date_optional_time||epoch_millis".to_string()
}

/// Vector similarity metric
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorSimilarity {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

impl Default for FieldType {
    fn default() -> Self {
        FieldType::Text {
            analyzer: default_analyzer(),
            search_analyzer: None,
            index_positions: true,
        }
    }
}

impl FieldType {
    /// Create a text field with default settings
    pub fn text() -> Self {
        FieldType::Text {
            analyzer: default_analyzer(),
            search_analyzer: None,
            index_positions: true,
        }
    }

    /// Create a text field with a specific analyzer
    pub fn text_with_analyzer(analyzer: impl Into<String>) -> Self {
        FieldType::Text {
            analyzer: analyzer.into(),
            search_analyzer: None,
            index_positions: true,
        }
    }

    /// Create a keyword field with default settings
    pub fn keyword() -> Self {
        FieldType::Keyword {
            ignore_above: default_ignore_above(),
            normalizer: None,
        }
    }

    /// Create a date field with default format
    pub fn date() -> Self {
        FieldType::Date {
            format: default_date_format(),
        }
    }

    /// Create a vector field
    pub fn vector(dimensions: usize) -> Self {
        FieldType::Vector {
            dimensions,
            similarity: VectorSimilarity::default(),
        }
    }

    /// Check if this field type supports full-text queries
    pub fn supports_fulltext(&self) -> bool {
        matches!(self, FieldType::Text { .. })
    }

    /// Check if this field type supports exact match queries
    pub fn supports_exact_match(&self) -> bool {
        matches!(
            self,
            FieldType::Keyword { .. }
                | FieldType::Long
                | FieldType::Double
                | FieldType::Boolean
                | FieldType::Date { .. }
        )
    }

    /// Check if this field type supports range queries
    pub fn supports_range(&self) -> bool {
        matches!(
            self,
            FieldType::Long | FieldType::Double | FieldType::Date { .. }
        )
    }

    /// Check if this field type supports aggregations
    pub fn supports_aggregation(&self) -> bool {
        !matches!(self, FieldType::Text { .. } | FieldType::Vector { .. })
    }

    /// Check if this field type supports sorting
    pub fn supports_sorting(&self) -> bool {
        matches!(
            self,
            FieldType::Keyword { .. }
                | FieldType::Long
                | FieldType::Double
                | FieldType::Boolean
                | FieldType::Date { .. }
        )
    }

    /// Check if this field type supports vector similarity search
    pub fn supports_vector_search(&self) -> bool {
        matches!(self, FieldType::Vector { .. })
    }

    /// Get the internal storage type name
    pub fn storage_type(&self) -> &'static str {
        match self {
            FieldType::Text { .. } => "text",
            FieldType::Keyword { .. } => "keyword",
            FieldType::Long => "long",
            FieldType::Double => "double",
            FieldType::Boolean => "boolean",
            FieldType::Date { .. } => "date",
            FieldType::Vector { .. } => "vector",
            FieldType::Object { .. } => "object",
        }
    }

    /// Validate a value against this field type
    pub fn validate(&self, value: &serde_json::Value) -> Result<(), String> {
        match self {
            FieldType::Text { .. } => {
                if !value.is_string() {
                    return Err("Text field requires a string value".to_string());
                }
            }
            FieldType::Keyword { ignore_above, .. } => {
                if let Some(s) = value.as_str() {
                    if s.len() > *ignore_above {
                        return Err(format!(
                            "Keyword exceeds ignore_above limit of {}",
                            ignore_above
                        ));
                    }
                } else if !value.is_string() {
                    return Err("Keyword field requires a string value".to_string());
                }
            }
            FieldType::Long => {
                if !value.is_i64() && !value.is_u64() {
                    return Err("Long field requires an integer value".to_string());
                }
            }
            FieldType::Double => {
                if !value.is_f64() && !value.is_i64() {
                    return Err("Double field requires a numeric value".to_string());
                }
            }
            FieldType::Boolean => {
                if !value.is_boolean() {
                    return Err("Boolean field requires a boolean value".to_string());
                }
            }
            FieldType::Date { .. } => {
                // Accept string (ISO format) or number (timestamp)
                if !value.is_string() && !value.is_number() {
                    return Err("Date field requires a string or number value".to_string());
                }
            }
            FieldType::Vector { dimensions, .. } => {
                if let Some(arr) = value.as_array() {
                    if arr.len() != *dimensions {
                        return Err(format!(
                            "Vector has {} dimensions, expected {}",
                            arr.len(),
                            dimensions
                        ));
                    }
                    for v in arr {
                        if !v.is_f64() && !v.is_i64() {
                            return Err("Vector elements must be numbers".to_string());
                        }
                    }
                } else {
                    return Err("Vector field requires an array value".to_string());
                }
            }
            FieldType::Object { .. } => {
                if !value.is_object() {
                    return Err("Object field requires an object value".to_string());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_field_type_text() {
        let field = FieldType::text();
        assert!(field.supports_fulltext());
        assert!(!field.supports_exact_match());
        assert!(!field.supports_range());
        assert!(!field.supports_aggregation());
    }

    #[test]
    fn test_field_type_keyword() {
        let field = FieldType::keyword();
        assert!(!field.supports_fulltext());
        assert!(field.supports_exact_match());
        assert!(!field.supports_range());
        assert!(field.supports_aggregation());
        assert!(field.supports_sorting());
    }

    #[test]
    fn test_field_type_long() {
        let field = FieldType::Long;
        assert!(field.supports_exact_match());
        assert!(field.supports_range());
        assert!(field.supports_aggregation());
        assert!(field.supports_sorting());
    }

    #[test]
    fn test_field_type_vector() {
        let field = FieldType::vector(384);
        assert!(field.supports_vector_search());
        assert!(!field.supports_fulltext());
        assert!(!field.supports_aggregation());
    }

    #[test]
    fn test_validate_text() {
        let field = FieldType::text();
        assert!(field.validate(&json!("hello")).is_ok());
        assert!(field.validate(&json!(123)).is_err());
    }

    #[test]
    fn test_validate_long() {
        let field = FieldType::Long;
        assert!(field.validate(&json!(123)).is_ok());
        assert!(field.validate(&json!("hello")).is_err());
    }

    #[test]
    fn test_validate_vector() {
        let field = FieldType::vector(3);
        assert!(field.validate(&json!([1.0, 2.0, 3.0])).is_ok());
        assert!(field.validate(&json!([1.0, 2.0])).is_err()); // Wrong dimensions
        assert!(field.validate(&json!("hello")).is_err());
    }

    #[test]
    fn test_serialization() {
        let field = FieldType::text_with_analyzer("english");
        let json = serde_json::to_string(&field).unwrap();
        // Externally tagged enum format: {"text":{"analyzer":"..."}}
        assert!(json.contains("\"text\""));
        assert!(json.contains("\"analyzer\":\"english\""));

        let deserialized: FieldType = serde_json::from_str(&json).unwrap();
        assert_eq!(field, deserialized);
    }
}
