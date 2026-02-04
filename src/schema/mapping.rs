//! Index mapping definitions
//!
//! Mappings define the schema for an index, including field types and indexing options.

use super::field_type::FieldType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dynamic mapping behavior for unmapped fields
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DynamicMapping {
    /// Automatically detect and map new fields (default)
    #[default]
    True,
    /// Ignore unmapped fields (don't index them)
    False,
    /// Reject documents with unmapped fields
    Strict,
    /// Inherit from parent mapping
    Runtime,
}

impl DynamicMapping {
    /// Check if new fields should be automatically mapped
    pub fn should_auto_map(&self) -> bool {
        matches!(self, DynamicMapping::True)
    }

    /// Check if unmapped fields should cause an error
    pub fn should_reject_unmapped(&self) -> bool {
        matches!(self, DynamicMapping::Strict)
    }
}

/// Field mapping configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMapping {
    /// Field data type
    pub field_type: FieldType,

    /// Whether to index this field (default: true)
    #[serde(default = "default_true")]
    pub index: bool,

    /// Whether to store the original value (default: false)
    #[serde(default)]
    pub store: bool,

    /// Whether to include in _all field (default: true for text)
    #[serde(default = "default_true")]
    pub include_in_all: bool,

    /// Copy field value to another field
    #[serde(default)]
    pub copy_to: Vec<String>,

    /// Null value replacement (stored as JSON string for bincode compatibility)
    #[serde(default)]
    pub null_value: Option<String>,

    /// Nested field mappings (for object types)
    #[serde(default)]
    pub properties: Option<HashMap<String, FieldMapping>>,
}

fn default_true() -> bool {
    true
}

impl Default for FieldMapping {
    fn default() -> Self {
        Self {
            field_type: FieldType::default(),
            index: true,
            store: false,
            include_in_all: true,
            copy_to: Vec::new(),
            null_value: None,
            properties: None,
        }
    }
}

impl FieldMapping {
    /// Create a new field mapping with the given type
    pub fn new(field_type: FieldType) -> Self {
        Self {
            field_type,
            ..Default::default()
        }
    }

    /// Create a text field mapping
    pub fn text() -> Self {
        Self::new(FieldType::text())
    }

    /// Create a keyword field mapping
    pub fn keyword() -> Self {
        Self::new(FieldType::keyword())
    }

    /// Create a long field mapping
    pub fn long() -> Self {
        Self::new(FieldType::Long)
    }

    /// Create a double field mapping
    pub fn double() -> Self {
        Self::new(FieldType::Double)
    }

    /// Create a boolean field mapping
    pub fn boolean() -> Self {
        Self::new(FieldType::Boolean)
    }

    /// Create a date field mapping
    pub fn date() -> Self {
        Self::new(FieldType::date())
    }

    /// Create a vector field mapping
    pub fn vector(dimensions: usize) -> Self {
        Self::new(FieldType::vector(dimensions))
    }

    /// Set whether the field should be indexed
    pub fn with_index(mut self, index: bool) -> Self {
        self.index = index;
        self
    }

    /// Set whether the field value should be stored
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Add a copy_to target
    pub fn copy_to(mut self, field: impl Into<String>) -> Self {
        self.copy_to.push(field.into());
        self
    }

    /// Set the null value replacement
    /// Set the null value replacement (as a JSON string)
    pub fn with_null_value(mut self, value: impl Into<String>) -> Self {
        self.null_value = Some(value.into());
        self
    }

    /// Set nested field properties (for object types)
    pub fn with_properties(mut self, properties: HashMap<String, FieldMapping>) -> Self {
        self.properties = Some(properties);
        self
    }
}

/// Index mapping (schema) definition
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexMapping {
    /// Field mappings
    #[serde(default)]
    pub properties: HashMap<String, FieldMapping>,

    /// Dynamic mapping behavior
    #[serde(default)]
    pub dynamic: DynamicMapping,

    /// Date detection patterns
    #[serde(default)]
    pub date_detection: bool,

    /// Numeric detection (auto-detect numbers in strings)
    #[serde(default)]
    pub numeric_detection: bool,

    /// Dynamic templates for auto-mapping
    #[serde(default)]
    pub dynamic_templates: Vec<DynamicTemplate>,
}

impl IndexMapping {
    /// Create a new empty mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mapping with dynamic enabled
    pub fn dynamic() -> Self {
        Self {
            dynamic: DynamicMapping::True,
            ..Default::default()
        }
    }

    /// Create a mapping with strict mode
    pub fn strict() -> Self {
        Self {
            dynamic: DynamicMapping::Strict,
            ..Default::default()
        }
    }

    /// Add a field mapping
    pub fn field(mut self, name: impl Into<String>, mapping: FieldMapping) -> Self {
        self.properties.insert(name.into(), mapping);
        self
    }

    /// Set dynamic mapping behavior
    pub fn with_dynamic(mut self, dynamic: DynamicMapping) -> Self {
        self.dynamic = dynamic;
        self
    }

    /// Enable date detection
    pub fn with_date_detection(mut self, enabled: bool) -> Self {
        self.date_detection = enabled;
        self
    }

    /// Enable numeric detection
    pub fn with_numeric_detection(mut self, enabled: bool) -> Self {
        self.numeric_detection = enabled;
        self
    }

    /// Get a field mapping by path (supports dot notation)
    pub fn get_field(&self, path: &str) -> Option<&FieldMapping> {
        let parts: Vec<&str> = path.split('.').collect();
        self.get_field_parts(&parts)
    }

    fn get_field_parts(&self, parts: &[&str]) -> Option<&FieldMapping> {
        if parts.is_empty() {
            return None;
        }

        let field = self.properties.get(parts[0])?;

        if parts.len() == 1 {
            Some(field)
        } else if let Some(ref props) = field.properties {
            // Recursively look up nested fields
            Self::get_nested_field_parts(props, &parts[1..])
        } else {
            None
        }
    }

    fn get_nested_field_parts<'a>(
        props: &'a HashMap<String, FieldMapping>,
        parts: &[&str],
    ) -> Option<&'a FieldMapping> {
        if parts.is_empty() {
            return None;
        }

        let field = props.get(parts[0])?;

        if parts.len() == 1 {
            Some(field)
        } else if let Some(ref nested_props) = field.properties {
            Self::get_nested_field_parts(nested_props, &parts[1..])
        } else {
            None
        }
    }

    /// Check if a field exists
    pub fn has_field(&self, path: &str) -> bool {
        self.get_field(path).is_some()
    }

    /// Get all field names (flattened with dot notation)
    pub fn field_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_field_names(&self.properties, "", &mut names);
        names
    }

    fn collect_field_names(
        &self,
        props: &HashMap<String, FieldMapping>,
        prefix: &str,
        names: &mut Vec<String>,
    ) {
        for (name, mapping) in props {
            let full_name = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}.{}", prefix, name)
            };

            names.push(full_name.clone());

            if let Some(ref nested_props) = mapping.properties {
                self.collect_field_names(nested_props, &full_name, names);
            }
        }
    }

    /// Auto-detect field type from a JSON value
    pub fn detect_field_type(value: &serde_json::Value) -> FieldType {
        match value {
            serde_json::Value::Null => FieldType::keyword(),
            serde_json::Value::Bool(_) => FieldType::Boolean,
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    FieldType::Long
                } else {
                    FieldType::Double
                }
            }
            serde_json::Value::String(_) => FieldType::text(),
            serde_json::Value::Array(arr) => {
                // Check if it looks like a vector
                if !arr.is_empty() && arr.iter().all(|v| v.is_number()) {
                    FieldType::vector(arr.len())
                } else {
                    // Default to text for other arrays
                    FieldType::text()
                }
            }
            serde_json::Value::Object(_) => FieldType::Object { dynamic: true },
        }
    }
}

/// Dynamic template for automatic field mapping
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DynamicTemplate {
    /// Template name
    pub name: String,

    /// Match condition
    pub match_condition: MatchCondition,

    /// Mapping to apply
    pub mapping: FieldMapping,
}

/// Match condition for dynamic templates
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MatchCondition {
    /// Match field names (glob pattern)
    #[serde(default)]
    pub match_pattern: Option<String>,

    /// Unmatch field names (glob pattern)
    #[serde(default)]
    pub unmatch: Option<String>,

    /// Match field path (full path pattern)
    #[serde(default)]
    pub path_match: Option<String>,

    /// Match mapping type
    #[serde(default)]
    pub match_mapping_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_field_mapping_creation() {
        let mapping = FieldMapping::text()
            .with_store(true)
            .copy_to("all_content");

        assert!(mapping.store);
        assert_eq!(mapping.copy_to, vec!["all_content"]);
    }

    #[test]
    fn test_index_mapping_builder() {
        let mapping = IndexMapping::new()
            .field("title", FieldMapping::text())
            .field("status", FieldMapping::keyword())
            .field("created_at", FieldMapping::date())
            .with_dynamic(DynamicMapping::Strict);

        assert_eq!(mapping.properties.len(), 3);
        assert!(mapping.has_field("title"));
        assert!(mapping.has_field("status"));
        assert!(!mapping.has_field("unknown"));
    }

    #[test]
    fn test_nested_field_lookup() {
        let mut author_props = HashMap::new();
        author_props.insert("name".to_string(), FieldMapping::text());
        author_props.insert("email".to_string(), FieldMapping::keyword());

        let mapping = IndexMapping::new()
            .field("title", FieldMapping::text())
            .field(
                "author",
                FieldMapping::new(FieldType::Object { dynamic: false })
                    .with_properties(author_props),
            );

        assert!(mapping.has_field("author"));
        assert!(mapping.has_field("author.name"));
        assert!(mapping.has_field("author.email"));
        assert!(!mapping.has_field("author.unknown"));
    }

    #[test]
    fn test_field_names() {
        let mut author_props = HashMap::new();
        author_props.insert("name".to_string(), FieldMapping::text());

        let mapping = IndexMapping::new()
            .field("title", FieldMapping::text())
            .field(
                "author",
                FieldMapping::new(FieldType::Object { dynamic: false })
                    .with_properties(author_props),
            );

        let names = mapping.field_names();
        assert!(names.contains(&"title".to_string()));
        assert!(names.contains(&"author".to_string()));
        assert!(names.contains(&"author.name".to_string()));
    }

    #[test]
    fn test_detect_field_type() {
        assert!(matches!(
            IndexMapping::detect_field_type(&json!("hello")),
            FieldType::Text { .. }
        ));
        assert!(matches!(
            IndexMapping::detect_field_type(&json!(123)),
            FieldType::Long
        ));
        assert!(matches!(
            IndexMapping::detect_field_type(&json!(3.14)),
            FieldType::Double
        ));
        assert!(matches!(
            IndexMapping::detect_field_type(&json!(true)),
            FieldType::Boolean
        ));
        assert!(matches!(
            IndexMapping::detect_field_type(&json!([1.0, 2.0, 3.0])),
            FieldType::Vector { dimensions: 3, .. }
        ));
    }

    #[test]
    fn test_dynamic_mapping() {
        assert!(DynamicMapping::True.should_auto_map());
        assert!(!DynamicMapping::False.should_auto_map());
        assert!(!DynamicMapping::Strict.should_auto_map());

        assert!(DynamicMapping::Strict.should_reject_unmapped());
        assert!(!DynamicMapping::True.should_reject_unmapped());
    }

    #[test]
    fn test_serialization() {
        let mapping = IndexMapping::new()
            .field("title", FieldMapping::text())
            .field("count", FieldMapping::long())
            .with_dynamic(DynamicMapping::Strict);

        let json = serde_json::to_string_pretty(&mapping).unwrap();
        let deserialized: IndexMapping = serde_json::from_str(&json).unwrap();

        assert_eq!(mapping.dynamic, deserialized.dynamic);
        assert_eq!(mapping.properties.len(), deserialized.properties.len());
    }
}
