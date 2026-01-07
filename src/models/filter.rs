use serde::{Deserialize, Serialize};

/// Filter for metadata-based search refinement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Filter {
    /// Filter by tag
    Tag(String),

    /// Filter by source
    Source(String),

    /// Filter by date range (Unix timestamps in seconds)
    DateRange { start: u64, end: u64 },

    /// Filter by custom metadata field
    Custom { key: String, value: String },
}

impl Filter {
    /// Create a tag filter
    pub fn tag(tag: impl Into<String>) -> Self {
        Filter::Tag(tag.into())
    }

    /// Create a source filter
    pub fn source(source: impl Into<String>) -> Self {
        Filter::Source(source.into())
    }

    /// Create a date range filter
    pub fn date_range(start: u64, end: u64) -> Self {
        Filter::DateRange { start, end }
    }

    /// Create a custom metadata filter
    pub fn custom(key: impl Into<String>, value: impl Into<String>) -> Self {
        Filter::Custom {
            key: key.into(),
            value: value.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_constructors() {
        let tag_filter = Filter::tag("rust");
        assert!(matches!(tag_filter, Filter::Tag(_)));

        let source_filter = Filter::source("github");
        assert!(matches!(source_filter, Filter::Source(_)));

        let date_filter = Filter::date_range(1000, 2000);
        assert!(matches!(date_filter, Filter::DateRange { .. }));

        let custom_filter = Filter::custom("author", "john");
        assert!(matches!(custom_filter, Filter::Custom { .. }));
    }
}
