//! Core types for the query system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Operator for combining terms in a match query
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchOperator {
    /// All terms must match (AND)
    And,
    /// At least one term must match (OR)
    #[default]
    Or,
}

/// Value type for range queries
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RangeValue {
    /// 64-bit integer
    Long(i64),
    /// 64-bit floating point
    Double(f64),
    /// String (for dates, keywords)
    String(String),
}

impl RangeValue {
    /// Convert to i64 if possible
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            RangeValue::Long(v) => Some(*v),
            RangeValue::Double(v) => Some(*v as i64),
            RangeValue::String(s) => s.parse().ok(),
        }
    }

    /// Convert to f64 if possible
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RangeValue::Long(v) => Some(*v as f64),
            RangeValue::Double(v) => Some(*v),
            RangeValue::String(s) => s.parse().ok(),
        }
    }

    /// Try to parse as a Unix timestamp (for date fields)
    pub fn as_timestamp(&self) -> Option<i64> {
        match self {
            RangeValue::Long(v) => Some(*v),
            RangeValue::Double(v) => Some(*v as i64),
            RangeValue::String(s) => {
                // Try parsing as ISO 8601 date
                // For now, just try parsing as integer
                s.parse().ok()
            }
        }
    }
}

/// Range bounds for range queries
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RangeBounds {
    /// Greater than or equal to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gte: Option<RangeValue>,
    /// Greater than
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gt: Option<RangeValue>,
    /// Less than or equal to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lte: Option<RangeValue>,
    /// Less than
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lt: Option<RangeValue>,
    /// Boost factor for scoring
    #[serde(default = "default_boost")]
    pub boost: f32,
}

fn default_boost() -> f32 {
    1.0
}

impl RangeBounds {
    /// Check if a value is within this range
    pub fn contains_i64(&self, value: i64) -> bool {
        if let Some(ref gte) = self.gte {
            if let Some(bound) = gte.as_i64() {
                if value < bound {
                    return false;
                }
            }
        }
        if let Some(ref gt) = self.gt {
            if let Some(bound) = gt.as_i64() {
                if value <= bound {
                    return false;
                }
            }
        }
        if let Some(ref lte) = self.lte {
            if let Some(bound) = lte.as_i64() {
                if value > bound {
                    return false;
                }
            }
        }
        if let Some(ref lt) = self.lt {
            if let Some(bound) = lt.as_i64() {
                if value >= bound {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a float value is within this range
    pub fn contains_f64(&self, value: f64) -> bool {
        if let Some(ref gte) = self.gte {
            if let Some(bound) = gte.as_f64() {
                if value < bound {
                    return false;
                }
            }
        }
        if let Some(ref gt) = self.gt {
            if let Some(bound) = gt.as_f64() {
                if value <= bound {
                    return false;
                }
            }
        }
        if let Some(ref lte) = self.lte {
            if let Some(bound) = lte.as_f64() {
                if value > bound {
                    return false;
                }
            }
        }
        if let Some(ref lt) = self.lt {
            if let Some(bound) = lt.as_f64() {
                if value >= bound {
                    return false;
                }
            }
        }
        true
    }
}

/// Query boost configuration
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryBoost {
    /// Base boost factor
    pub value: f32,
    /// Per-field boost overrides
    #[serde(default)]
    pub fields: HashMap<String, f32>,
}

/// Query execution statistics
#[derive(Clone, Debug, Default)]
pub struct QueryStats {
    /// Number of documents matched
    pub docs_matched: u64,
    /// Number of postings read
    pub postings_read: u64,
    /// Number of filter cache hits
    pub filter_cache_hits: u64,
    /// Number of filter cache misses
    pub filter_cache_misses: u64,
    /// Query execution time in microseconds
    pub execution_time_us: u64,
}

/// Minimum should match configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MinimumShouldMatch {
    /// Exact count
    Count(usize),
    /// Percentage (e.g., "75%")
    Percentage(String),
}

impl MinimumShouldMatch {
    /// Calculate the minimum number of clauses that should match
    pub fn calculate(&self, total_clauses: usize) -> usize {
        match self {
            MinimumShouldMatch::Count(n) => *n,
            MinimumShouldMatch::Percentage(s) => {
                let pct: f64 = s
                    .trim_end_matches('%')
                    .parse()
                    .unwrap_or(100.0)
                    / 100.0;
                ((total_clauses as f64) * pct).ceil() as usize
            }
        }
    }
}

impl Default for MinimumShouldMatch {
    fn default() -> Self {
        MinimumShouldMatch::Count(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_value_conversions() {
        let long = RangeValue::Long(42);
        assert_eq!(long.as_i64(), Some(42));
        assert_eq!(long.as_f64(), Some(42.0));

        let double = RangeValue::Double(3.14);
        assert_eq!(double.as_i64(), Some(3));
        assert_eq!(double.as_f64(), Some(3.14));

        let string = RangeValue::String("100".to_string());
        assert_eq!(string.as_i64(), Some(100));
    }

    #[test]
    fn test_range_bounds() {
        let bounds = RangeBounds {
            gte: Some(RangeValue::Long(10)),
            lt: Some(RangeValue::Long(20)),
            ..Default::default()
        };

        assert!(bounds.contains_i64(10));
        assert!(bounds.contains_i64(15));
        assert!(!bounds.contains_i64(20));
        assert!(!bounds.contains_i64(9));
    }

    #[test]
    fn test_minimum_should_match() {
        assert_eq!(MinimumShouldMatch::Count(2).calculate(5), 2);
        assert_eq!(
            MinimumShouldMatch::Percentage("75%".to_string()).calculate(4),
            3
        );
    }
}
