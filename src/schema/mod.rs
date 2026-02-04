//! Schema and field type system
//!
//! This module defines the schema system for Squidex, including:
//! - Field types (Text, Keyword, Long, Double, Boolean, Date, Vector)
//! - Index mappings (field configuration)
//! - Dynamic mapping behavior

mod field_type;
mod mapping;

pub use field_type::FieldType;
pub use mapping::{DynamicMapping, FieldMapping, IndexMapping};
