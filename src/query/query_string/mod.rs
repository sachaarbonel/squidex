//! Lucene-style query string parser
//!
//! Supports syntax like:
//! - `title:rust AND tags:tutorial`
//! - `status:published AND created_at:[2024-01-01 TO 2024-12-31]`
//! - `content:"exact phrase"~2`
//! - `title:prog*`
//! - `author:john~`
//!
//! # Grammar
//!
//! ```text
//! query       := or_expr
//! or_expr     := and_expr (OR and_expr)*
//! and_expr    := not_expr (AND not_expr)*
//! not_expr    := NOT? primary
//! primary     := field_query | grouped | term
//! field_query := TERM COLON (range | wildcard | fuzzy | phrase | term)
//! range       := '[' value TO value ']' | '{' value TO value '}'
//! wildcard    := TERM with * or ?
//! fuzzy       := TERM TILDE distance?
//! phrase      := QUOTED TILDE slop?
//! grouped     := '(' or_expr ')'
//! ```
//!
//! # Example
//!
//! ```rust
//! use squidex::query::query_string::QueryStringParser;
//!
//! let mut parser = QueryStringParser::new("title:rust AND status:published").unwrap();
//! let query = parser.parse().unwrap();
//! ```

pub mod lexer;
pub mod parser;

pub use lexer::{Lexer, Token};
pub use parser::QueryStringParser;
