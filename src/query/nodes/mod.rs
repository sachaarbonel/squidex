//! Concrete query node implementations
//!
//! This module provides implementations of the `QueryNode` trait for
//! various query types.

mod all_docs;
mod bool_query;
mod fuzzy_query;
mod match_query;
mod phrase_query;
mod prefix_query;
mod range_query;
mod term_query;
mod terms_query;
mod wildcard_query;

pub use all_docs::AllDocsQuery;
pub use bool_query::BoolQuery;
pub use fuzzy_query::{damerau_levenshtein_distance, levenshtein_distance, FuzzyQuery};
pub use match_query::MatchQuery;
pub use phrase_query::PhraseQuery;
pub use prefix_query::PrefixQuery;
pub use range_query::RangeQuery;
pub use term_query::TermQuery;
pub use terms_query::TermsQuery;
pub use wildcard_query::WildcardQuery;
