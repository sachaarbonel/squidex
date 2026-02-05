//! Query DSL parser
//!
//! Parses JSON query DSL into query AST nodes.
//! The syntax is compatible with a subset of Elasticsearch Query DSL.

use crate::error::SquidexError;
use crate::query::ast::QueryNode;
use crate::query::nodes::{
    AllDocsQuery, BoolQuery, FuzzyQuery, MatchQuery, PhraseQuery, PrefixQuery, RangeQuery,
    TermQuery, TermsQuery, WildcardQuery,
};
use crate::query::types::{MatchOperator, MinimumShouldMatch, RangeBounds, RangeValue};
use crate::Result;
use serde_json::{Map, Value};

/// Query parser for JSON DSL
pub struct QueryParser;

impl QueryParser {
    /// Parse a JSON query into an AST node
    ///
    /// # Example
    ///
    /// ```json
    /// {
    ///   "bool": {
    ///     "must": [
    ///       { "match": { "content": "rust" } }
    ///     ],
    ///     "filter": [
    ///       { "range": { "year": { "gte": 2024 } } }
    ///     ]
    ///   }
    /// }
    /// ```
    pub fn parse(json: &Value) -> Result<Box<dyn QueryNode>> {
        match json {
            Value::Object(map) => Self::parse_query_object(map),
            _ => Err(SquidexError::InvalidRequest(
                "Query must be a JSON object".to_string(),
            )),
        }
    }

    /// Parse a JSON string into an AST node
    pub fn parse_str(json_str: &str) -> Result<Box<dyn QueryNode>> {
        let value: Value = serde_json::from_str(json_str).map_err(|e| {
            SquidexError::InvalidRequest(format!("Invalid JSON: {}", e))
        })?;
        Self::parse(&value)
    }

    fn parse_query_object(map: &Map<String, Value>) -> Result<Box<dyn QueryNode>> {
        // Handle wrapped query: { "query": { ... } }
        if let Some(query) = map.get("query") {
            return Self::parse(query);
        }

        // Determine query type
        if let Some(bool_query) = map.get("bool") {
            return Self::parse_bool(bool_query);
        }
        if let Some(match_query) = map.get("match") {
            return Self::parse_match(match_query);
        }
        if let Some(match_all) = map.get("match_all") {
            return Self::parse_match_all(match_all);
        }
        if let Some(term_query) = map.get("term") {
            return Self::parse_term(term_query);
        }
        if let Some(terms_query) = map.get("terms") {
            return Self::parse_terms(terms_query);
        }
        if let Some(range_query) = map.get("range") {
            return Self::parse_range(range_query);
        }
        if let Some(wildcard_query) = map.get("wildcard") {
            return Self::parse_wildcard(wildcard_query);
        }
        if let Some(prefix_query) = map.get("prefix") {
            return Self::parse_prefix(prefix_query);
        }
        if let Some(fuzzy_query) = map.get("fuzzy") {
            return Self::parse_fuzzy(fuzzy_query);
        }
        if let Some(phrase_query) = map.get("match_phrase") {
            return Self::parse_match_phrase(phrase_query);
        }

        Err(SquidexError::InvalidRequest(format!(
            "Unknown query type. Expected one of: bool, match, match_all, match_phrase, term, terms, range, wildcard, prefix, fuzzy. Got keys: {:?}",
            map.keys().collect::<Vec<_>>()
        )))
    }

    /// Parse a bool query
    fn parse_bool(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("bool query must be an object".to_string())
        })?;

        let mut query = BoolQuery::new();

        // Parse must clauses
        if let Some(must) = map.get("must") {
            query.must = Self::parse_clause_array(must)?;
        }

        // Parse should clauses
        if let Some(should) = map.get("should") {
            query.should = Self::parse_clause_array(should)?;
        }

        // Parse must_not clauses
        if let Some(must_not) = map.get("must_not") {
            query.must_not = Self::parse_clause_array(must_not)?;
        }

        // Parse filter clauses
        if let Some(filter) = map.get("filter") {
            query.filter = Self::parse_clause_array(filter)?;
        }

        // Parse minimum_should_match
        if let Some(msm) = map.get("minimum_should_match") {
            query.minimum_should_match = Self::parse_minimum_should_match(msm)?;
        }

        // Parse boost
        if let Some(boost) = map.get("boost") {
            query.boost = boost.as_f64().unwrap_or(1.0) as f32;
        }

        Ok(Box::new(query))
    }

    /// Parse an array of query clauses
    fn parse_clause_array(value: &Value) -> Result<Vec<Box<dyn QueryNode>>> {
        match value {
            Value::Array(arr) => arr.iter().map(Self::parse).collect(),
            // Single clause can be provided without array wrapper
            obj @ Value::Object(_) => Ok(vec![Self::parse(obj)?]),
            _ => Err(SquidexError::InvalidRequest(
                "Clause must be an array or object".to_string(),
            )),
        }
    }

    /// Parse minimum_should_match
    fn parse_minimum_should_match(value: &Value) -> Result<MinimumShouldMatch> {
        match value {
            Value::Number(n) => Ok(MinimumShouldMatch::Count(n.as_u64().unwrap_or(1) as usize)),
            Value::String(s) => {
                if s.ends_with('%') {
                    Ok(MinimumShouldMatch::Percentage(s.clone()))
                } else {
                    let count: usize = s.parse().map_err(|_| {
                        SquidexError::InvalidRequest(format!(
                            "Invalid minimum_should_match: {}",
                            s
                        ))
                    })?;
                    Ok(MinimumShouldMatch::Count(count))
                }
            }
            _ => Err(SquidexError::InvalidRequest(
                "minimum_should_match must be a number or string".to_string(),
            )),
        }
    }

    /// Parse a match query
    fn parse_match(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("match query must be an object".to_string())
        })?;

        // Match query has the form: { "field": "text" } or { "field": { "query": "text", ... } }
        let (field, query_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("match query must specify a field".to_string())
        })?;

        let mut query = match query_spec {
            Value::String(text) => MatchQuery::new(field.clone(), text.clone()),
            Value::Object(spec) => {
                let text = spec
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "match query spec must have 'query' field".to_string(),
                        )
                    })?;

                let mut q = MatchQuery::new(field.clone(), text);

                // Parse operator
                if let Some(op) = spec.get("operator") {
                    let op_str = op.as_str().unwrap_or("or");
                    q.operator = match op_str.to_lowercase().as_str() {
                        "and" => MatchOperator::And,
                        "or" => MatchOperator::Or,
                        _ => MatchOperator::Or,
                    };
                }

                // Parse boost
                if let Some(boost) = spec.get("boost") {
                    q.boost = boost.as_f64().unwrap_or(1.0) as f32;
                }

                // Parse analyzer
                if let Some(analyzer) = spec.get("analyzer") {
                    q.analyzer = analyzer.as_str().map(String::from);
                }

                // Parse minimum_should_match
                if let Some(msm) = spec.get("minimum_should_match") {
                    q.minimum_should_match = msm.as_str().map(String::from);
                }

                q
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "match query value must be a string or object".to_string(),
                ))
            }
        };

        Ok(Box::new(query))
    }

    /// Parse a match_all query
    fn parse_match_all(value: &Value) -> Result<Box<dyn QueryNode>> {
        let boost = value
            .as_object()
            .and_then(|m| m.get("boost"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        Ok(Box::new(AllDocsQuery { boost }))
    }

    /// Parse a term query
    fn parse_term(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("term query must be an object".to_string())
        })?;

        // Term query has the form: { "field": "value" } or { "field": { "value": "...", ... } }
        let (field, term_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("term query must specify a field".to_string())
        })?;

        let (term, boost) = match term_spec {
            Value::String(t) => (t.clone(), 1.0),
            Value::Number(n) => (n.to_string(), 1.0),
            Value::Bool(b) => (b.to_string(), 1.0),
            Value::Object(spec) => {
                let term = spec
                    .get("value")
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        _ => v.to_string(),
                    })
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "term query spec must have 'value' field".to_string(),
                        )
                    })?;

                let boost = spec.get("boost").and_then(|v| v.as_f64()).unwrap_or(1.0);

                (term, boost as f32)
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "term query value must be a string, number, boolean, or object".to_string(),
                ))
            }
        };

        Ok(Box::new(TermQuery::new(field.clone(), term).with_boost(boost)))
    }

    /// Parse a terms query
    fn parse_terms(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("terms query must be an object".to_string())
        })?;

        // Terms query has the form: { "field": ["value1", "value2", ...] }
        let (field, terms_spec) = map
            .iter()
            .find(|(k, _)| *k != "boost")
            .ok_or_else(|| {
                SquidexError::InvalidRequest("terms query must specify a field".to_string())
            })?;

        let terms = match terms_spec {
            Value::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => v.to_string(),
                })
                .collect(),
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "terms query value must be an array".to_string(),
                ))
            }
        };

        let boost = map
            .get("boost")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        Ok(Box::new(TermsQuery::new(field.clone(), terms).with_boost(boost)))
    }

    /// Parse a range query
    fn parse_range(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("range query must be an object".to_string())
        })?;

        // Range query has the form: { "field": { "gte": ..., "lte": ..., ... } }
        let (field, range_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("range query must specify a field".to_string())
        })?;

        let spec = range_spec.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("range query spec must be an object".to_string())
        })?;

        let bounds = RangeBounds {
            gte: spec.get("gte").map(Self::parse_range_value),
            gt: spec.get("gt").map(Self::parse_range_value),
            lte: spec.get("lte").map(Self::parse_range_value),
            lt: spec.get("lt").map(Self::parse_range_value),
            boost: spec.get("boost").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        };

        Ok(Box::new(
            RangeQuery::new(field.clone()).with_bounds(bounds),
        ))
    }

    /// Parse a range value
    fn parse_range_value(value: &Value) -> RangeValue {
        match value {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    RangeValue::Long(i)
                } else if let Some(f) = n.as_f64() {
                    RangeValue::Double(f)
                } else {
                    RangeValue::String(n.to_string())
                }
            }
            Value::String(s) => RangeValue::String(s.clone()),
            _ => RangeValue::String(value.to_string()),
        }
    }

    /// Parse a wildcard query
    ///
    /// Format: { "wildcard": { "field": "pattern*" } }
    /// or: { "wildcard": { "field": { "value": "pattern*", "boost": 1.5 } } }
    fn parse_wildcard(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("wildcard query must be an object".to_string())
        })?;

        let (field, wildcard_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("wildcard query must specify a field".to_string())
        })?;

        let (pattern, boost) = match wildcard_spec {
            Value::String(p) => (p.clone(), 1.0),
            Value::Object(spec) => {
                let pattern = spec
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "wildcard query spec must have 'value' field".to_string(),
                        )
                    })?
                    .to_string();
                let boost = spec.get("boost").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                (pattern, boost)
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "wildcard query value must be a string or object".to_string(),
                ))
            }
        };

        Ok(Box::new(WildcardQuery::new(field.clone(), pattern).with_boost(boost)))
    }

    /// Parse a prefix query
    ///
    /// Format: { "prefix": { "field": "prefix" } }
    /// or: { "prefix": { "field": { "value": "prefix", "boost": 1.5 } } }
    fn parse_prefix(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("prefix query must be an object".to_string())
        })?;

        let (field, prefix_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("prefix query must specify a field".to_string())
        })?;

        let (prefix, boost) = match prefix_spec {
            Value::String(p) => (p.clone(), 1.0),
            Value::Object(spec) => {
                let prefix = spec
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "prefix query spec must have 'value' field".to_string(),
                        )
                    })?
                    .to_string();
                let boost = spec.get("boost").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                (prefix, boost)
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "prefix query value must be a string or object".to_string(),
                ))
            }
        };

        Ok(Box::new(PrefixQuery::new(field.clone(), prefix).with_boost(boost)))
    }

    /// Parse a fuzzy query
    ///
    /// Format: { "fuzzy": { "field": "term" } }
    /// or: { "fuzzy": { "field": { "value": "term", "fuzziness": 2, "prefix_length": 0, "boost": 1.0 } } }
    fn parse_fuzzy(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("fuzzy query must be an object".to_string())
        })?;

        let (field, fuzzy_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("fuzzy query must specify a field".to_string())
        })?;

        let mut query = match fuzzy_spec {
            Value::String(term) => FuzzyQuery::new(field.clone(), term.clone()),
            Value::Object(spec) => {
                let term = spec
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "fuzzy query spec must have 'value' field".to_string(),
                        )
                    })?
                    .to_string();

                let mut q = FuzzyQuery::new(field.clone(), term);

                if let Some(fuzziness) = spec.get("fuzziness") {
                    if let Some(f) = fuzziness.as_u64() {
                        q = q.with_fuzziness(f as u32);
                    } else if fuzziness.as_str() == Some("AUTO") {
                        // AUTO fuzziness: based on term length
                        // 0-2 chars: 0, 3-5 chars: 1, >5 chars: 2
                        q = q.with_fuzziness(2);
                    }
                }

                if let Some(prefix_length) = spec.get("prefix_length").and_then(|v| v.as_u64()) {
                    q = q.with_prefix_length(prefix_length as usize);
                }

                if let Some(max_exp) = spec.get("max_expansions").and_then(|v| v.as_u64()) {
                    q = q.with_max_expansions(max_exp as usize);
                }

                if let Some(boost) = spec.get("boost").and_then(|v| v.as_f64()) {
                    q = q.with_boost(boost as f32);
                }

                q
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "fuzzy query value must be a string or object".to_string(),
                ))
            }
        };

        Ok(Box::new(query))
    }

    /// Parse a match_phrase query
    ///
    /// Format: { "match_phrase": { "field": "exact phrase" } }
    /// or: { "match_phrase": { "field": { "query": "exact phrase", "slop": 0, "boost": 1.0 } } }
    fn parse_match_phrase(value: &Value) -> Result<Box<dyn QueryNode>> {
        let map = value.as_object().ok_or_else(|| {
            SquidexError::InvalidRequest("match_phrase query must be an object".to_string())
        })?;

        let (field, phrase_spec) = map.iter().next().ok_or_else(|| {
            SquidexError::InvalidRequest("match_phrase query must specify a field".to_string())
        })?;

        let query = match phrase_spec {
            Value::String(phrase) => PhraseQuery::new(field.clone(), phrase.clone()),
            Value::Object(spec) => {
                let phrase = spec
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        SquidexError::InvalidRequest(
                            "match_phrase query spec must have 'query' field".to_string(),
                        )
                    })?
                    .to_string();

                let mut q = PhraseQuery::new(field.clone(), phrase);

                if let Some(slop) = spec.get("slop").and_then(|v| v.as_u64()) {
                    q = q.with_slop(slop as u32);
                }

                if let Some(boost) = spec.get("boost").and_then(|v| v.as_f64()) {
                    q = q.with_boost(boost as f32);
                }

                q
            }
            _ => {
                return Err(SquidexError::InvalidRequest(
                    "match_phrase query value must be a string or object".to_string(),
                ))
            }
        };

        Ok(Box::new(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_match_simple() {
        let json = r#"{ "match": { "content": "rust programming" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "match");
    }

    #[test]
    fn test_parse_match_with_options() {
        let json = r#"{
            "match": {
                "content": {
                    "query": "rust programming",
                    "operator": "and",
                    "boost": 2.0
                }
            }
        }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "match");
        assert_eq!(query.boost(), 2.0);
    }

    #[test]
    fn test_parse_term() {
        let json = r#"{ "term": { "status": "published" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "term");
    }

    #[test]
    fn test_parse_term_with_boost() {
        let json = r#"{ "term": { "status": { "value": "published", "boost": 1.5 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "term");
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_parse_terms() {
        let json = r#"{ "terms": { "tags": ["rust", "programming", "tutorial"] } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "terms");
    }

    #[test]
    fn test_parse_range() {
        let json = r#"{ "range": { "year": { "gte": 2020, "lte": 2024 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_parse_range_with_strings() {
        let json = r#"{ "range": { "date": { "gte": "2024-01-01", "lt": "2025-01-01" } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_parse_bool_query() {
        let json = r#"{
            "bool": {
                "must": [
                    { "match": { "content": "rust" } }
                ],
                "should": [
                    { "term": { "tags": "tutorial" } }
                ],
                "must_not": [
                    { "term": { "status": "draft" } }
                ],
                "filter": [
                    { "range": { "year": { "gte": 2024 } } }
                ]
            }
        }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_parse_bool_with_minimum_should_match() {
        let json = r#"{
            "bool": {
                "should": [
                    { "term": { "tags": "rust" } },
                    { "term": { "tags": "programming" } },
                    { "term": { "tags": "tutorial" } }
                ],
                "minimum_should_match": 2
            }
        }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_parse_match_all() {
        let json = r#"{ "match_all": {} }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "all_docs");
    }

    #[test]
    fn test_parse_wrapped_query() {
        let json = r#"{ "query": { "match": { "content": "rust" } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "match");
    }

    #[test]
    fn test_parse_nested_bool() {
        let json = r#"{
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                { "term": { "tag": "rust" } },
                                { "term": { "tag": "go" } }
                            ]
                        }
                    }
                ],
                "filter": [
                    { "range": { "date": { "gte": "2024-01-01" } } }
                ]
            }
        }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = QueryParser::parse_str("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unknown_query_type() {
        let json = r#"{ "unknown_query": { "field": "value" } }"#;
        let result = QueryParser::parse_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_wildcard_simple() {
        let json = r#"{ "wildcard": { "title": "prog*" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "wildcard");
    }

    #[test]
    fn test_parse_wildcard_with_boost() {
        let json = r#"{ "wildcard": { "title": { "value": "prog*", "boost": 2.0 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "wildcard");
        assert_eq!(query.boost(), 2.0);
    }

    #[test]
    fn test_parse_prefix_simple() {
        let json = r#"{ "prefix": { "title": "rust" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "prefix");
    }

    #[test]
    fn test_parse_prefix_with_boost() {
        let json = r#"{ "prefix": { "title": { "value": "rust", "boost": 1.5 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "prefix");
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_parse_fuzzy_simple() {
        let json = r#"{ "fuzzy": { "content": "roust" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "fuzzy");
    }

    #[test]
    fn test_parse_fuzzy_with_options() {
        let json = r#"{ "fuzzy": { "content": { "value": "roust", "fuzziness": 1, "prefix_length": 2 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "fuzzy");
    }

    #[test]
    fn test_parse_match_phrase_simple() {
        let json = r#"{ "match_phrase": { "content": "rust programming" } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_parse_match_phrase_with_slop() {
        let json = r#"{ "match_phrase": { "content": { "query": "rust programming", "slop": 2 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_parse_match_phrase_with_boost() {
        let json = r#"{ "match_phrase": { "content": { "query": "rust programming", "boost": 2.5 } } }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "phrase");
        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_parse_complex_with_new_queries() {
        let json = r#"{
            "bool": {
                "must": [
                    { "match_phrase": { "title": "rust programming" } }
                ],
                "should": [
                    { "prefix": { "tags": "tut" } },
                    { "fuzzy": { "author": "john" } }
                ],
                "must_not": [
                    { "wildcard": { "status": "draft*" } }
                ]
            }
        }"#;
        let query = QueryParser::parse_str(json).unwrap();
        assert_eq!(query.query_type(), "bool");
    }
}
