//! Recursive descent parser for query strings
//!
//! # Grammar
//!
//! ```text
//! query       := or_expr
//! or_expr     := and_expr (OR and_expr)*
//! and_expr    := not_expr (AND? not_expr)*
//! not_expr    := (NOT | '-')? primary
//! primary     := field_query | grouped | term_expr
//! field_query := TERM COLON value_expr
//! value_expr  := range | phrase | term_with_modifiers
//! range       := '[' value TO value ']' | '{' value TO value '}'
//! phrase      := QUOTED modifiers?
//! term_with_modifiers := term modifiers?
//! modifiers   := (TILDE distance?)? (CARET boost?)?
//! grouped     := '(' or_expr ')'
//! ```

use super::lexer::{Lexer, Token};
use crate::error::SquidexError;
use crate::query::ast::QueryNode;
use crate::query::nodes::{BoolQuery, MatchQuery, RangeQuery, TermQuery};
use crate::query::types::{MatchOperator, RangeBounds, RangeValue};
use crate::Result;

// Forward declare new query types that we'll create
use crate::query::nodes::{FuzzyQuery, PhraseQuery, PrefixQuery, WildcardQuery};

/// Default field to search when no field is specified
const DEFAULT_FIELD: &str = "content";

/// Parser for Lucene-style query strings
pub struct QueryStringParser {
    lexer: Lexer,
    current_token: Token,
    /// Default field for unqualified terms
    default_field: String,
    /// Default operator between terms (AND or OR)
    default_operator: MatchOperator,
}

impl QueryStringParser {
    /// Create a new parser for the given query string
    pub fn new(input: &str) -> Result<Self> {
        let mut lexer = Lexer::new(input);
        let current_token = lexer.next_token()?;

        Ok(Self {
            lexer,
            current_token,
            default_field: DEFAULT_FIELD.to_string(),
            default_operator: MatchOperator::Or,
        })
    }

    /// Set the default field for unqualified terms
    pub fn with_default_field(mut self, field: impl Into<String>) -> Self {
        self.default_field = field.into();
        self
    }

    /// Set the default operator between terms
    pub fn with_default_operator(mut self, operator: MatchOperator) -> Self {
        self.default_operator = operator;
        self
    }

    /// Parse the query string into a query AST
    pub fn parse(&mut self) -> Result<Box<dyn QueryNode>> {
        let query = self.parse_or_expr()?;

        // Ensure we've consumed all input
        if self.current_token != Token::Eof {
            return Err(SquidexError::QueryParseError(format!(
                "Unexpected token after query: {:?}",
                self.current_token
            )));
        }

        Ok(query)
    }

    /// Parse: or_expr := and_expr (OR and_expr)*
    fn parse_or_expr(&mut self) -> Result<Box<dyn QueryNode>> {
        let mut clauses = vec![self.parse_and_expr()?];

        while self.current_token == Token::Or {
            self.advance()?;
            clauses.push(self.parse_and_expr()?);
        }

        if clauses.len() == 1 {
            Ok(clauses.into_iter().next().unwrap())
        } else {
            Ok(Box::new(BoolQuery {
                must: vec![],
                should: clauses,
                must_not: vec![],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }))
        }
    }

    /// Parse: and_expr := not_expr (AND? not_expr)*
    ///
    /// When default_operator is OR, adjacent terms (without explicit AND) are collected
    /// and combined with OR. When default_operator is AND, they're combined with AND.
    fn parse_and_expr(&mut self) -> Result<Box<dyn QueryNode>> {
        let mut explicit_and_clauses = vec![self.parse_not_expr()?];
        let mut implicit_clauses: Vec<Box<dyn QueryNode>> = vec![];

        loop {
            // Explicit AND - always combine with must
            if self.current_token == Token::And {
                self.advance()?;
                // If we had implicit clauses, they need to be combined first
                if !implicit_clauses.is_empty() {
                    let last = explicit_and_clauses.pop().unwrap();
                    implicit_clauses.insert(0, last);
                    let combined = self.combine_with_default_operator(implicit_clauses);
                    explicit_and_clauses.push(combined);
                    implicit_clauses = vec![];
                }
                explicit_and_clauses.push(self.parse_not_expr()?);
            }
            // Implicit combination when next token is NOT/Minus or a primary expression
            else if self.current_token == Token::Not || self.current_token == Token::Minus {
                // NOT/Minus always starts a new clause that gets AND-ed
                explicit_and_clauses.push(self.parse_not_expr()?);
            }
            // Implicit combination for adjacent terms (based on default_operator)
            else if self.is_start_of_primary() {
                implicit_clauses.push(self.parse_not_expr()?);
            } else {
                break;
            }
        }

        // Combine any remaining implicit clauses with the last explicit clause
        if !implicit_clauses.is_empty() {
            let last = explicit_and_clauses.pop().unwrap();
            implicit_clauses.insert(0, last);
            let combined = self.combine_with_default_operator(implicit_clauses);
            explicit_and_clauses.push(combined);
        }

        if explicit_and_clauses.len() == 1 {
            Ok(explicit_and_clauses.into_iter().next().unwrap())
        } else {
            Ok(Box::new(BoolQuery {
                must: explicit_and_clauses,
                should: vec![],
                must_not: vec![],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }))
        }
    }

    /// Combine clauses using the default operator (AND -> must, OR -> should)
    fn combine_with_default_operator(&self, clauses: Vec<Box<dyn QueryNode>>) -> Box<dyn QueryNode> {
        if clauses.len() == 1 {
            return clauses.into_iter().next().unwrap();
        }

        match self.default_operator {
            MatchOperator::And => Box::new(BoolQuery {
                must: clauses,
                should: vec![],
                must_not: vec![],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }),
            MatchOperator::Or => Box::new(BoolQuery {
                must: vec![],
                should: clauses,
                must_not: vec![],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }),
        }
    }

    /// Parse: not_expr := (NOT | '-')? primary | ('+')? primary
    fn parse_not_expr(&mut self) -> Result<Box<dyn QueryNode>> {
        let is_negated = if self.current_token == Token::Not || self.current_token == Token::Minus {
            self.advance()?;
            true
        } else {
            false
        };

        let is_required = if self.current_token == Token::Plus {
            self.advance()?;
            true
        } else {
            false
        };

        let inner = self.parse_primary()?;

        if is_negated {
            Ok(Box::new(BoolQuery {
                must: vec![],
                should: vec![],
                must_not: vec![inner],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }))
        } else if is_required {
            Ok(Box::new(BoolQuery {
                must: vec![inner],
                should: vec![],
                must_not: vec![],
                filter: vec![],
                minimum_should_match: Default::default(),
                boost: 1.0,
            }))
        } else {
            Ok(inner)
        }
    }

    /// Parse: primary := field_query | grouped | term_expr
    fn parse_primary(&mut self) -> Result<Box<dyn QueryNode>> {
        match &self.current_token {
            Token::LeftParen => {
                self.advance()?; // consume '('
                let expr = self.parse_or_expr()?;
                self.expect(Token::RightParen)?;
                Ok(expr)
            }
            Token::Term(term) => {
                let term_str = term.clone();
                self.advance()?;

                // Check if this is a field query (term followed by colon)
                if self.current_token == Token::Colon {
                    self.advance()?; // consume ':'
                    self.parse_field_value(&term_str)
                } else {
                    // Unqualified term - search in default field
                    self.parse_term_with_modifiers(&self.default_field.clone(), term_str)
                }
            }
            Token::QuotedString(text) => {
                let phrase = text.clone();
                self.advance()?;

                // Parse modifiers (slop, boost)
                let (slop, boost) = self.parse_modifiers()?;

                Ok(Box::new(PhraseQuery::new(
                    self.default_field.clone(),
                    phrase,
                )
                .with_slop(slop.unwrap_or(0))
                .with_boost(boost.unwrap_or(1.0))))
            }
            Token::Number(n) => {
                let term = n.to_string();
                self.advance()?;
                Ok(Box::new(
                    TermQuery::new(self.default_field.clone(), term).with_boost(1.0),
                ))
            }
            Token::Asterisk => {
                // Standalone wildcard - match all
                self.advance()?;
                Ok(Box::new(MatchQuery::new(
                    self.default_field.clone(),
                    "*".to_string(),
                )))
            }
            _ => Err(SquidexError::QueryParseError(format!(
                "Unexpected token: {:?}",
                self.current_token
            ))),
        }
    }

    /// Parse field value after field:
    fn parse_field_value(&mut self, field: &str) -> Result<Box<dyn QueryNode>> {
        match &self.current_token {
            Token::LeftBracket | Token::LeftBrace => self.parse_range_query(field),
            Token::QuotedString(text) => {
                let phrase = text.clone();
                self.advance()?;

                let (slop, boost) = self.parse_modifiers()?;

                Ok(Box::new(
                    PhraseQuery::new(field, phrase)
                        .with_slop(slop.unwrap_or(0))
                        .with_boost(boost.unwrap_or(1.0)),
                ))
            }
            Token::Term(term) => {
                let term_str = term.clone();
                self.advance()?;
                self.parse_term_with_modifiers(field, term_str)
            }
            Token::Number(n) => {
                let term = n.to_string();
                self.advance()?;
                let (_, boost) = self.parse_modifiers()?;
                Ok(Box::new(
                    TermQuery::new(field, term).with_boost(boost.unwrap_or(1.0)),
                ))
            }
            Token::Asterisk => {
                // field:* matches all documents with the field
                self.advance()?;
                Ok(Box::new(MatchQuery::new(field, "*".to_string())))
            }
            Token::LeftParen => {
                // Grouped query for field: field:(a OR b)
                self.advance()?;
                let inner = self.parse_or_expr()?;
                self.expect(Token::RightParen)?;
                // TODO: Apply field to inner query
                Ok(inner)
            }
            _ => Err(SquidexError::QueryParseError(format!(
                "Expected value after field '{}:', got {:?}",
                field, self.current_token
            ))),
        }
    }

    /// Parse a term with optional wildcard/fuzzy/boost modifiers
    fn parse_term_with_modifiers(&mut self, field: &str, term: String) -> Result<Box<dyn QueryNode>> {
        // Check for wildcards in term
        let has_wildcard = term.contains('*') || term.contains('?');

        // Check for prefix query (term ends with *)
        let is_prefix = term.ends_with('*') && !term[..term.len() - 1].contains('*');

        // Parse modifiers (fuzzy, boost)
        let (fuzziness, boost) = self.parse_modifiers()?;

        if let Some(fuzz) = fuzziness {
            // Fuzzy query
            Ok(Box::new(
                FuzzyQuery::new(field, term)
                    .with_fuzziness(fuzz)
                    .with_boost(boost.unwrap_or(1.0)),
            ))
        } else if is_prefix {
            // Prefix query
            let prefix = term[..term.len() - 1].to_string();
            Ok(Box::new(
                PrefixQuery::new(field, prefix).with_boost(boost.unwrap_or(1.0)),
            ))
        } else if has_wildcard {
            // Wildcard query
            Ok(Box::new(
                WildcardQuery::new(field, term).with_boost(boost.unwrap_or(1.0)),
            ))
        } else {
            // Exact term query
            Ok(Box::new(
                TermQuery::new(field, term).with_boost(boost.unwrap_or(1.0)),
            ))
        }
    }

    /// Parse optional modifiers (tilde for fuzzy/slop, caret for boost)
    fn parse_modifiers(&mut self) -> Result<(Option<u32>, Option<f32>)> {
        let mut fuzziness = None;
        let mut boost = None;

        // Check for fuzzy/slop modifier
        if let Token::Tilde(distance) = &self.current_token {
            fuzziness = Some(distance.unwrap_or(2));
            self.advance()?;
        }

        // Check for boost modifier
        if let Token::Caret(boost_val) = &self.current_token {
            boost = Some(boost_val.unwrap_or(2.0));
            self.advance()?;
        }

        Ok((fuzziness, boost))
    }

    /// Parse range query: [low TO high] or {low TO high}
    fn parse_range_query(&mut self, field: &str) -> Result<Box<dyn QueryNode>> {
        let inclusive_lower = self.current_token == Token::LeftBracket;
        self.advance()?; // consume '[' or '{'

        let lower = self.parse_range_value()?;
        self.expect(Token::To)?;
        let upper = self.parse_range_value()?;

        let inclusive_upper = self.current_token == Token::RightBracket;
        if self.current_token != Token::RightBracket && self.current_token != Token::RightBrace {
            return Err(SquidexError::QueryParseError(
                "Expected ']' or '}' at end of range".to_string(),
            ));
        }
        self.advance()?;

        // Parse optional boost
        let (_, boost) = self.parse_modifiers()?;

        let bounds = RangeBounds {
            gte: if inclusive_lower { lower.clone() } else { None },
            gt: if !inclusive_lower { lower } else { None },
            lte: if inclusive_upper { upper.clone() } else { None },
            lt: if !inclusive_upper { upper } else { None },
            boost: boost.unwrap_or(1.0),
        };

        Ok(Box::new(RangeQuery::new(field).with_bounds(bounds)))
    }

    /// Parse a single range value
    fn parse_range_value(&mut self) -> Result<Option<RangeValue>> {
        match &self.current_token {
            Token::Asterisk => {
                self.advance()?;
                Ok(None) // Unbounded
            }
            Token::Number(n) => {
                let val = *n;
                self.advance()?;
                if val.fract() == 0.0 {
                    Ok(Some(RangeValue::Long(val as i64)))
                } else {
                    Ok(Some(RangeValue::Double(val)))
                }
            }
            Token::Term(s) => {
                let term = s.clone();
                self.advance()?;
                // Try to parse as date or keep as string
                if let Ok(i) = term.parse::<i64>() {
                    Ok(Some(RangeValue::Long(i)))
                } else {
                    Ok(Some(RangeValue::String(term)))
                }
            }
            Token::QuotedString(s) => {
                let val = s.clone();
                self.advance()?;
                Ok(Some(RangeValue::String(val)))
            }
            _ => Err(SquidexError::QueryParseError(format!(
                "Expected range value, got: {:?}",
                self.current_token
            ))),
        }
    }

    /// Check if current token can start a primary expression
    fn is_start_of_primary(&self) -> bool {
        matches!(
            self.current_token,
            Token::Term(_)
                | Token::QuotedString(_)
                | Token::Number(_)
                | Token::LeftParen
                | Token::Plus
                | Token::Minus
                | Token::Asterisk
        )
    }

    /// Advance to the next token
    fn advance(&mut self) -> Result<()> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    /// Expect a specific token and advance
    fn expect(&mut self, expected: Token) -> Result<()> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(&expected) {
            self.advance()
        } else {
            Err(SquidexError::QueryParseError(format!(
                "Expected {:?}, got {:?}",
                expected, self.current_token
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_query(input: &str) -> Result<Box<dyn QueryNode>> {
        QueryStringParser::new(input)?.parse()
    }

    #[test]
    fn test_simple_term() {
        let query = parse_query("rust").unwrap();
        assert_eq!(query.query_type(), "term");
    }

    #[test]
    fn test_field_term() {
        let query = parse_query("title:rust").unwrap();
        assert_eq!(query.query_type(), "term");
    }

    #[test]
    fn test_quoted_phrase() {
        let query = parse_query("\"hello world\"").unwrap();
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_field_phrase() {
        let query = parse_query("title:\"hello world\"").unwrap();
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_phrase_with_slop() {
        let query = parse_query("\"hello world\"~2").unwrap();
        assert_eq!(query.query_type(), "phrase");
    }

    #[test]
    fn test_boolean_and() {
        let query = parse_query("rust AND programming").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_boolean_or() {
        let query = parse_query("rust OR python").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_boolean_not() {
        let query = parse_query("NOT draft").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_minus_exclusion() {
        let query = parse_query("-draft").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_plus_required() {
        let query = parse_query("+rust").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_grouped_query() {
        let query = parse_query("(rust OR python) AND programming").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_wildcard_query() {
        let query = parse_query("title:prog*").unwrap();
        assert_eq!(query.query_type(), "prefix");
    }

    #[test]
    fn test_wildcard_middle() {
        let query = parse_query("title:p*ing").unwrap();
        assert_eq!(query.query_type(), "wildcard");
    }

    #[test]
    fn test_fuzzy_query() {
        let query = parse_query("title:rust~2").unwrap();
        assert_eq!(query.query_type(), "fuzzy");
    }

    #[test]
    fn test_fuzzy_default_distance() {
        let query = parse_query("title:rust~").unwrap();
        assert_eq!(query.query_type(), "fuzzy");
    }

    #[test]
    fn test_boost() {
        let query = parse_query("title:rust^2.5").unwrap();
        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_range_inclusive() {
        let query = parse_query("year:[2020 TO 2024]").unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_range_exclusive() {
        let query = parse_query("year:{2020 TO 2024}").unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_range_unbounded_lower() {
        let query = parse_query("year:[* TO 2024]").unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_range_unbounded_upper() {
        let query = parse_query("year:[2020 TO *]").unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_complex_query() {
        let query =
            parse_query("title:rust AND (tags:tutorial OR tags:guide) NOT status:draft").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_field_with_date_range() {
        let query = parse_query("created_at:[2024-01-01 TO 2024-12-31]").unwrap();
        assert_eq!(query.query_type(), "range");
    }

    #[test]
    fn test_default_operator_and() {
        let query = QueryStringParser::new("rust programming")
            .unwrap()
            .with_default_operator(MatchOperator::And)
            .parse()
            .unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_default_operator_or() {
        // Default operator is OR, so "rust programming" becomes a bool with should clauses
        let query = QueryStringParser::new("rust programming")
            .unwrap()
            .with_default_operator(MatchOperator::Or)
            .parse()
            .unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_implicit_or_with_default() {
        // Default operator is OR by default
        let query = parse_query("rust programming").unwrap();
        assert_eq!(query.query_type(), "bool");
    }

    #[test]
    fn test_empty_query() {
        // Empty query should error
        let result = parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_unmatched_paren() {
        let result = parse_query("(rust AND python");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_fields() {
        let query = parse_query("title:rust AND content:programming AND tags:tutorial").unwrap();
        assert_eq!(query.query_type(), "bool");
    }
}
