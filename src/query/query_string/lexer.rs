//! Lexer for query string syntax
//!
//! Tokenizes Lucene-style query strings into a stream of tokens.

use crate::error::SquidexError;
use crate::Result;

/// Token types for query string parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A term (unquoted word)
    Term(String),
    /// A quoted string (phrase)
    QuotedString(String),
    /// A number (integer or float)
    Number(f64),

    /// AND operator
    And,
    /// OR operator
    Or,
    /// NOT operator
    Not,
    /// Colon separator (field:value)
    Colon,

    /// Asterisk wildcard (matches any characters)
    Asterisk,
    /// Question mark wildcard (matches single character)
    Question,

    /// Tilde with optional distance/fuzziness
    Tilde(Option<u32>),
    /// Caret for boosting with optional boost value
    Caret(Option<f32>),

    /// Left square bracket (inclusive range start)
    LeftBracket,
    /// Right square bracket (inclusive range end)
    RightBracket,
    /// Left curly brace (exclusive range start)
    LeftBrace,
    /// Right curly brace (exclusive range end)
    RightBrace,
    /// TO keyword for ranges
    To,

    /// Left parenthesis (grouping)
    LeftParen,
    /// Right parenthesis (grouping)
    RightParen,

    /// Plus sign (required term, equivalent to AND)
    Plus,
    /// Minus sign (excluded term, equivalent to NOT)
    Minus,

    /// End of input
    Eof,
}

impl Token {
    /// Check if this token is a term-like token (can be a field name or value)
    pub fn is_term_like(&self) -> bool {
        matches!(self, Token::Term(_) | Token::Number(_))
    }
}

/// Lexer for tokenizing query strings
pub struct Lexer {
    input: Vec<char>,
    position: usize,
}

impl Lexer {
    /// Create a new lexer for the given input string
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
        }
    }

    /// Get the next token from the input
    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        if self.position >= self.input.len() {
            return Ok(Token::Eof);
        }

        let ch = self.current_char();

        match ch {
            ':' => {
                self.advance();
                Ok(Token::Colon)
            }
            '*' => {
                self.advance();
                Ok(Token::Asterisk)
            }
            '?' => {
                self.advance();
                Ok(Token::Question)
            }
            '~' => {
                self.advance();
                let distance = self.read_unsigned_int();
                Ok(Token::Tilde(distance))
            }
            '^' => {
                self.advance();
                let boost = self.read_float();
                Ok(Token::Caret(boost))
            }
            '[' => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            ']' => {
                self.advance();
                Ok(Token::RightBracket)
            }
            '{' => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            '}' => {
                self.advance();
                Ok(Token::RightBrace)
            }
            '(' => {
                self.advance();
                Ok(Token::LeftParen)
            }
            ')' => {
                self.advance();
                Ok(Token::RightParen)
            }
            '+' => {
                self.advance();
                Ok(Token::Plus)
            }
            '-' => {
                self.advance();
                Ok(Token::Minus)
            }
            '"' => {
                self.advance();
                self.read_quoted_string()
            }
            '\'' => {
                self.advance();
                self.read_single_quoted_string()
            }
            _ if ch.is_ascii_digit() => self.read_numeric_or_term(),
            _ if Self::is_term_start(ch) => self.read_term(),
            _ => Err(SquidexError::QueryParseError(format!(
                "Unexpected character at position {}: '{}'",
                self.position, ch
            ))),
        }
    }

    /// Peek at the next token without consuming it
    pub fn peek_token(&mut self) -> Result<Token> {
        let saved_position = self.position;
        let token = self.next_token()?;
        self.position = saved_position;
        Ok(token)
    }

    /// Check if the lexer has reached the end of input
    pub fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Get remaining input as string (for error messages)
    pub fn remaining(&self) -> String {
        self.input[self.position..].iter().collect()
    }

    fn read_term(&mut self) -> Result<Token> {
        let mut term = String::new();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if Self::is_term_char(ch) {
                term.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Check for keywords (case-insensitive)
        match term.to_uppercase().as_str() {
            "AND" => Ok(Token::And),
            "OR" => Ok(Token::Or),
            "NOT" => Ok(Token::Not),
            "TO" => Ok(Token::To),
            _ => Ok(Token::Term(term)),
        }
    }

    fn read_quoted_string(&mut self) -> Result<Token> {
        let mut s = String::new();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == '"' {
                self.advance();
                return Ok(Token::QuotedString(s));
            }
            if ch == '\\' {
                self.advance();
                if self.position < self.input.len() {
                    let escaped = self.current_char();
                    match escaped {
                        '"' | '\\' => s.push(escaped),
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        _ => {
                            s.push('\\');
                            s.push(escaped);
                        }
                    }
                    self.advance();
                }
            } else {
                s.push(ch);
                self.advance();
            }
        }

        Err(SquidexError::QueryParseError(
            "Unterminated quoted string".to_string(),
        ))
    }

    fn read_single_quoted_string(&mut self) -> Result<Token> {
        let mut s = String::new();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch == '\'' {
                self.advance();
                return Ok(Token::QuotedString(s));
            }
            if ch == '\\' {
                self.advance();
                if self.position < self.input.len() {
                    let escaped = self.current_char();
                    match escaped {
                        '\'' | '\\' => s.push(escaped),
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        _ => {
                            s.push('\\');
                            s.push(escaped);
                        }
                    }
                    self.advance();
                }
            } else {
                s.push(ch);
                self.advance();
            }
        }

        Err(SquidexError::QueryParseError(
            "Unterminated single-quoted string".to_string(),
        ))
    }

    /// Read a numeric token or a term that starts with digits (like dates)
    fn read_numeric_or_term(&mut self) -> Result<Token> {
        let start_pos = self.position;
        let mut str_val = String::new();
        let mut has_dot = false;
        let mut has_non_numeric = false;

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                str_val.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot {
                // Check if next char is a digit (to avoid matching "123.")
                if self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    has_dot = true;
                    str_val.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else if ch == '-' && self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                // Date-like format: 2024-01-15
                has_non_numeric = true;
                str_val.push(ch);
                self.advance();
            } else if Self::is_term_char(ch) && (ch == '_' || ch.is_alphabetic()) {
                // Mixed alphanumeric term
                has_non_numeric = true;
                str_val.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if has_non_numeric {
            // It's a term (like a date or mixed alphanumeric)
            Ok(Token::Term(str_val))
        } else {
            // It's a pure number
            str_val
                .parse::<f64>()
                .map(Token::Number)
                .map_err(|_| SquidexError::QueryParseError(format!("Invalid number: {}", str_val)))
        }
    }

    fn read_unsigned_int(&mut self) -> Option<u32> {
        let mut num_str = String::new();

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        num_str.parse().ok()
    }

    fn read_float(&mut self) -> Option<f32> {
        let mut num_str = String::new();
        let mut has_dot = false;

        while self.position < self.input.len() {
            let ch = self.current_char();
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot {
                has_dot = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        num_str.parse().ok()
    }

    fn current_char(&self) -> char {
        self.input[self.position]
    }

    fn peek(&self) -> Option<char> {
        if self.position + 1 < self.input.len() {
            Some(self.input[self.position + 1])
        } else {
            None
        }
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.current_char().is_whitespace() {
            self.advance();
        }
    }

    /// Check if a character can start a term
    fn is_term_start(ch: char) -> bool {
        ch.is_alphanumeric() || ch == '_' || ch == '@' || ch == '#'
    }

    /// Check if a character can be part of a term
    fn is_term_char(ch: char) -> bool {
        ch.is_alphanumeric()
            || ch == '_'
            || ch == '-'
            || ch == '.'
            || ch == '@'
            || ch == '#'
            || ch == '*'
            || ch == '?'
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_term() {
        let mut lexer = Lexer::new("hello");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("hello".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_field_value() {
        let mut lexer = Lexer::new("title:rust");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("title".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("rust".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_boolean_operators() {
        let mut lexer = Lexer::new("a AND b OR c NOT d");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::And);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("b".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Or);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("c".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Not);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("d".to_string()));
    }

    #[test]
    fn test_quoted_string() {
        let mut lexer = Lexer::new("\"hello world\"");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::QuotedString("hello world".to_string())
        );
    }

    #[test]
    fn test_quoted_string_escaped() {
        let mut lexer = Lexer::new("\"hello \\\"world\\\"\"");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::QuotedString("hello \"world\"".to_string())
        );
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14");
        assert_eq!(lexer.next_token().unwrap(), Token::Number(42.0));
        assert_eq!(lexer.next_token().unwrap(), Token::Number(3.14));
    }

    #[test]
    fn test_minus_before_number() {
        // Minus is treated as operator, not negative number
        let mut lexer = Lexer::new("-10");
        assert_eq!(lexer.next_token().unwrap(), Token::Minus);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(10.0));
    }

    #[test]
    fn test_fuzzy() {
        let mut lexer = Lexer::new("rust~2");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("rust".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Tilde(Some(2)));
    }

    #[test]
    fn test_fuzzy_no_distance() {
        let mut lexer = Lexer::new("rust~");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("rust".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Tilde(None));
    }

    #[test]
    fn test_boost() {
        let mut lexer = Lexer::new("rust^2.5");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("rust".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Caret(Some(2.5)));
    }

    #[test]
    fn test_wildcard_in_term() {
        let mut lexer = Lexer::new("prog*");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("prog*".to_string()));
    }

    #[test]
    fn test_range() {
        let mut lexer = Lexer::new("[10 TO 20]");
        assert_eq!(lexer.next_token().unwrap(), Token::LeftBracket);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(10.0));
        assert_eq!(lexer.next_token().unwrap(), Token::To);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(20.0));
        assert_eq!(lexer.next_token().unwrap(), Token::RightBracket);
    }

    #[test]
    fn test_exclusive_range() {
        let mut lexer = Lexer::new("{10 TO 20}");
        assert_eq!(lexer.next_token().unwrap(), Token::LeftBrace);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(10.0));
        assert_eq!(lexer.next_token().unwrap(), Token::To);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(20.0));
        assert_eq!(lexer.next_token().unwrap(), Token::RightBrace);
    }

    #[test]
    fn test_grouping() {
        let mut lexer = Lexer::new("(a OR b)");
        assert_eq!(lexer.next_token().unwrap(), Token::LeftParen);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Or);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("b".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::RightParen);
    }

    #[test]
    fn test_plus_minus() {
        let mut lexer = Lexer::new("+required -excluded");
        assert_eq!(lexer.next_token().unwrap(), Token::Plus);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Term("required".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Minus);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Term("excluded".to_string())
        );
    }

    #[test]
    fn test_complex_query() {
        let mut lexer =
            Lexer::new("title:rust AND (tags:tutorial OR tags:guide) AND year:[2020 TO 2024]");

        // title:rust
        assert_eq!(lexer.next_token().unwrap(), Token::Term("title".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("rust".to_string()));

        // AND
        assert_eq!(lexer.next_token().unwrap(), Token::And);

        // (tags:tutorial OR tags:guide)
        assert_eq!(lexer.next_token().unwrap(), Token::LeftParen);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("tags".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Term("tutorial".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Or);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("tags".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("guide".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::RightParen);

        // AND year:[2020 TO 2024]
        assert_eq!(lexer.next_token().unwrap(), Token::And);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("year".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::LeftBracket);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(2020.0));
        assert_eq!(lexer.next_token().unwrap(), Token::To);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(2024.0));
        assert_eq!(lexer.next_token().unwrap(), Token::RightBracket);

        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_unterminated_string() {
        let mut lexer = Lexer::new("\"unterminated");
        assert!(lexer.next_token().is_err());
    }

    #[test]
    fn test_case_insensitive_operators() {
        let mut lexer = Lexer::new("a and b or c not d");
        assert_eq!(lexer.next_token().unwrap(), Token::Term("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::And);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("b".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Or);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("c".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Not);
        assert_eq!(lexer.next_token().unwrap(), Token::Term("d".to_string()));
    }

    #[test]
    fn test_date_like_term() {
        let mut lexer = Lexer::new("2024-01-15");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Term("2024-01-15".to_string())
        );
    }
}
