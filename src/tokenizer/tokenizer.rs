use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use stop_words::{get, LANGUAGE};
use unicode_segmentation::UnicodeSegmentation;

use crate::config::TokenizerConfig;

/// Text tokenizer with stemming and stopword removal
pub struct Tokenizer {
    config: TokenizerConfig,
    stemmer: Option<Stemmer>,
    stopwords: HashSet<String>,
}

impl Tokenizer {
    /// Create a new tokenizer from configuration
    pub fn new(config: &TokenizerConfig) -> Self {
        let stemmer = if config.stem {
            Some(Stemmer::create(Algorithm::English))
        } else {
            None
        };

        let stopwords = if config.remove_stopwords {
            get(LANGUAGE::English)
                .into_iter()
                .map(|s| s.to_lowercase())
                .collect()
        } else {
            HashSet::new()
        };

        Self {
            config: config.clone(),
            stemmer,
            stopwords,
        }
    }

    /// Tokenize text into a vector of terms
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text
            .unicode_words()
            .map(|word| {
                let mut token = word.to_string();

                if self.config.lowercase {
                    token = token.to_lowercase();
                }

                token
            })
            .filter(|token| {
                token.len() >= self.config.min_token_length
                    && token.len() <= self.config.max_token_length
                    && !self.stopwords.contains(token)
            })
            .collect();

        if let Some(stemmer) = &self.stemmer {
            tokens = tokens
                .into_iter()
                .map(|token| stemmer.stem(&token).to_string())
                .collect();
        }

        tokens
    }

    /// Compute term frequencies for a tokenized document
    pub fn compute_term_frequencies(&self, text: &str) -> std::collections::HashMap<String, u32> {
        let tokens = self.tokenize(text);
        let mut freq = std::collections::HashMap::new();
        for token in tokens {
            *freq.entry(token).or_insert(0) += 1;
        }
        freq
    }

    /// Get unique terms from text
    pub fn unique_terms(&self, text: &str) -> HashSet<String> {
        self.tokenize(text).into_iter().collect()
    }

    /// Tokenize text and track positions for phrase queries
    ///
    /// Returns a map from each term to its positions in the document.
    /// Positions are 0-indexed and count all tokens including stopwords
    /// (stopwords are filtered but still increment position).
    ///
    /// # Example
    ///
    /// ```
    /// use squidex::tokenizer::Tokenizer;
    /// use squidex::config::TokenizerConfig;
    ///
    /// let tokenizer = Tokenizer::new(&TokenizerConfig::default());
    /// let positions = tokenizer.tokenize_with_positions("hello world hello");
    /// // "hello" appears at positions 0 and 2
    /// // "world" appears at position 1
    /// ```
    pub fn tokenize_with_positions(&self, text: &str) -> std::collections::HashMap<String, Vec<u32>> {
        let mut positions: std::collections::HashMap<String, Vec<u32>> = std::collections::HashMap::new();
        let mut pos = 0u32;

        for word in text.unicode_words() {
            let mut token = word.to_string();

            if self.config.lowercase {
                token = token.to_lowercase();
            }

            // Check length constraints
            if token.len() < self.config.min_token_length
                || token.len() > self.config.max_token_length
            {
                pos += 1;
                continue;
            }

            // Check stopwords (after lowercasing)
            if self.stopwords.contains(&token) {
                pos += 1;
                continue;
            }

            // Apply stemming
            if let Some(stemmer) = &self.stemmer {
                token = stemmer.stem(&token).to_string();
            }

            positions.entry(token).or_default().push(pos);
            pos += 1;
        }

        positions
    }

    /// Tokenize and return (term, position) pairs in order
    ///
    /// This is useful for building posting lists with position data.
    pub fn tokenize_with_positions_ordered(&self, text: &str) -> Vec<(String, u32)> {
        let mut results = Vec::new();
        let mut pos = 0u32;

        for word in text.unicode_words() {
            let mut token = word.to_string();

            if self.config.lowercase {
                token = token.to_lowercase();
            }

            // Check length constraints
            if token.len() < self.config.min_token_length
                || token.len() > self.config.max_token_length
            {
                pos += 1;
                continue;
            }

            // Check stopwords
            if self.stopwords.contains(&token) {
                pos += 1;
                continue;
            }

            // Apply stemming
            if let Some(stemmer) = &self.stemmer {
                token = stemmer.stem(&token).to_string();
            }

            results.push((token, pos));
            pos += 1;
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let tokens = tokenizer.tokenize("Hello World! This is a test.");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_stopword_removal() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: true,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let tokens = tokenizer.tokenize("This is a document about the system");

        // Common stopwords should be removed
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"the".to_string()));

        // At least some content words should remain
        let has_content = !tokens.is_empty();
        assert!(
            has_content,
            "Should have some tokens after stopword removal"
        );
    }

    #[test]
    fn test_stemming() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: true,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let tokens = tokenizer.tokenize("running runs runner");

        // All should stem to "run"
        assert!(tokens.iter().all(|t| t.starts_with("run")));
    }

    #[test]
    fn test_term_frequencies() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };
        let tokenizer = Tokenizer::new(&config);

        let freq = tokenizer.compute_term_frequencies("apple apple banana");
        assert_eq!(freq.get("apple"), Some(&2));
        assert_eq!(freq.get("banana"), Some(&1));
    }

    #[test]
    fn test_min_max_token_length() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: false,
            min_token_length: 3,
            max_token_length: 5,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let tokens = tokenizer.tokenize("a ab abc abcd abcde abcdef");

        // Only tokens with length 3-5 should remain
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"ab".to_string()));
        assert!(tokens.contains(&"abc".to_string()));
        assert!(tokens.contains(&"abcd".to_string()));
        assert!(tokens.contains(&"abcde".to_string()));
        assert!(!tokens.contains(&"abcdef".to_string()));
    }

    #[test]
    fn test_tokenize_with_positions() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let positions = tokenizer.tokenize_with_positions("hello world hello again");

        assert_eq!(positions.get("hello"), Some(&vec![0, 2]));
        assert_eq!(positions.get("world"), Some(&vec![1]));
        assert_eq!(positions.get("again"), Some(&vec![3]));
    }

    #[test]
    fn test_tokenize_with_positions_stopwords() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: true,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        // "the" is a stopword, so positions should skip it but count it
        let positions = tokenizer.tokenize_with_positions("rust the programming");

        // "rust" is at position 0
        // "the" is filtered but position increments to 1
        // "programming" is at position 2
        assert_eq!(positions.get("rust"), Some(&vec![0]));
        assert_eq!(positions.get("programming"), Some(&vec![2]));
        assert!(positions.get("the").is_none());
    }

    #[test]
    fn test_tokenize_with_positions_ordered() {
        let config = TokenizerConfig {
            lowercase: true,
            remove_stopwords: false,
            stem: false,
            min_token_length: 2,
            max_token_length: 50,
            language: "english".to_string(),
        };

        let tokenizer = Tokenizer::new(&config);
        let ordered = tokenizer.tokenize_with_positions_ordered("rust is great for programming");

        assert_eq!(ordered.len(), 5);
        assert_eq!(ordered[0], ("rust".to_string(), 0));
        assert_eq!(ordered[1], ("is".to_string(), 1));
        assert_eq!(ordered[2], ("great".to_string(), 2));
        assert_eq!(ordered[3], ("for".to_string(), 3));
        assert_eq!(ordered[4], ("programming".to_string(), 4));
    }
}
