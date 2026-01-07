/// Scoring functions for search operations

/// BM25 parameters
pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// Compute BM25 score for a term in a document
///
/// # Arguments
/// * `tf` - Term frequency in document
/// * `df` - Document frequency (how many documents contain the term)
/// * `total_docs` - Total number of documents in the index
/// * `doc_len` - Length of the document (in tokens)
/// * `avg_doc_len` - Average document length across all documents
///
/// # Returns
/// BM25 relevance score
pub fn bm25_score(tf: f32, df: f32, total_docs: f32, doc_len: f32, avg_doc_len: f32) -> f32 {
    // Inverse document frequency
    let idf = ((total_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

    // Length normalization
    let norm = 1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len);

    // BM25 formula
    idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * norm)
}

/// Compute cosine similarity between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity score in range [0, 1]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).max(0.0).min(1.0)
    }
}

/// Compute Euclidean distance-based similarity between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Similarity score (higher is better)
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dist: f32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();

    1.0 / (1.0 + dist)
}

/// Compute dot product between two vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product score
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalize a score to [0, 1] range given a maximum possible score
///
/// # Arguments
/// * `score` - Raw score
/// * `max_score` - Maximum possible score
///
/// # Returns
/// Normalized score in [0, 1]
pub fn normalize_score(score: f32, max_score: f32) -> f32 {
    if max_score == 0.0 {
        0.0
    } else {
        (score / max_score).max(0.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_score() {
        // Test with typical values
        let score = bm25_score(5.0, 10.0, 1000.0, 100.0, 100.0);
        assert!(score > 0.0);

        // Higher TF should give higher score (with same other params)
        let score1 = bm25_score(1.0, 10.0, 1000.0, 100.0, 100.0);
        let score2 = bm25_score(5.0, 10.0, 1000.0, 100.0, 100.0);
        assert!(score2 > score1);

        // Lower DF (rarer term) should give higher score
        let score1 = bm25_score(5.0, 100.0, 1000.0, 100.0, 100.0);
        let score2 = bm25_score(5.0, 10.0, 1000.0, 100.0, 100.0);
        assert!(score2 > score1);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim < 1e-5);

        // Opposite vectors
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0); // Clamped to 0

        // Zero vector
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_euclidean_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = euclidean_similarity(&a, &b);
        assert_eq!(sim, 1.0);

        // Different vectors
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = euclidean_similarity(&a, &b);
        assert_eq!(sim, 1.0 / 2.0); // distance is 1, so similarity is 1/(1+1) = 0.5

        // Further apart
        let a = vec![0.0, 0.0];
        let b = vec![10.0, 0.0];
        let sim = euclidean_similarity(&a, &b);
        assert!(sim < 0.1); // Should be small
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert_eq!(dot, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dot = dot_product(&a, &b);
        assert_eq!(dot, 0.0);
    }

    #[test]
    fn test_normalize_score() {
        assert_eq!(normalize_score(5.0, 10.0), 0.5);
        assert_eq!(normalize_score(10.0, 10.0), 1.0);
        assert_eq!(normalize_score(0.0, 10.0), 0.0);
        assert_eq!(normalize_score(15.0, 10.0), 1.0); // Clamped to 1.0
        assert_eq!(normalize_score(5.0, 0.0), 0.0); // Max is zero
    }

    #[test]
    fn test_mismatched_dimensions() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(euclidean_similarity(&a, &b), 0.0);
        assert_eq!(dot_product(&a, &b), 0.0);
    }
}
