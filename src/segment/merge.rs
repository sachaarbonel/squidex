//! Tiered merge policy for segment management
//!
//! Segment size & merge policy:
//! - max_merged_segment = 5GB (NVMe-friendly)
//! - segments_per_tier = 10
//! - Merge score accounts for size, deletes %, and search cost (segment count)

use std::sync::Arc;

use super::reader::{SegmentMeta, SegmentReader};
use super::types::SegmentId;

/// Configuration for the tiered merge policy
#[derive(Clone, Debug)]
pub struct MergePolicyConfig {
    /// Maximum size for a merged segment (default: 5GB)
    pub max_merged_segment_bytes: u64,
    /// Target number of segments per tier (default: 10)
    pub segments_per_tier: usize,
    /// Minimum number of segments to merge at once
    pub min_merge_count: usize,
    /// Maximum number of segments to merge at once
    pub max_merge_count: usize,
    /// Delete ratio threshold to force merge (default: 0.15 = 15%)
    pub delete_ratio_threshold: f64,
    /// Minimum segment size to be considered for merging
    pub floor_segment_bytes: u64,
}

impl Default for MergePolicyConfig {
    fn default() -> Self {
        Self {
            max_merged_segment_bytes: 5 * 1024 * 1024 * 1024, // 5GB
            segments_per_tier: 10,
            min_merge_count: 2,
            max_merge_count: 10,
            delete_ratio_threshold: 0.15,
            floor_segment_bytes: 1024 * 1024, // 1MB
        }
    }
}

/// A candidate merge operation
#[derive(Clone, Debug)]
pub struct MergeCandidate {
    /// Segment IDs to merge
    pub segment_ids: Vec<SegmentId>,
    /// Total size after merge (estimate)
    pub estimated_size: u64,
    /// Merge score (higher = more urgent)
    pub score: f64,
    /// Reason for merge
    pub reason: MergeReason,
}

/// Reason why segments should be merged
#[derive(Clone, Debug, PartialEq)]
pub enum MergeReason {
    /// Too many segments in a tier
    TierOverflow,
    /// High delete ratio
    HighDeleteRatio,
    /// Small segments should be combined
    SmallSegments,
    /// Forced merge (e.g., for optimization)
    Forced,
}

/// Tiered merge policy implementation
pub struct TieredMergePolicy {
    config: MergePolicyConfig,
}

impl TieredMergePolicy {
    pub fn new(config: MergePolicyConfig) -> Self {
        Self { config }
    }

    /// Find merge candidates from a list of segments
    pub fn find_merges(&self, segments: &[Arc<SegmentReader>]) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();

        if segments.len() < self.config.min_merge_count {
            return candidates;
        }

        // Check for high-delete segments first (priority)
        let high_delete_candidate = self.find_high_delete_merge(segments);
        if let Some(candidate) = high_delete_candidate {
            candidates.push(candidate);
        }

        // Find tiered merges
        let tiered_candidates = self.find_tiered_merges(segments);
        candidates.extend(tiered_candidates);

        // Sort by score (highest first)
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Find segments with high delete ratios that should be merged
    fn find_high_delete_merge(&self, segments: &[Arc<SegmentReader>]) -> Option<MergeCandidate> {
        let high_delete_segments: Vec<_> = segments
            .iter()
            .filter(|s| s.delete_ratio() > self.config.delete_ratio_threshold)
            .cloned()
            .collect();

        if high_delete_segments.len() >= self.config.min_merge_count {
            let segment_ids: Vec<_> = high_delete_segments.iter().map(|s| s.id()).collect();
            let estimated_size: u64 = high_delete_segments
                .iter()
                .map(|s| s.meta().size_bytes)
                .sum();
            let avg_delete_ratio: f64 = high_delete_segments
                .iter()
                .map(|s| s.delete_ratio())
                .sum::<f64>()
                / high_delete_segments.len() as f64;

            Some(MergeCandidate {
                segment_ids,
                estimated_size,
                score: avg_delete_ratio * 100.0, // High priority for deletes
                reason: MergeReason::HighDeleteRatio,
            })
        } else {
            None
        }
    }

    /// Find tiered merge candidates
    fn find_tiered_merges(&self, segments: &[Arc<SegmentReader>]) -> Vec<MergeCandidate> {
        let mut candidates = Vec::new();

        // Group segments by size tier
        let tiers = self.group_by_tier(segments);

        for (tier_idx, tier_segments) in tiers.iter().enumerate() {
            if tier_segments.len() > self.config.segments_per_tier {
                // Need to merge some segments in this tier
                let merge_count = (tier_segments.len() - self.config.segments_per_tier + 1)
                    .min(self.config.max_merge_count)
                    .max(self.config.min_merge_count);

                // Select smallest segments in the tier
                let mut sorted_segments = tier_segments.clone();
                sorted_segments.sort_by_key(|s| s.meta().size_bytes);

                let to_merge: Vec<_> = sorted_segments
                    .into_iter()
                    .take(merge_count)
                    .collect();

                if to_merge.len() >= self.config.min_merge_count {
                    let segment_ids: Vec<_> = to_merge.iter().map(|s| s.id()).collect();
                    let estimated_size: u64 = to_merge.iter().map(|s| s.meta().size_bytes).sum();

                    // Score based on tier (lower tier = more urgent) and segment count
                    let score = (10.0 - tier_idx as f64).max(1.0) * to_merge.len() as f64;

                    candidates.push(MergeCandidate {
                        segment_ids,
                        estimated_size,
                        score,
                        reason: MergeReason::TierOverflow,
                    });
                }
            }
        }

        candidates
    }

    /// Group segments into tiers by size
    fn group_by_tier(&self, segments: &[Arc<SegmentReader>]) -> Vec<Vec<Arc<SegmentReader>>> {
        // Calculate tier boundaries
        // Tier 0: floor_segment_bytes to floor * segments_per_tier
        // Tier 1: floor * segments_per_tier to floor * segments_per_tier^2
        // etc.

        let floor = self.config.floor_segment_bytes;
        let ratio = self.config.segments_per_tier as u64;

        let max_tier = 10; // Reasonable maximum
        let mut tiers: Vec<Vec<Arc<SegmentReader>>> = vec![Vec::new(); max_tier];

        for segment in segments {
            let size = segment.meta().size_bytes.max(floor);
            let tier = self.size_to_tier(size, floor, ratio).min(max_tier - 1);
            tiers[tier].push(segment.clone());
        }

        // Remove empty tiers from the end
        while tiers.last().map(|t| t.is_empty()).unwrap_or(false) {
            tiers.pop();
        }

        tiers
    }

    /// Calculate which tier a segment belongs to based on size
    fn size_to_tier(&self, size: u64, floor: u64, ratio: u64) -> usize {
        if size <= floor {
            return 0;
        }

        let mut tier_max = floor * ratio;
        let mut tier = 0;

        while size > tier_max && tier < 10 {
            tier += 1;
            tier_max *= ratio;
        }

        tier
    }

    /// Check if a merge would exceed the maximum segment size
    pub fn would_exceed_max_size(&self, segments: &[Arc<SegmentReader>]) -> bool {
        let total_size: u64 = segments.iter().map(|s| s.meta().size_bytes).sum();
        total_size > self.config.max_merged_segment_bytes
    }

    /// Calculate merge score for a set of segments
    pub fn calculate_merge_score(&self, segments: &[Arc<SegmentReader>]) -> f64 {
        if segments.is_empty() {
            return 0.0;
        }

        let total_size: u64 = segments.iter().map(|s| s.meta().size_bytes).sum();
        let avg_delete_ratio: f64 = segments.iter().map(|s| s.delete_ratio()).sum::<f64>()
            / segments.len() as f64;

        // Score components:
        // 1. Delete ratio (0-100 points)
        // 2. Segment count reduction benefit (0-50 points)
        // 3. Size efficiency (0-50 points, smaller is better for merging)

        let delete_score = avg_delete_ratio * 100.0;
        let count_score = (segments.len() as f64 - 1.0) * 10.0; // Benefit of reducing N to 1
        let size_score = if total_size < self.config.max_merged_segment_bytes {
            50.0 * (1.0 - total_size as f64 / self.config.max_merged_segment_bytes as f64)
        } else {
            0.0
        };

        delete_score + count_score + size_score
    }
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        Self::new(MergePolicyConfig::default())
    }
}

/// Simple merge scheduler that tracks pending and running merges
pub struct MergeScheduler {
    /// Pending merge candidates
    pending: Vec<MergeCandidate>,
    /// Currently running merges (segment IDs being merged)
    running: Vec<Vec<SegmentId>>,
    /// Maximum concurrent merges
    max_concurrent: usize,
}

impl MergeScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Vec::new(),
            running: Vec::new(),
            max_concurrent,
        }
    }

    /// Add merge candidates
    pub fn add_candidates(&mut self, candidates: Vec<MergeCandidate>) {
        for candidate in candidates {
            // Don't add if any segment is already being merged
            let overlaps = self.running.iter().any(|running| {
                candidate.segment_ids.iter().any(|id| running.contains(id))
            });

            if !overlaps {
                self.pending.push(candidate);
            }
        }

        // Sort by score
        self.pending.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the next merge to execute (if any)
    pub fn next_merge(&mut self) -> Option<MergeCandidate> {
        if self.running.len() >= self.max_concurrent {
            return None;
        }

        // Find first candidate that doesn't overlap with running merges
        let idx = self.pending.iter().position(|candidate| {
            !self.running.iter().any(|running| {
                candidate.segment_ids.iter().any(|id| running.contains(id))
            })
        })?;

        let candidate = self.pending.remove(idx);
        self.running.push(candidate.segment_ids.clone());
        Some(candidate)
    }

    /// Mark a merge as complete
    pub fn complete_merge(&mut self, segment_ids: &[SegmentId]) {
        self.running.retain(|running| running != segment_ids);
    }

    /// Check if any merges are pending
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Check if any merges are running
    pub fn has_running(&self) -> bool {
        !self.running.is_empty()
    }

    /// Get number of running merges
    pub fn running_count(&self) -> usize {
        self.running.len()
    }
}

impl Default for MergeScheduler {
    fn default() -> Self {
        Self::new(2) // Default to 2 concurrent merges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{
        DocNoMap, DocValuesReader, PostingsReader, SegmentStatistics, TermDictionaryBuilder,
    };

    fn create_test_segment(id: u64, size_bytes: u64, delete_ratio: f64) -> Arc<SegmentReader> {
        let term_dict = TermDictionaryBuilder::new().build().unwrap();
        let doc_count = 100u32;
        let live_doc_count = (doc_count as f64 * (1.0 - delete_ratio)) as u32;

        let meta = SegmentMeta {
            id: SegmentId::new(id),
            min_raft_index: 0,
            max_raft_index: 100,
            doc_count,
            live_doc_count,
            size_bytes,
            created_at: 0,
        };

        Arc::new(SegmentReader::from_memory(
            meta,
            term_dict,
            PostingsReader::new(Vec::new()),
            DocValuesReader::new(),
            SegmentStatistics::new(),
            DocNoMap::new(),
        ))
    }

    #[test]
    fn test_tier_calculation() {
        let policy = TieredMergePolicy::default();

        // Floor is 1MB, ratio is 10
        let floor = 1024 * 1024;
        let ratio = 10;

        assert_eq!(policy.size_to_tier(floor / 2, floor, ratio), 0);
        assert_eq!(policy.size_to_tier(floor, floor, ratio), 0);
        assert_eq!(policy.size_to_tier(floor * 5, floor, ratio), 0);
        assert_eq!(policy.size_to_tier(floor * 10, floor, ratio), 0);
        assert_eq!(policy.size_to_tier(floor * 11, floor, ratio), 1);
        assert_eq!(policy.size_to_tier(floor * 100, floor, ratio), 1);
        assert_eq!(policy.size_to_tier(floor * 101, floor, ratio), 2);
    }

    #[test]
    fn test_high_delete_merge() {
        let policy = TieredMergePolicy::new(MergePolicyConfig {
            delete_ratio_threshold: 0.10,
            min_merge_count: 2,
            ..Default::default()
        });

        let segments = vec![
            create_test_segment(1, 1024 * 1024, 0.20), // 20% deleted
            create_test_segment(2, 1024 * 1024, 0.15), // 15% deleted
            create_test_segment(3, 1024 * 1024, 0.05), // 5% deleted
        ];

        let candidates = policy.find_merges(&segments);

        assert!(!candidates.is_empty());
        let high_delete = candidates
            .iter()
            .find(|c| c.reason == MergeReason::HighDeleteRatio);
        assert!(high_delete.is_some());

        let candidate = high_delete.unwrap();
        assert_eq!(candidate.segment_ids.len(), 2); // segments 1 and 2
    }

    #[test]
    fn test_tiered_merge() {
        let policy = TieredMergePolicy::new(MergePolicyConfig {
            segments_per_tier: 3,
            min_merge_count: 2,
            floor_segment_bytes: 1024,
            ..Default::default()
        });

        // Create 5 small segments (should overflow tier 0)
        let segments: Vec<_> = (0..5)
            .map(|i| create_test_segment(i, 1024 * 2, 0.0))
            .collect();

        let candidates = policy.find_merges(&segments);

        assert!(!candidates.is_empty());
        let tiered = candidates
            .iter()
            .find(|c| c.reason == MergeReason::TierOverflow);
        assert!(tiered.is_some());
    }

    #[test]
    fn test_merge_scheduler() {
        let mut scheduler = MergeScheduler::new(2);

        let candidate1 = MergeCandidate {
            segment_ids: vec![SegmentId::new(1), SegmentId::new(2)],
            estimated_size: 1000,
            score: 50.0,
            reason: MergeReason::TierOverflow,
        };

        let candidate2 = MergeCandidate {
            segment_ids: vec![SegmentId::new(3), SegmentId::new(4)],
            estimated_size: 2000,
            score: 30.0,
            reason: MergeReason::TierOverflow,
        };

        scheduler.add_candidates(vec![candidate1.clone(), candidate2.clone()]);

        assert!(scheduler.has_pending());

        let merge1 = scheduler.next_merge();
        assert!(merge1.is_some());
        assert_eq!(scheduler.running_count(), 1);

        let merge2 = scheduler.next_merge();
        assert!(merge2.is_some());
        assert_eq!(scheduler.running_count(), 2);

        // Should not get another merge (at max concurrent)
        assert!(scheduler.next_merge().is_none());

        // Complete one
        scheduler.complete_merge(&candidate1.segment_ids);
        assert_eq!(scheduler.running_count(), 1);
    }

    #[test]
    fn test_overlapping_merges() {
        let mut scheduler = MergeScheduler::new(2);

        let candidate1 = MergeCandidate {
            segment_ids: vec![SegmentId::new(1), SegmentId::new(2)],
            estimated_size: 1000,
            score: 50.0,
            reason: MergeReason::TierOverflow,
        };

        // Overlaps with candidate1 (segment 2)
        let candidate2 = MergeCandidate {
            segment_ids: vec![SegmentId::new(2), SegmentId::new(3)],
            estimated_size: 2000,
            score: 30.0,
            reason: MergeReason::TierOverflow,
        };

        scheduler.add_candidates(vec![candidate1.clone()]);

        let merge1 = scheduler.next_merge();
        assert!(merge1.is_some());

        // Try to add overlapping candidate - should be rejected
        scheduler.add_candidates(vec![candidate2]);

        // Should not get another merge (overlapping)
        assert!(scheduler.next_merge().is_none());
    }
}
