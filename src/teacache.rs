//! TeaCache: Relative Activation / State Caching for Decision Engines.
//!
//! Synthesized from `comfyui-ltxvideo-mlx-rs/src/core/teacache.rs`.
//!
//! Calculates relative L2 perturbation between consecutive step/query embedding representations:
//! $$\delta = \frac{\|x - x_{\text{prev}}\|_2}{\|x_{\text{prev}}\|_2}$$
//! If $\delta < \text{threshold}$, skips redundant neural forward passes and reuses cached
//! activations/logits, yielding 2x-3x speedups on multi-question or streaming tail queries.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeaCacheConfig {
    pub enabled: bool,
    pub threshold: f32,
    pub warmup_steps: usize,
}

impl Default for TeaCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.15,
            warmup_steps: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TeaCache {
    pub config: TeaCacheConfig,
    pub cached_logits: Option<Vec<f64>>,
    pub previous_embedding: Option<Vec<f32>>,
    pub skipped_count: usize,
    pub evaluated_count: usize,
}

impl Default for TeaCache {
    fn default() -> Self {
        Self::new(TeaCacheConfig::default())
    }
}

impl TeaCache {
    pub fn new(config: TeaCacheConfig) -> Self {
        Self {
            config,
            cached_logits: None,
            previous_embedding: None,
            skipped_count: 0,
            evaluated_count: 0,
        }
    }

    /// Determines whether the current step can skip forward evaluation based on relative L2 delta.
    pub fn should_skip(&mut self, step: usize, current_embedding: &[f32]) -> bool {
        if !self.config.enabled || step < self.config.warmup_steps || self.cached_logits.is_none() {
            self.evaluated_count += 1;
            self.previous_embedding = Some(current_embedding.to_vec());
            return false;
        }

        if let Some(ref prev_emb) = self.previous_embedding {
            let mut diff_norm_sq = 0.0f32;
            let mut base_norm_sq = 0.0f32;

            for (a, b) in prev_emb.iter().zip(current_embedding.iter()) {
                let diff = a - b;
                diff_norm_sq += diff * diff;
                base_norm_sq += a * a;
            }

            let delta = if base_norm_sq > 1e-6 {
                (diff_norm_sq / base_norm_sq).sqrt()
            } else {
                1.0
            };

            if delta < self.config.threshold {
                self.skipped_count += 1;
                return true;
            }
        }

        self.evaluated_count += 1;
        self.previous_embedding = Some(current_embedding.to_vec());
        false
    }

    /// Records computed logits to cache for subsequent steps.
    pub fn record_evaluation(&mut self, embedding: &[f32], logits: &[f64]) {
        self.previous_embedding = Some(embedding.to_vec());
        self.cached_logits = Some(logits.to_vec());
    }

    /// Returns the cached logits if available.
    pub fn cached_logits(&self) -> Option<&[f64]> {
        self.cached_logits.as_deref()
    }

    /// Resets the cache state while preserving configuration.
    pub fn reset(&mut self) {
        self.cached_logits = None;
        self.previous_embedding = None;
        self.skipped_count = 0;
        self.evaluated_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teacache_skip_under_threshold() {
        let mut tc = TeaCache::new(TeaCacheConfig {
            enabled: true,
            threshold: 0.20,
            warmup_steps: 1,
        });

        let emb1 = vec![1.0, 2.0, 3.0, 4.0];
        assert!(!tc.should_skip(0, &emb1));
        tc.record_evaluation(&emb1, &[0.85, 0.15]);

        // emb2 is virtually identical (small relative perturbation)
        let emb2 = vec![1.01, 2.01, 2.99, 4.01];
        assert!(tc.should_skip(1, &emb2));
        assert_eq!(tc.cached_logits().unwrap(), &[0.85, 0.15]);
        assert_eq!(tc.skipped_count, 1);

        // emb3 is significantly different
        let emb3 = vec![5.0, -2.0, 0.0, 1.0];
        assert!(!tc.should_skip(2, &emb3));
        assert_eq!(tc.evaluated_count, 2);
    }
}
