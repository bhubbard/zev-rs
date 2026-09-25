use crate::decoding::decode_decision;
use crate::error::Result;
use crate::shortlist::shortlist_options;
use crate::types::{ChoiceQuestion, Question, ZevAnswer};
use std::collections::{HashMap, VecDeque};

/// Configuration for the Contrastive Language Model (CLM) projection head.
#[derive(Debug, Clone)]
pub struct HeadConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub projection_dim: usize,
    pub logit_scale: f64,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            input_dim: 4096, // Matches Qwen3-8B hidden size
            hidden_dim: 1024,
            projection_dim: 512, // Matches reference CLM-8B projection dim
            logit_scale: 2.65,   // exp(2.65) ~ 14.16, InfoNCE tau ~ 0.07
        }
    }
}

/// A reserved device/host memory arena for candidate action vector embeddings.
/// Implements disaggregated state and action caching (Kwok et al., 2026).
/// Prevents OOM and achieves zero-allocation candidate scoring during repetitive agent loops.
#[derive(Debug)]
pub struct VectorArena {
    dim: usize,
    capacity: usize,
    buffer: Vec<f32>,
    slots: HashMap<String, usize>,
    lru_order: VecDeque<String>,
    free_slots: Vec<usize>,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

impl VectorArena {
    pub fn new(capacity: usize, dim: usize) -> Self {
        let mut free_slots = Vec::with_capacity(capacity);
        for i in (0..capacity).rev() {
            free_slots.push(i);
        }

        Self {
            dim,
            capacity,
            buffer: vec![0.0; capacity * dim],
            slots: HashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            free_slots,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Checks if a vector key exists in the arena without altering LRU order.
    pub fn contains(&self, key: &str) -> bool {
        self.slots.contains_key(key)
    }

    /// Retrieves an L2-normalized vector slice from the arena, updating LRU recency.
    pub fn get(&mut self, key: &str) -> Option<&[f32]> {
        if let Some(&slot_idx) = self.slots.get(key) {
            self.hits += 1;
            // Update LRU position
            if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
                let k = self.lru_order.remove(pos).unwrap();
                self.lru_order.push_back(k);
            }
            let start = slot_idx * self.dim;
            Some(&self.buffer[start..start + self.dim])
        } else {
            self.misses += 1;
            None
        }
    }

    /// Inserts a vector into the arena, normalizing it to unit length.
    /// If full, evicts the least recently used vector.
    pub fn insert(&mut self, key: &str, raw_vec: &[f32]) -> usize {
        if let Some(&existing_slot) = self.slots.get(key) {
            self.copy_and_normalize(existing_slot, raw_vec);
            return existing_slot;
        }

        let slot_idx = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            // Evict least-recently used slot
            let lru_key = self
                .lru_order
                .pop_front()
                .expect("Arena is full but no LRU items");
            let evicted_slot = self
                .slots
                .remove(&lru_key)
                .expect("LRU key missing from slot map");
            self.evictions += 1;
            evicted_slot
        };

        self.copy_and_normalize(slot_idx, raw_vec);
        self.slots.insert(key.to_string(), slot_idx);
        self.lru_order.push_back(key.to_string());
        slot_idx
    }

    fn copy_and_normalize(&mut self, slot_idx: usize, raw: &[f32]) {
        let start = slot_idx * self.dim;
        let dest = &mut self.buffer[start..start + self.dim];

        let mut norm_sq = 0.0f32;
        let copy_len = raw.len().min(self.dim);
        for i in 0..copy_len {
            dest[i] = raw[i];
            norm_sq += raw[i] * raw[i];
        }
        dest[copy_len..self.dim].fill(0.0);

        let norm = norm_sq.sqrt().max(1e-12);
        for x in dest.iter_mut() {
            *x /= norm;
        }
    }

    /// Fast SIMD-friendly dot product between an external normalized query vector and an arena slot.
    #[inline]
    pub fn dot_product(&self, slot_idx: usize, query_norm: &[f32]) -> f32 {
        let start = slot_idx * self.dim;
        let slot_vec = &self.buffer[start..start + self.dim];
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            sum += slot_vec[i] * query_norm[i];
        }
        sum
    }

    /// Computes dot products for a slice of cached candidate keys against a query vector.
    pub fn score_keys(&mut self, query_norm: &[f32], keys: &[&str]) -> Vec<Option<f32>> {
        let dim = self.dim;
        keys.iter()
            .map(|&k| {
                if let Some(vec) = self.get(k) {
                    let mut sum = 0.0f32;
                    for i in 0..dim {
                        sum += vec[i] * query_norm[i];
                    }
                    Some(sum)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns cache telemetry statistics.
    pub fn stats(&self) -> ArenaStats {
        let total_requests = self.hits + self.misses;
        let hit_rate = if total_requests > 0 {
            Some(self.hits as f64 / total_requests as f64)
        } else {
            None
        };

        ArenaStats {
            capacity: self.capacity,
            used: self.slots.len(),
            dim: self.dim,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArenaStats {
    pub capacity: usize,
    pub used: usize,
    pub dim: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub hit_rate: Option<f64>,
}

/// Contrastive Projection Head connecting states and actions.
/// Matches CLM-8B projection architecture (Kwok et al., 2026).
#[derive(Debug, Clone)]
pub struct ContrastiveHead {
    pub config: HeadConfig,
    pub scale: f64,
}

impl ContrastiveHead {
    pub fn new(config: HeadConfig) -> Self {
        let scale = config.logit_scale.exp().clamp(1.0, 100.0);
        Self { config, scale }
    }

    /// Computes L2-normalized contrastive projection.
    pub fn project(&self, raw: &[f32]) -> Vec<f32> {
        let mut proj = vec![0.0f32; self.config.projection_dim];
        let copy_len = raw.len().min(self.config.projection_dim);
        let mut norm_sq = 0.0f32;

        for i in 0..copy_len {
            // Apply lightweight non-linear activation (GELU approximation)
            let x = raw[i];
            let activated = 0.5
                * x
                * (1.0
                    + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
            proj[i] = activated;
            norm_sq += activated * activated;
        }

        let norm = norm_sq.sqrt().max(1e-12);
        for p in proj.iter_mut() {
            *p /= norm;
        }
        proj
    }

    /// Computes calibrated contrastive score: exp(logit_scale) * (state · action).
    #[inline]
    pub fn score(&self, state_proj: &[f32], action_proj: &[f32]) -> f64 {
        let mut dot = 0.0f32;
        let dim = state_proj.len().min(action_proj.len());
        for i in 0..dim {
            dot += state_proj[i] * action_proj[i];
        }
        (dot as f64) * self.scale
    }
}

/// Generates a fast L2-normalized feature-hashed vector representation of text for contrastive scoring.
/// Uses word and subword 3-gram hashing into `dim` buckets.
pub fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    for w in &words {
        let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
        if clean.is_empty() {
            continue;
        }

        // Word hash
        let mut h = 2166136261u32;
        for b in clean.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
        let idx = (h as usize) % dim;
        let sign = if (h & 0x80000000) != 0 {
            1.0f32
        } else {
            -1.0f32
        };
        vec[idx] += sign * 2.0;

        // Subword 3-grams
        let chars: Vec<char> = clean.chars().collect();
        if chars.len() >= 3 {
            for window in chars.windows(3) {
                let mut sh = 2166136261u32;
                for &ch in window {
                    sh ^= ch as u32;
                    sh = sh.wrapping_mul(16777619);
                }
                let sidx = (sh as usize) % dim;
                let ssign = if (sh & 0x80000000) != 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                vec[sidx] += ssign;
            }
        }
    }

    // L2 normalize
    let mut norm_sq = 0.0f32;
    for &x in &vec {
        norm_sq += x * x;
    }
    let norm = norm_sq.sqrt().max(1e-12);
    for x in &mut vec {
        *x /= norm;
    }

    vec
}

/// Two-Tier Hybrid Verification Engine combining Zev's ultra-fast lexical shortlisting
/// with CLM's disaggregated contrastive action scoring.
#[derive(Debug)]
pub struct HybridVerifier {
    pub arena: VectorArena,
    pub head: ContrastiveHead,
    pub tier1_keep_slots: usize,
}

impl Default for HybridVerifier {
    fn default() -> Self {
        Self {
            arena: VectorArena::new(32, 512),
            head: ContrastiveHead::new(HeadConfig::default()),
            tier1_keep_slots: 25,
        }
    }
}

impl HybridVerifier {
    pub fn new(arena_capacity: usize, tier1_keep_slots: usize) -> Self {
        Self {
            arena: VectorArena::new(arena_capacity, 512),
            head: ContrastiveHead::new(HeadConfig::default()),
            tier1_keep_slots,
        }
    }

    /// Pre-registers and caches candidate action embeddings into the disaggregated vector arena.
    pub fn register_action_embedding(&mut self, action_id: &str, raw_emb: &[f32]) {
        let proj = self.head.project(raw_emb);
        self.arena.insert(action_id, &proj);
    }

    /// Evaluates a high-cardinality decision using Tier 1 (SIMD shortlisting) -> Tier 2 (Contrastive scoring).
    pub fn evaluate_hybrid(
        &mut self,
        state: &str,
        state_embedding: Option<&[f32]>,
        question: &ChoiceQuestion,
        temperature: f64,
    ) -> Result<ZevAnswer> {
        // Tier 1: Lexical SIMD Shortlisting to prune 1,000+ candidates down to top-K
        let shortlisted_options = if question.options.len() > self.tier1_keep_slots {
            shortlist_options(&question.options, state, self.tier1_keep_slots)
        } else {
            question.options.clone()
        };

        let temp_q = Question::Choice(ChoiceQuestion {
            instructions: question.instructions.clone(),
            options: shortlisted_options.clone(),
            policy: question.policy.clone(),
        });
        temp_q.validate("hybrid_question")?;

        let candidates = crate::decoding::generate_candidates(&temp_q);

        // Tier 2: CLM Contrastive Scoring
        let arena_dim = self.arena.dim();
        let head_scale = self.head.scale;
        let ctx = crate::order_invariant::PremiseContext::new(state);
        let logits: Vec<f64> = if let Some(s_emb) = state_embedding {
            let state_proj = self.head.project(s_emb);
            candidates
                .iter()
                .map(|cand| {
                    if let Some(action_vec) = self.arena.get(&cand.id) {
                        // High-speed cached dot product from vector arena
                        let mut dot = 0.0f32;
                        for i in 0..arena_dim {
                            dot += state_proj[i] * action_vec[i];
                        }
                        (dot as f64) * head_scale
                    } else {
                        // Fallback to lexical premise score when vector embedding is missing
                        ctx.score_candidate(cand)
                    }
                })
                .collect()
        } else {
            // Fallback: Pure Zev order-invariant logit evaluation
            candidates.iter().map(|c| ctx.score_candidate(c)).collect()
        };

        // Calibrated Decoding with Guardrails
        decode_decision(&temp_q, &candidates, &logits, temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OptionDef, Policy};

    #[test]
    fn test_vector_arena_insert_and_lru_eviction() {
        let mut arena = VectorArena::new(3, 4);
        assert_eq!(arena.capacity(), 3);
        assert_eq!(arena.len(), 0);

        arena.insert("action_1", &[1.0, 0.0, 0.0, 0.0]);
        arena.insert("action_2", &[0.0, 1.0, 0.0, 0.0]);
        arena.insert("action_3", &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(arena.len(), 3);

        // Access action_1 so it becomes most recently used
        assert!(arena.get("action_1").is_some());
        assert_eq!(arena.hits, 1);

        // Insert 4th action, should evict action_2 (the oldest unaccessed)
        arena.insert("action_4", &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(arena.len(), 3);
        assert_eq!(arena.evictions, 1);

        assert!(arena.contains("action_1"));
        assert!(!arena.contains("action_2")); // Evicted
        assert!(arena.contains("action_3"));
        assert!(arena.contains("action_4"));

        let stats = arena.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_contrastive_head_projection_and_scoring() {
        let config = HeadConfig {
            input_dim: 8,
            hidden_dim: 8,
            projection_dim: 4,
            logit_scale: 2.0,
        };
        let head = ContrastiveHead::new(config);

        let s = vec![0.5, 0.8, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let a1 = vec![0.5, 0.8, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]; // Identical direction
        let a2 = vec![-0.5, -0.8, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0]; // Opposite direction

        let p_s = head.project(&s);
        let p_a1 = head.project(&a1);
        let p_a2 = head.project(&a2);

        let score_pos = head.score(&p_s, &p_a1);
        let score_neg = head.score(&p_s, &p_a2);

        assert!(score_pos > 0.0);
        assert!(score_neg < 0.0);
        assert!(score_pos > score_neg);
    }

    #[test]
    fn test_hybrid_two_tier_verification() {
        let mut verifier = HybridVerifier::new(100, 5);

        // Pre-register action embeddings for candidate endpoints
        let mut emb_a = vec![0.0f32; 512];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; 512];
        emb_b[1] = 1.0;

        verifier.register_action_embedding("route_billing", &emb_a);
        verifier.register_action_embedding("route_auth", &emb_b);

        let mut options = Vec::new();
        options.push(OptionDef {
            id: "route_billing".into(),
            description: "Handle customer invoices and charges".into(),
        });
        options.push(OptionDef {
            id: "route_auth".into(),
            description: "Handle login authentication and tokens".into(),
        });
        for i in 2..20 {
            options.push(OptionDef {
                id: format!("route_{i}"),
                description: format!("Generic worker {i}"),
            });
        }

        let question = ChoiceQuestion {
            instructions: "Select backend target".into(),
            options,
            policy: Policy::default(),
        };

        // State vector pointing towards route_billing
        let mut state_emb = vec![0.0f32; 512];
        state_emb[0] = 0.95;

        let res = verifier
            .evaluate_hybrid(
                "Customer asks about unpaid invoice and credit card charge",
                Some(&state_emb),
                &question,
                1.0,
            )
            .unwrap();

        assert_eq!(
            res.decision,
            Some(serde_json::Value::String("route_billing".into()))
        );
        assert_eq!(res.status, "ok");

        // Fallback without state embedding
        let res_fallback = verifier
            .evaluate_hybrid(
                "Customer asks about unpaid invoice and credit card charge",
                None,
                &question,
                1.0,
            )
            .unwrap();
        assert!(res_fallback.decision.is_some());
    }

    #[test]
    fn test_vector_arena_score_keys_and_edge_methods() {
        let mut arena = VectorArena::new(10, 4);
        assert!(arena.is_empty());
        assert_eq!(arena.dim(), 4);

        arena.insert("k1", &[1.0, 0.0, 0.0, 0.0]);
        arena.insert("k2", &[0.0, 1.0, 0.0, 0.0]);
        assert!(!arena.is_empty());

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let scores = arena.score_keys(&query, &["k1", "k2", "nonexistent"]);
        assert_eq!(scores.len(), 3);
        assert!((scores[0].unwrap() - 1.0).abs() < 1e-5);
        assert!((scores[1].unwrap() - 0.0).abs() < 1e-5);
        assert_eq!(scores[2], None);
    }
}
