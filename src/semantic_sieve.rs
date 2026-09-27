//! Semantic sieve for zev-rs leveraging dense vector representations and MLX embeddings.
//!
//! Bridges the gap between sub-6µs SIMD lexical heuristics and 150ms full LLM generation,
//! allowing high-confidence semantic routing in sub-millisecond execution time.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SieveResult {
    pub candidate_id: String,
    pub top_score: f64,
    pub margin: f64,
    pub scores: BTreeMap<String, f64>,
    pub decisive: bool,
}

#[derive(Debug, Clone)]
pub struct CandidateVector {
    pub id: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SemanticSieve {
    pub candidates: Vec<CandidateVector>,
    pub margin_threshold: f64,
    pub min_confidence: f64,
}

impl SemanticSieve {
    pub fn new(margin_threshold: f64, min_confidence: f64) -> Self {
        Self {
            candidates: Vec::new(),
            margin_threshold,
            min_confidence,
        }
    }

    /// Add a candidate with a pre-normalized dense vector
    pub fn add_candidate(&mut self, id: impl Into<String>, mut vector: Vec<f32>) {
        Self::normalize_l2(&mut vector);
        self.candidates.push(CandidateVector {
            id: id.into(),
            vector,
        });
    }

    /// L2 normalize a slice of floats in-place
    pub fn normalize_l2(v: &mut [f32]) {
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-9 {
            let inv = 1.0 / norm;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    /// Dot product between two normalized vectors (cosine similarity)
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
        }
        dot
    }

    /// Evaluates a query embedding vector against all candidates
    pub fn evaluate_vector(&self, query: &[f32]) -> Option<SieveResult> {
        if self.candidates.is_empty() {
            return None;
        }

        let mut norm_query = query.to_vec();
        Self::normalize_l2(&mut norm_query);

        let mut scores: Vec<(String, f64)> = self
            .candidates
            .iter()
            .map(|c| {
                let score = Self::cosine_similarity(&norm_query, &c.vector) as f64;
                (c.id.clone(), score)
            })
            .collect();

        // Sort descending by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top1 = scores[0].clone();
        let top2_score = if scores.len() > 1 {
            scores[1].1
        } else {
            0.0
        };
        let margin = top1.1 - top2_score;

        let decisive = margin >= self.margin_threshold && top1.1 >= self.min_confidence;

        let score_map: BTreeMap<String, f64> = scores.into_iter().collect();

        Some(SieveResult {
            candidate_id: top1.0,
            top_score: top1.1,
            margin,
            scores: score_map,
            decisive,
        })
    }

    /// Generate lightweight deterministic pseudo-embedding from text n-grams
    /// for fast testing and zero-weight fallback environments
    pub fn hash_embed(text: &str, dims: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; dims];
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for (w_idx, word) in words.iter().enumerate() {
            let mut h = 2166136261u32;
            for b in word.bytes() {
                h ^= b as u32;
                h = h.wrapping_mul(16777619);
            }
            let idx = (h as usize) % dims;
            let weight = 1.0 / (1.0 + (w_idx as f32 * 0.05));
            vec[idx] += weight;

            // Character tri-grams
            if word.len() >= 3 {
                for window in word.as_bytes().windows(3) {
                    let mut tri_h = 2166136261u32;
                    for &b in window {
                        tri_h ^= b as u32;
                        tri_h = tri_h.wrapping_mul(16777619);
                    }
                    let tri_idx = (tri_h as usize) % dims;
                    vec[tri_idx] += 0.5 * weight;
                }
            }
        }

        Self::normalize_l2(&mut vec);
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_sieve_decisive_routing() {
        let mut sieve = SemanticSieve::new(0.20, 0.40);

        // Pre-embed candidates into 64-dim vector space
        let billing_vec = SemanticSieve::hash_embed("invoice billing payment charge refund payment subscription", 64);
        let tech_vec = SemanticSieve::hash_embed("database server outage cluster connection error crash", 64);
        let sales_vec = SemanticSieve::hash_embed("enterprise contract discount pricing custom quote sales", 64);

        sieve.add_candidate("billing", billing_vec);
        sieve.add_candidate("tech", tech_vec);
        sieve.add_candidate("sales", sales_vec);

        // Test Query 1: Billing inquiry
        let query_billing = SemanticSieve::hash_embed("I have an unexpected charge on my subscription invoice", 64);
        let res1 = sieve.evaluate_vector(&query_billing).expect("Evaluation should succeed");

        assert_eq!(res1.candidate_id, "billing");
        assert!(res1.decisive);
        assert!(res1.margin >= 0.20);

        // Test Query 2: Technical outage
        let query_tech = SemanticSieve::hash_embed("Production database has connection error and server crashed", 64);
        let res2 = sieve.evaluate_vector(&query_tech).expect("Evaluation should succeed");

        assert_eq!(res2.candidate_id, "tech");
        assert!(res2.decisive);
    }
}
