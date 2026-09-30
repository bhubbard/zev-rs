//! Metal GPU-accelerated semantic sieve for zev-rs using mlx-rs.
//!
//! Provides ultra-low latency (<1ms) vector-based semantic routing and candidate
//! scoring on Apple Silicon unified memory.

use crate::error::{Result, ZevError};
use crate::semantic_sieve::SieveResult;
use mlx_rs::{ops, Array};
use std::collections::BTreeMap;

/// Metal GPU-accelerated semantic sieve using Apple MLX.
///
/// Stores candidate vectors in Apple Silicon Unified Memory and evaluates
/// single or batched queries using fused Metal matrix-multiplication kernels.
#[derive(Debug)]
pub struct MlxSemanticSieve {
    candidate_ids: Vec<String>,
    /// Pre-transposed candidate matrix of shape `[dim, num_candidates]`.
    /// Storing as `[dim, N]` allows direct `[B, dim] @ [dim, N] -> [B, N]` matmul
    /// with contiguous row-major memory for each query's candidate scores.
    candidates_t: Array,
    dim: usize,
    margin_threshold: f64,
    min_confidence: f64,
}

impl MlxSemanticSieve {
    /// Constructs a new `MlxSemanticSieve` from a list of candidate IDs and their embedding vectors.
    ///
    /// All vectors must have the same non-zero dimension. Each vector is normalized to unit L2 norm
    /// prior to loading into unified memory.
    pub fn from_candidates(
        candidates: &[(String, Vec<f32>)],
        margin_threshold: f64,
        min_confidence: f64,
    ) -> Result<Self> {
        if candidates.is_empty() {
            return Err(ZevError::InvalidRequest(
                "MlxSemanticSieve requires at least one candidate".to_string(),
            ));
        }

        let dim = candidates[0].1.len();
        if dim == 0 {
            return Err(ZevError::InvalidRequest(
                "Candidate embedding dimension cannot be 0".to_string(),
            ));
        }

        let n = candidates.len();
        let mut flat_data = Vec::with_capacity(n * dim);
        let mut candidate_ids = Vec::with_capacity(n);

        for (id, vec) in candidates {
            if vec.len() != dim {
                return Err(ZevError::InvalidRequest(format!(
                    "Candidate '{id}' dimension mismatch: expected {dim}, got {}",
                    vec.len()
                )));
            }

            candidate_ids.push(id.clone());

            // L2 normalize before loading into memory
            let mut norm_vec = vec.clone();
            normalize_l2(&mut norm_vec);
            flat_data.extend_from_slice(&norm_vec);
        }

        // Create [N, dim] candidate matrix in Unified Memory
        let c_matrix = Array::from_slice(&flat_data, &[n as i32, dim as i32]);

        // Pre-transpose to [dim, N] for fast [B, dim] x [dim, N] batch scoring
        let candidates_t = ops::transpose(&c_matrix)
            .map_err(|e| ZevError::Evaluation(format!("MLX transpose error: {e}")))?;
        candidates_t
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX eval error: {e}")))?;

        Ok(Self {
            candidate_ids,
            candidates_t,
            dim,
            margin_threshold,
            min_confidence,
        })
    }

    /// Embedding dimension of the sieve.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of candidate targets.
    pub fn candidate_count(&self) -> usize {
        self.candidate_ids.len()
    }

    /// Candidate IDs registered in the sieve.
    pub fn candidate_ids(&self) -> &[String] {
        &self.candidate_ids
    }

    /// Evaluates a single query embedding against all candidate routes on the Metal GPU.
    pub fn evaluate_vector(&self, query: &[f32]) -> Result<Option<SieveResult>> {
        crate::qos::elevate_thread_qos();
        if query.len() != self.dim {
            return Err(ZevError::InvalidRequest(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }

        let mut norm_query = query.to_vec();
        normalize_l2(&mut norm_query);

        // Query shape: [1, dim]
        let q_arr = Array::from_slice(&norm_query, &[1, self.dim as i32]);

        // [1, dim] @ [dim, N] -> [1, N]
        let scores_arr = ops::matmul(&q_arr, &self.candidates_t)
            .map_err(|e| ZevError::Evaluation(format!("MLX matmul error: {e}")))?;
        scores_arr
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX eval error: {e}")))?;

        let scores_slice = scores_arr
            .try_as_slice::<f32>()
            .map_err(|e| ZevError::Evaluation(format!("MLX as_slice error: {e}")))?;

        Ok(Some(self.build_result(scores_slice)))
    }

    /// Evaluates a batch of query embeddings in a single GPU dispatch.
    ///
    /// Computes `[B, dim] @ [dim, N] -> [B, N]` in parallel on Apple Silicon GPU cores.
    pub fn evaluate_batch(&self, queries: &[Vec<f32>]) -> Result<Vec<SieveResult>> {
        crate::qos::elevate_thread_qos();
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let b = queries.len();
        let mut flat_queries = Vec::with_capacity(b * self.dim);

        for (idx, q) in queries.iter().enumerate() {
            if q.len() != self.dim {
                return Err(ZevError::InvalidRequest(format!(
                    "Batch query at index {idx} dimension mismatch: expected {}, got {}",
                    self.dim,
                    q.len()
                )));
            }
            let mut norm_q = q.clone();
            normalize_l2(&mut norm_q);
            flat_queries.extend_from_slice(&norm_q);
        }

        // [B, dim] batch tensor
        let q_batch = Array::from_slice(&flat_queries, &[b as i32, self.dim as i32]);

        // [B, dim] @ [dim, N] -> [B, N]
        let scores_batch = ops::matmul(&q_batch, &self.candidates_t)
            .map_err(|e| ZevError::Evaluation(format!("MLX batch matmul error: {e}")))?;
        scores_batch
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX eval error: {e}")))?;

        let flat_scores = scores_batch
            .try_as_slice::<f32>()
            .map_err(|e| ZevError::Evaluation(format!("MLX as_slice error: {e}")))?;

        let n = self.candidate_ids.len();
        let mut results = Vec::with_capacity(b);

        for i in 0..b {
            let row_scores = &flat_scores[i * n..(i + 1) * n];
            results.push(self.build_result(row_scores));
        }

        Ok(results)
    }

    /// Helper to convert a contiguous candidate score slice into a structured `SieveResult`.
    fn build_result(&self, scores: &[f32]) -> SieveResult {
        let mut score_pairs: Vec<(String, f64)> = self
            .candidate_ids
            .iter()
            .zip(scores.iter())
            .map(|(id, &s)| (id.clone(), s as f64))
            .collect();

        // Sort descending by score
        score_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top1 = score_pairs[0].clone();
        let top2_score = if score_pairs.len() > 1 {
            score_pairs[1].1
        } else {
            0.0
        };
        let margin = top1.1 - top2_score;
        let decisive = margin >= self.margin_threshold && top1.1 >= self.min_confidence;

        let score_map: BTreeMap<String, f64> = score_pairs.into_iter().collect();

        SieveResult {
            candidate_id: top1.0,
            top_score: top1.1,
            margin,
            scores: score_map,
            decisive,
        }
    }
}

/// In-place L2 normalization.
#[inline]
fn normalize_l2(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-9 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
