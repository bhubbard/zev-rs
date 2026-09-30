//! GPU-accelerated embedding projection and dead-letter queue (DLQ) batch clustering using MLX.

use crate::error::{Result, ZevError};
use mlx_rs::{ops, Array};

/// GPU-accelerated projection head for state/action representations.
///
/// Projects raw representations through a linear layer and normalizes on Metal GPU cores.
#[derive(Debug)]
pub struct MlxProjectionHead {
    /// Projection weights [input_dim, output_dim]
    weights: Array,
    input_dim: usize,
    output_dim: usize,
}

impl MlxProjectionHead {
    /// Creates a projection head with deterministic pseudo-random orthogonal weights
    /// or predefined projection weights.
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(ZevError::InvalidRequest(
                "Dimensions must be greater than 0".to_string(),
            ));
        }

        // Initialize pseudo-random orthogonal weights
        let mut flat = Vec::with_capacity(input_dim * output_dim);
        for i in 0..input_dim {
            for j in 0..output_dim {
                let seed = ((i * 10007 + j * 37 + 1) as f32).sin();
                let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
                flat.push(seed * scale);
            }
        }

        let weights = Array::from_slice(&flat, &[input_dim as i32, output_dim as i32]);
        weights
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX weights eval error: {e}")))?;

        Ok(Self {
            weights,
            input_dim,
            output_dim,
        })
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Projects a single vector or batch of vectors on the Metal GPU:
    /// `[B, input_dim] @ [input_dim, output_dim] -> [B, output_dim]`.
    pub fn project_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let b = inputs.len();
        let mut flat_in = Vec::with_capacity(b * self.input_dim);

        for (idx, v) in inputs.iter().enumerate() {
            if v.len() != self.input_dim {
                return Err(ZevError::InvalidRequest(format!(
                    "Vector at index {idx} dimension mismatch: expected {}, got {}",
                    self.input_dim,
                    v.len()
                )));
            }
            flat_in.extend_from_slice(v);
        }

        let x = Array::from_slice(&flat_in, &[b as i32, self.input_dim as i32]);

        // Matmul: [B, input_dim] @ [input_dim, output_dim] -> [B, output_dim]
        let projected = ops::matmul(&x, &self.weights)
            .map_err(|e| ZevError::Evaluation(format!("MLX project matmul error: {e}")))?;
        projected
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX project eval error: {e}")))?;

        let slice = projected
            .try_as_slice::<f32>()
            .map_err(|e| ZevError::Evaluation(format!("MLX project as_slice error: {e}")))?;

        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let row = &slice[i * self.output_dim..(i + 1) * self.output_dim];
            let mut v = row.to_vec();
            // L2 normalize
            let norm_sq: f32 = v.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-9 {
                let inv = 1.0 / norm;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
            out.push(v);
        }

        Ok(out)
    }
}

/// Cluster assignment for Dead-Letter Queue (DLQ) batch categorization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DlqClusterSummary {
    pub cluster_id: usize,
    pub item_indices: Vec<usize>,
    pub representative_index: usize,
    pub size: usize,
}

/// GPU-accelerated batch clustering for Dead-Letter Queue (DLQ) failure triage.
pub struct MlxDlqClusterer;

impl MlxDlqClusterer {
    /// Clusters a batch of failure embeddings on Metal GPU using pairwise cosine similarity.
    ///
    /// Items with pairwise cosine similarity >= `similarity_threshold` are grouped into the same cluster.
    pub fn cluster_failures(
        embeddings: &[Vec<f32>],
        similarity_threshold: f32,
    ) -> Result<Vec<DlqClusterSummary>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        let b = embeddings.len();
        let dim = embeddings[0].len();
        if dim == 0 {
            return Err(ZevError::InvalidRequest(
                "Embedding dimension cannot be 0".to_string(),
            ));
        }

        let mut flat = Vec::with_capacity(b * dim);
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != dim {
                return Err(ZevError::InvalidRequest(format!(
                    "Embedding at index {i} dimension mismatch: expected {dim}, got {}",
                    emb.len()
                )));
            }
            let mut norm_v = emb.clone();
            let norm_sq: f32 = norm_v.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-9 {
                let inv = 1.0 / norm;
                for x in norm_v.iter_mut() {
                    *x *= inv;
                }
            }
            flat.extend_from_slice(&norm_v);
        }

        // [B, dim]
        let matrix = Array::from_slice(&flat, &[b as i32, dim as i32]);

        // Pairwise similarity: [B, dim] @ [dim, B] -> [B, B]
        let matrix_t = ops::transpose(&matrix)
            .map_err(|e| ZevError::Evaluation(format!("MLX transpose error: {e}")))?;
        let sim_matrix = ops::matmul(&matrix, &matrix_t)
            .map_err(|e| ZevError::Evaluation(format!("MLX pairwise matmul error: {e}")))?;
        sim_matrix
            .eval()
            .map_err(|e| ZevError::Evaluation(format!("MLX sim eval error: {e}")))?;

        let sim_slice = sim_matrix
            .try_as_slice::<f32>()
            .map_err(|e| ZevError::Evaluation(format!("MLX sim as_slice error: {e}")))?;

        // Greedy threshold clustering on the pairwise matrix
        let mut visited = vec![false; b];
        let mut clusters = Vec::new();
        let mut cluster_id = 0;

        for i in 0..b {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let mut members = vec![i];

            for j in (i + 1)..b {
                if visited[j] {
                    continue;
                }
                let sim = sim_slice[i * b + j];
                if sim >= similarity_threshold {
                    visited[j] = true;
                    members.push(j);
                }
            }

            clusters.push(DlqClusterSummary {
                cluster_id,
                item_indices: members.clone(),
                representative_index: i,
                size: members.len(),
            });
            cluster_id += 1;
        }

        // Sort clusters by size descending (largest failure clusters first)
        clusters.sort_by_key(|a| std::cmp::Reverse(a.size));

        Ok(clusters)
    }
}
