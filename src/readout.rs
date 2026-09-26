use crate::error::{Result, ZevError};

/// Common token spelling variants for boolean/affirmative and negative decision classes.
pub const DEFAULT_TRUE_SPELLINGS: &[&str] = &[
    "true", "True", "TRUE", " true", " True",
    "yes", "Yes", "YES", " yes", " Yes",
    "1", " 1",
];

pub const DEFAULT_FALSE_SPELLINGS: &[&str] = &[
    "false", "False", "FALSE", " false", " False",
    "no", "No", "NO", " no", " No",
    "0", " 0",
];

/// A vocabulary pool mapping a semantic class to multiple tokenizer token IDs.
#[derive(Debug, Clone)]
pub struct ClassTokenPool {
    pub class_name: String,
    pub token_ids: Vec<u32>,
    pub spellings: Vec<String>,
}

impl ClassTokenPool {
    pub fn new(class_name: impl Into<String>, token_ids: Vec<u32>, spellings: Vec<String>) -> Self {
        Self {
            class_name: class_name.into(),
            token_ids,
            spellings,
        }
    }
}

/// Pruned output projection head for candidate-only decision scoring.
///
/// Discards the vast majority of unused vocabulary rows, retaining only
/// rows corresponding to the valid decision classes and their spelling variants.
#[derive(Debug, Clone)]
pub struct PrunedHead {
    pub hidden_dim: usize,
    pub retained_token_ids: Vec<u32>,
    /// Flattened weights of size [retained_token_count, hidden_dim]
    pub weights: Vec<f32>,
}

impl PrunedHead {
    pub fn new(hidden_dim: usize, retained_token_ids: Vec<u32>, weights: Vec<f32>) -> Self {
        Self {
            hidden_dim,
            retained_token_ids,
            weights,
        }
    }

    /// Evaluates logits for the retained token subset given a final hidden state.
    pub fn forward_logits(&self, hidden_state: &[f32]) -> Result<Vec<f32>> {
        if hidden_state.len() != self.hidden_dim {
            return Err(ZevError::Internal(format!(
                "Hidden state dimension {} != head hidden dim {}",
                hidden_state.len(),
                self.hidden_dim
            )));
        }

        let num_tokens = self.retained_token_ids.len();
        let mut logits = Vec::with_capacity(num_tokens);

        for i in 0..num_tokens {
            let offset = i * self.hidden_dim;
            let weight_row = &self.weights[offset..offset + self.hidden_dim];
            let mut dot = 0.0f32;
            for j in 0..self.hidden_dim {
                dot += hidden_state[j] * weight_row[j];
            }
            logits.push(dot);
        }

        Ok(logits)
    }
}

/// Binary readout scoring affirmative against negative with multi-spelling max-pooling.
#[derive(Debug, Clone)]
pub struct BinaryReadout {
    pub true_indices: Vec<usize>,
    pub false_indices: Vec<usize>,
}

impl BinaryReadout {
    pub fn new(true_indices: Vec<usize>, false_indices: Vec<usize>) -> Self {
        Self {
            true_indices,
            false_indices,
        }
    }

    /// Computes max logit for TRUE variants and FALSE variants.
    pub fn max_logits(&self, retained_logits: &[f32]) -> (f32, f32) {
        let max_true = self
            .true_indices
            .iter()
            .map(|&idx| retained_logits.get(idx).copied().unwrap_or(f32::NEG_INFINITY))
            .fold(f32::NEG_INFINITY, f32::max);

        let max_false = self
            .false_indices
            .iter()
            .map(|&idx| retained_logits.get(idx).copied().unwrap_or(f32::NEG_INFINITY))
            .fold(f32::NEG_INFINITY, f32::max);

        (max_true, max_false)
    }

    /// Returns a calibrated yes-against-no sigmoid score: sigmoid(max_true - max_false).
    pub fn score(&self, retained_logits: &[f32]) -> f64 {
        let (t, f) = self.max_logits(retained_logits);
        let diff = (t - f) as f64;
        1.0 / (1.0 + (-diff).exp())
    }

    /// Returns true if affirmative logit exceeds negative logit.
    pub fn decision(&self, retained_logits: &[f32]) -> bool {
        let (t, f) = self.max_logits(retained_logits);
        t > f
    }
}
