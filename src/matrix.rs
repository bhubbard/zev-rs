use crate::calibration::scaled_softmax;
use crate::error::{Result, ZevError};
use std::collections::{BTreeMap, HashMap};

/// Pre-encoded matrix of partner options for fast asymmetric evaluation.
///
/// In Quail-style joins or large taxonomy classification, candidate criteria (partners)
/// are fixed across many incoming items (anchors). Pre-vectorizing partners enables
/// evaluating an anchor across all M options in a single SIMD dot-product pass.
#[derive(Debug, Clone)]
pub struct PartnerMatrix {
    /// Option IDs in row order
    pub ids: Vec<String>,
    /// Vocabulary dictionary mapping terms to column indices
    pub vocab: HashMap<String, usize>,
    /// Flattened row-major matrix of dimension [M, D]
    matrix: Vec<f32>,
    /// Number of candidate options M
    num_partners: usize,
    /// Vector dimension D (vocabulary size)
    dim: usize,
}

impl PartnerMatrix {
    /// Builds a partner matrix from candidate option definitions using an exact vocabulary dictionary.
    pub fn from_options(options: &[(&str, &str)], _dim_hint: usize) -> Self {
        let num_partners = options.len();
        let mut ids = Vec::with_capacity(num_partners);
        let mut vocab = HashMap::new();

        // 1. Build vocabulary across all partner definitions
        for &(id, desc) in options {
            ids.push(id.to_string());
            let full_text = format!("{id} {desc}");
            let terms = extract_terms(&full_text);
            for (term, _weight) in terms {
                let next_idx = vocab.len();
                vocab.entry(term).or_insert(next_idx);
            }
        }

        let dim = vocab.len().max(1);
        let mut matrix = vec![0.0f32; num_partners * dim];

        // 2. Vectorize and L2-normalize each partner row
        for (i, &(id, desc)) in options.iter().enumerate() {
            let full_text = format!("{id} {desc}");
            let terms = extract_terms(&full_text);
            let offset = i * dim;

            for (term, weight) in terms {
                if let Some(&idx) = vocab.get(&term) {
                    matrix[offset + idx] += weight;
                }
            }

            // L2 normalize
            let norm_sq: f32 = matrix[offset..offset + dim].iter().map(|v| v * v).sum();
            if norm_sq > 0.0 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for v in &mut matrix[offset..offset + dim] {
                    *v *= inv_norm;
                }
            }
        }

        Self {
            ids,
            vocab,
            matrix,
            num_partners,
            dim,
        }
    }

    #[inline]
    pub fn num_partners(&self) -> usize {
        self.num_partners
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluates an anchor vector against all partner rows using SIMD dot products.
    /// Returns raw logits for each partner option.
    pub fn evaluate_anchor_logits(&self, anchor_vec: &[f32]) -> Result<Vec<f64>> {
        if anchor_vec.len() != self.dim {
            return Err(ZevError::Internal(format!(
                "Anchor vector dimension {} does not match matrix dimension {}",
                anchor_vec.len(),
                self.dim
            )));
        }

        let mut logits = Vec::with_capacity(self.num_partners);
        for i in 0..self.num_partners {
            let offset = i * self.dim;
            let partner_slice = &self.matrix[offset..offset + self.dim];

            // Dot product with logit scaling
            let mut dot = 0.0f32;
            for j in 0..self.dim {
                dot += anchor_vec[j] * partner_slice[j];
            }
            logits.push((dot * 12.0) as f64);
        }

        Ok(logits)
    }

    /// Evaluates an anchor text against all partners, returning calibrated probabilities.
    pub fn evaluate_anchor(
        &self,
        anchor_text: &str,
        temperature: f64,
    ) -> Result<BTreeMap<String, f64>> {
        let mut anchor_vec = vec![0.0f32; self.dim];
        let terms = extract_terms(anchor_text);

        for (term, weight) in terms {
            if let Some(&idx) = self.vocab.get(&term) {
                anchor_vec[idx] += weight;
            }
        }

        // L2 normalize
        let norm_sq: f32 = anchor_vec.iter().map(|v| v * v).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for v in &mut anchor_vec {
                *v *= inv_norm;
            }
        }

        let logits = self.evaluate_anchor_logits(&anchor_vec)?;
        let probs = scaled_softmax(&logits, temperature)?;

        let mut result = BTreeMap::new();
        for (id, prob) in self.ids.iter().cloned().zip(probs) {
            result.insert(id, prob);
        }
        Ok(result)
    }

    /// Finds the top partner and its probability.
    pub fn top_partner(&self, anchor_text: &str, temperature: f64) -> Result<(String, f64)> {
        let probs = self.evaluate_anchor(anchor_text, temperature)?;
        let (best_id, best_prob) = probs
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ZevError::Internal("Empty partner matrix".to_string()))?;
        Ok((best_id, best_prob))
    }
}

/// Evaluator coordinating fast Anchor-Partner asymmetric scoring.
pub struct AnchorPartnerEvaluator {
    matrix: PartnerMatrix,
    temperature: f64,
}

impl AnchorPartnerEvaluator {
    pub fn new(matrix: PartnerMatrix, temperature: f64) -> Self {
        Self {
            matrix,
            temperature,
        }
    }

    pub fn matrix(&self) -> &PartnerMatrix {
        &self.matrix
    }

    /// Evaluates a stream of anchor items against the pre-encoded partner matrix.
    pub fn evaluate_anchors(&self, anchors: &[(&str, &str)]) -> Result<Vec<(String, String, f64)>> {
        let mut results = Vec::with_capacity(anchors.len());
        for &(id, text) in anchors {
            let (top_partner, prob) = self.matrix.top_partner(text, self.temperature)?;
            results.push((id.to_string(), top_partner, prob));
        }
        Ok(results)
    }
}

/// Extracts terms (words, stems, and bigrams) with their associated term weights.
fn extract_terms(text: &str) -> Vec<(String, f32)> {
    let lower = text.to_lowercase();
    let mut words: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in lower.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }

    let mut terms = Vec::new();
    for word in &words {
        terms.push((word.clone(), 1.0));
        let s = stem(word);
        if s != word {
            terms.push((s.to_string(), 1.2));
        }
    }

    for window in words.windows(2) {
        let bigram = format!("{}_{}", window[0], window[1]);
        terms.push((bigram, 1.5));
    }

    terms
}

#[inline]
fn stem(w: &str) -> &str {
    if w.ends_with("ing") && w.len() > 5 {
        &w[..w.len() - 3]
    } else if w.ends_with("ed") && w.len() > 4 {
        &w[..w.len() - 2]
    } else if w.ends_with('s') && !w.ends_with("ss") && w.len() > 3 {
        &w[..w.len() - 1]
    } else {
        w
    }
}
