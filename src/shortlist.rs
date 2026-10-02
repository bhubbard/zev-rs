use crate::types::OptionDef;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const DEFAULT_SHORTLIST_K: usize = 20;

/// Shortlist metadata recording pruned options and similarity scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShortlistResult {
    /// Kept option IDs in descending similarity rank.
    pub labels: Vec<String>,
    /// Cosine similarity scores for each kept option (None if lexical overlap was used).
    pub scores: Option<Vec<f64>>,
    /// Requested top-k budget.
    pub k: usize,
    /// Total original candidate count.
    pub n: usize,
    /// Whether all labels were kept without pruning (e.g. k >= n).
    pub passthrough: bool,
}

/// Compute cosine similarity between one query vector and a set of candidate document vectors.
/// Synthesized from `laya-rs`.
pub fn cosine_similarity(query: &[f64], docs: &[Vec<f64>]) -> Vec<f64> {
    let q_norm = query.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let mut sims = vec![0.0; docs.len()];

    if q_norm == 0.0 || !q_norm.is_finite() {
        return sims;
    }

    for (i, doc) in docs.iter().enumerate() {
        if doc.len() != query.len() {
            continue;
        }
        let d_norm = doc.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if d_norm == 0.0 || !d_norm.is_finite() {
            sims[i] = 0.0;
            continue;
        }
        let dot: f64 = query.iter().zip(doc.iter()).map(|(&q, &d)| q * d).sum();
        let sim = dot / (q_norm * d_norm);
        sims[i] = if sim.is_nan() { 0.0 } else { sim };
    }

    sims
}

/// Shortlists candidate options using dense vector embeddings and cosine similarity.
/// If options <= max_slots, returns the list unchanged.
pub fn shortlist_by_embedding(
    options: &[OptionDef],
    state_vec: &[f64],
    option_vecs: &[Vec<f64>],
    max_slots: usize,
) -> Vec<OptionDef> {
    if options.len() <= max_slots || option_vecs.len() != options.len() {
        return options.to_vec();
    }

    let sims = cosine_similarity(state_vec, option_vecs);
    let mut indexed: Vec<(f64, &OptionDef)> = sims.into_iter().zip(options.iter()).collect();

    indexed.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.id.cmp(&b.1.id))
    });

    indexed
        .into_iter()
        .take(max_slots)
        .map(|(_, opt)| opt.clone())
        .collect()
}

/// Shortlists candidate options using lightweight token-set overlap with ID boost and canonical tie-breaking.
/// If options <= max_slots, returns the list unchanged.
pub fn shortlist_options(options: &[OptionDef], state: &str, max_slots: usize) -> Vec<OptionDef> {
    if options.len() <= max_slots {
        return options.to_vec();
    }

    let state_tokens: HashSet<String> = state
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 2)
        .collect();

    let mut scored: Vec<(f64, &OptionDef)> = options
        .iter()
        .map(|opt| {
            let mut matches = 0.0;
            let mut opt_token_count = 0usize;

            // Direct ID match & constituent ID token match
            let id_lower = opt.id.to_lowercase();
            if state_tokens.contains(&id_lower) {
                matches += 4.0;
            }
            for part in id_lower.split(['_', '-']) {
                if part.len() > 2 {
                    opt_token_count += 1;
                    if state_tokens.contains(part) {
                        matches += 2.5;
                    }
                }
            }

            // Description tokens
            for word in opt.description.split_whitespace() {
                let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
                if trimmed.len() > 2 {
                    opt_token_count += 1;
                    let word_lower = trimmed.to_lowercase();
                    if state_tokens.contains(&word_lower) {
                        // Weighted by length (discriminative tokens carry more weight)
                        let weight = 1.0 + (word_lower.len() as f64 - 3.0).clamp(0.0, 3.0) * 0.2;
                        matches += weight;
                    }
                }
            }

            let denom = ((state_tokens.len().max(1) * opt_token_count.max(1)) as f64)
                .sqrt()
                .max(1.0);
            let score = matches / denom;

            (score, opt)
        })
        .collect();

    // Sort descending by score, breaking ties canonically by ID
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.id.cmp(&b.1.id))
    });

    // Keep top max_slots
    scored
        .into_iter()
        .take(max_slots)
        .map(|(_, opt)| opt.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let q = vec![1.0, 0.0, 0.0];
        let docs = vec![
            vec![1.0, 0.0, 0.0],  // identical => 1.0
            vec![0.0, 1.0, 0.0],  // orthogonal => 0.0
            vec![-1.0, 0.0, 0.0], // opposite => -1.0
        ];
        let sims = cosine_similarity(&q, &docs);
        assert!((sims[0] - 1.0).abs() < 1e-5);
        assert!((sims[1] - 0.0).abs() < 1e-5);
        assert!((sims[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_shortlist_by_embedding() {
        let options = vec![
            OptionDef {
                id: "opt_a".into(),
                description: "first".into(),
            },
            OptionDef {
                id: "opt_b".into(),
                description: "second".into(),
            },
            OptionDef {
                id: "opt_c".into(),
                description: "third".into(),
            },
        ];
        let state_vec = vec![0.9, 0.1];
        let opt_vecs = vec![vec![0.1, 0.9], vec![0.85, 0.15], vec![0.0, 1.0]];
        let pruned = shortlist_by_embedding(&options, &state_vec, &opt_vecs, 2);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].id, "opt_b"); // Highest cosine match
    }
}
