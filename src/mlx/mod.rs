//! MLX backend for Apple Silicon hardware acceleration.
//!
//! Provides Metal GPU-accelerated semantic routing, batch vector evaluation,
//! projection heads, local SLM decision heads (Gemma 4 & Qwen), and dead-letter
//! queue (DLQ) clustering directly in unified memory.

pub mod embedding;
pub mod sieve;
pub mod triage;

pub use embedding::{DlqClusterSummary, MlxDlqClusterer, MlxProjectionHead};
pub use sieve::MlxSemanticSieve;
pub use triage::{MlxTriageClassifier, SlmModelFamily};

/// Speculative fallback evaluator powered by Apple MLX on Metal.
pub fn evaluate_speculative_mlx(
    state: &str,
    question: &crate::types::Question,
    candidates: &[crate::types::Candidate],
    _conf_thresh: f64,
    _margin_thresh: f64,
) -> crate::error::Result<Option<crate::types::ZevAnswer>> {
    let family_var = std::env::var("MLX_MODEL_FAMILY")
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    let family = if family_var == "qwen" {
        SlmModelFamily::Qwen
    } else {
        // Default to Google Gemma (Gemma 4 / Gemma 2)
        SlmModelFamily::Gemma
    };

    let classifier = MlxTriageClassifier::new(family, 128, 64)?;
    let ans = classifier.evaluate_candidates(state, question, candidates)?;
    Ok(Some(ans))
}
