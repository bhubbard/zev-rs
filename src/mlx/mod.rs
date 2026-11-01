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

/// Ensures the MLX default device is initialized safely, falling back to CPU (Accelerate/AMX)
/// when `ZEV_MLX_DEVICE=cpu` is set.
pub fn ensure_mlx_device() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let force_cpu = std::env::var("ZEV_MLX_DEVICE")
            .map(|v| v.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        if force_cpu {
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        }
    });
}

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
