//! Neural backend integration for zev-rs using apfel-rs (Apple Intelligence / FoundationModels).
//!
//! Provides speculative neural fallback when fast SIMD heuristics encounter low-confidence
//! or ambiguous inputs.

#[cfg(feature = "neural")]
use std::sync::Arc;
#[cfg(feature = "neural")]
use apfel::{default_engine, BackendEngine, GenerateRequest};
#[cfg(feature = "neural")]
use crate::error::{Result, ZevError};
#[cfg(feature = "neural")]
use crate::types::{Candidate, Question, ZevAnswer};

#[cfg(feature = "neural")]
pub struct ApfelNeuralBackend {
    engine: Arc<dyn BackendEngine>,
}

#[cfg(feature = "neural")]
impl Default for ApfelNeuralBackend {
    fn default() -> Self {
        Self {
            engine: default_engine(),
        }
    }
}

#[cfg(feature = "neural")]
impl ApfelNeuralBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_engine(engine: Arc<dyn BackendEngine>) -> Self {
        Self { engine }
    }

    /// Evaluates candidates using on-device Apple Intelligence FoundationModels
    pub fn evaluate_candidates(
        &self,
        state: &str,
        question: &Question,
        candidates: &[Candidate],
    ) -> Result<ZevAnswer> {
        let mut prompt = format!(
            "You are an on-device decision classifier. Evaluate the following context and select the best matching option.\n\nContext:\n{}\n\nQuestion:\n{}\n\nAvailable Options:\n",
            state.trim(),
            question.instructions()
        );

        for c in candidates {
            prompt.push_str(&format!("- [{}] {}\n", c.id, c.description));
        }

        prompt.push_str("\nRespond strictly with the option ID that best matches. If the context is unrelated or insufficient, respond with '__insufficient__'. Output only the option ID and nothing else.");

        let req = GenerateRequest {
            prompt,
            system_prompt: Some("You are an accurate, deterministic on-device classifier.".into()),
            messages: None,
            temperature: Some(0.1),
            top_p: Some(0.9),
            max_tokens: Some(16),
            permissive: true,
            seed: Some(42),
        };

        let resp = self.engine.generate(&req)
            .map_err(|e| ZevError::Evaluation(format!("Apfel neural generation failed: {e}")))?;

        let raw_decision = resp.content.trim().trim_matches(|c| c == '"' || c == '\'' || c == '[' || c == ']');

        // Match against candidate IDs
        let matched_id = candidates
            .iter()
            .find(|c| c.id.eq_ignore_ascii_case(raw_decision))
            .map(|c| c.id.clone())
            .unwrap_or_else(|| {
                // Fallback to substring search in response
                candidates
                    .iter()
                    .find(|c| raw_decision.to_lowercase().contains(&c.id.to_lowercase()))
                    .map(|c| c.id.clone())
                    .unwrap_or_else(|| "__insufficient__".into())
            });

        let mut probabilities = std::collections::BTreeMap::new();
        let mut logits = std::collections::BTreeMap::new();
        for c in candidates {
            if c.id == matched_id {
                probabilities.insert(c.id.clone(), 0.95);
                logits.insert(c.id.clone(), 3.0);
            } else {
                let remainder = 0.05 / (candidates.len().max(2) - 1) as f64;
                probabilities.insert(c.id.clone(), remainder);
                logits.insert(c.id.clone(), 0.0);
            }
        }

        let status = if matched_id == "__insufficient__" {
            "insufficient_evidence".to_string()
        } else {
            "ok".to_string()
        };

        let uncertainty = crate::types::UncertaintyMetrics {
            top_probability: if status == "ok" { 0.95 } else { 0.0 },
            entropy_nats: 0.1,
            concentration: 0.9,
            unavailable_probability: if matched_id == "__insufficient__" { 0.95 } else { 0.0 },
        };

        let q_type = match question {
            Question::Boolean(_) => "boolean",
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Numeric(_) => "numeric",
        };

        Ok(ZevAnswer {
            question_type: q_type.to_string(),
            status: status.clone(),
            decision: if status == "ok" { Some(serde_json::Value::String(matched_id)) } else { None },
            confidence: if status == "ok" { 0.95 } else { 0.0 },
            probabilities,
            logits,
            uncertainty,
            statistics: None,
            expected_value: None,
            temperature: 1.0,
        })
    }
}
