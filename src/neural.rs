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
            "You are a strict classifier. Classify the input context into EXACTLY one of the candidate IDs, or '__insufficient__' if the context is unrelated, out-of-domain, or lacks evidence.\n\n\
            Example 1:\n\
            Context: Preheat oven to 350 degrees and bake almond cookies for 10 minutes.\n\
            Decision: [__insufficient__]\n\n\
            Example 2:\n\
            Context: Customer requested a refund for invoice INV-9042 and wants to cancel subscription.\n\
            Decision: [billing]\n\n\
            Question:\n{}\n\nCandidate Options:\n",
            question.instructions()
        );

        // Mitigate "Lost in the Middle" and attention dilution for long candidate lists:
        // Use lightweight SIMD token-set shortlisting to narrow lists > 6 down to top 6 relevant candidates
        let effective_candidates: Vec<Candidate> = if candidates.len() > 6 {
            let opt_defs: Vec<crate::types::OptionDef> = candidates
                .iter()
                .map(|c| crate::types::OptionDef {
                    id: c.id.clone(),
                    description: c.description.clone(),
                })
                .collect();
            let shortlisted = crate::shortlist::shortlist_options(&opt_defs, state, 6);
            let mut list = Vec::new();
            for s in shortlisted {
                if let Some(c) = candidates.iter().find(|c| c.id == s.id) {
                    list.push(c.clone());
                }
            }
            list
        } else {
            candidates.to_vec()
        };

        for (i, c) in effective_candidates.iter().enumerate() {
            prompt.push_str(&format!("{}. [{}] {}\n", i + 1, c.id, c.description));
        }

        prompt.push_str(&format!(
            "\nContext to Classify:\n{}\n\nDecision (respond ONLY with the option ID inside brackets like [id] or [__insufficient__], no explanation):",
            state.trim()
        ));

        let req = GenerateRequest {
            prompt,
            system_prompt: Some("You are an accurate, deterministic on-device classifier. Respond strictly with [id] or [__insufficient__].".into()),
            messages: None,
            temperature: Some(0.0),
            top_p: Some(0.9),
            max_tokens: Some(32),
            permissive: true,
            seed: Some(42),
        };

        let resp = self.engine.generate(&req)
            .map_err(|e| ZevError::Evaluation(format!("Apfel neural generation failed: {e}")))?;

        let raw = resp.content.trim();
        let matched_id = parse_candidate_choice(raw, candidates);

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
            margin: if status == "ok" { Some(0.90) } else { None },
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
            source: Some("neural".to_string()),
        })
    }
}

#[cfg(feature = "neural")]
fn parse_candidate_choice(raw: &str, candidates: &[Candidate]) -> String {
    let trimmed = raw.trim();

    // 1. Bracket extraction: [id]
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed[start..].find(']') {
            let inside = trimmed[start + 1..start + end].trim();
            if inside.eq_ignore_ascii_case("__insufficient__")
                || inside.eq_ignore_ascii_case("insufficient")
                || inside.eq_ignore_ascii_case("none")
            {
                return "__insufficient__".to_string();
            }
            if let Some(c) = candidates.iter().find(|c| c.id.eq_ignore_ascii_case(inside)) {
                return c.id.clone();
            }
        }
    }

    let lower = trimmed.to_lowercase();

    // 2. Insufficient / out-of-domain indicators
    if lower.contains("__insufficient__")
        || lower.contains("insufficient")
        || lower.contains("unrelated")
        || lower.contains("out of domain")
        || lower.contains("none of the")
    {
        return "__insufficient__".to_string();
    }

    // 3. Numbered choice prefix (e.g., "1.", "1", "Option 1")
    for (i, c) in candidates.iter().enumerate() {
        let idx = i + 1;
        if trimmed == format!("{idx}")
            || trimmed.starts_with(&format!("{idx}."))
            || trimmed.starts_with(&format!("{idx} "))
            || lower.starts_with(&format!("option {idx}"))
        {
            return c.id.clone();
        }
    }

    // 4. Exact ID match
    for c in candidates {
        if c.id.eq_ignore_ascii_case(trimmed) {
            return c.id.clone();
        }
    }

    // 5. Keyword presence in output
    for c in candidates {
        if lower.contains(&c.id.to_lowercase()) {
            return c.id.clone();
        }
    }

    "__insufficient__".to_string()
}

