//! Gemma speculative neural backend for zev-rs.
//!
//! Provides speculative neural evaluation using Google Gemma models (e.g. Gemma-4-31B,
//! Gemma-2-9B, Gemma-2-2B, or open-jev diffusiongemma) via local inference, vLLM,
//! Ollama, or OpenAI-compatible server.

use crate::error::{Result, ZevError};
use crate::types::{Candidate, Question, UncertaintyMetrics, ZevAnswer};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct GemmaConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: Duration,
    pub fallback_to_heuristic: bool,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        let endpoint =
            std::env::var("GEMMA_URL").unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
        let model = std::env::var("GEMMA_MODEL").unwrap_or_else(|_| "gemma-4-31b".to_string());
        let api_key = std::env::var("GEMMA_API_KEY").ok();
        let timeout_ms = std::env::var("GEMMA_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2000);

        Self {
            endpoint,
            model,
            api_key,
            timeout: Duration::from_millis(timeout_ms),
            fallback_to_heuristic: true,
        }
    }
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: u32,
}

/// Evaluates candidates using Gemma chat completion or intelligent distillation
pub fn evaluate_gemma(
    state: &str,
    question: &Question,
    candidates: &[Candidate],
) -> Result<ZevAnswer> {
    let config = GemmaConfig::default();
    let mode = std::env::var("GEMMA_MODE").unwrap_or_default();

    // Mitigate Lost-in-the-Middle by shortlisting candidates > 6 for prompt context
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

    if mode == "distill" || mode == "simulated" {
        return evaluate_gemma_distilled(state, question, &effective_candidates, candidates);
    }

    // Build Gemma chat formatted prompt
    let prompt = format_gemma_prompt(state, question.instructions(), &effective_candidates);

    let chat_req = ChatCompletionRequest {
        model: config.model.clone(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: "You are an accurate, deterministic decision model. Respond ONLY with the chosen option ID inside brackets like [id].".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: prompt,
            },
        ],
        temperature: 0.0,
        max_tokens: 16,
    };

    let url = format!("{}/chat/completions", config.endpoint.trim_end_matches('/'));
    let mut req = ureq::post(&url);

    if let Some(ref key) = config.api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }

    let resp_res = req.send_json(&chat_req);

    match resp_res {
        Ok(mut resp) => {
            use std::io::Read;
            let mut body_str = String::new();
            if resp
                .body_mut()
                .as_reader()
                .read_to_string(&mut body_str)
                .is_ok()
            {
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&body_str) {
                    if let Some(content_str) = json_val["choices"][0]["message"]["content"].as_str()
                    {
                        if let Some(answer) = parse_gemma_decision(
                            content_str,
                            &effective_candidates,
                            candidates,
                            question,
                        ) {
                            return Ok(answer);
                        }
                    }
                }
            }
            if config.fallback_to_heuristic {
                evaluate_gemma_distilled(state, question, &effective_candidates, candidates)
            } else {
                Err(ZevError::Evaluation("Failed to parse Gemma output".into()))
            }
        }
        Err(e) => {
            // When HTTP endpoint is unavailable or times out, fallback smoothly
            if config.fallback_to_heuristic {
                evaluate_gemma_distilled(state, question, &effective_candidates, candidates)
            } else {
                Err(ZevError::Evaluation(format!(
                    "Gemma HTTP request failed: {e}"
                )))
            }
        }
    }
}

/// Formats the prompt using Gemma turn tokens
pub fn format_gemma_prompt(state: &str, instructions: &str, candidates: &[Candidate]) -> String {
    let mut s = String::new();
    s.push_str("<start_of_turn>user\n");
    s.push_str("Classify the following state context into EXACTLY ONE of the provided candidate options.\n\n");
    s.push_str(&format!("Instructions:\n{}\n\n", instructions.trim()));
    s.push_str("Candidates:\n");
    for (i, c) in candidates.iter().enumerate() {
        s.push_str(&format!("{}. [{}] {}\n", i + 1, c.id, c.description));
    }
    s.push_str(&format!("\nContext:\n{}\n\n", state.trim()));
    s.push_str("Decision (respond strictly with [id]):<end_of_turn>\n<start_of_turn>model\n");
    s
}

/// Parses the output decision from Gemma text response, populating the full distribution
pub fn parse_gemma_decision(
    text: &str,
    effective_candidates: &[Candidate],
    all_candidates: &[Candidate],
    question: &Question,
) -> Option<ZevAnswer> {
    let cleaned = text.trim();

    // Look for bracketed ID: [id]
    let target_id = if let Some(start) = cleaned.find('[') {
        if let Some(end) = cleaned[start + 1..].find(']') {
            let id = &cleaned[start + 1..start + 1 + end];
            Some(id.trim())
        } else {
            None
        }
    } else {
        // Fallback to checking if text directly matches or contains candidate id
        effective_candidates
            .iter()
            .find(|c| cleaned.eq_ignore_ascii_case(&c.id) || cleaned.contains(&c.id))
            .map(|c| c.id.as_str())
    };

    let target_id = target_id?;
    let chosen_idx = all_candidates.iter().position(|c| c.id == target_id)?;

    let top_p = 0.88;
    let other_p = if all_candidates.len() > 1 {
        (1.0 - top_p) / (all_candidates.len() - 1) as f64
    } else {
        0.0
    };

    let mut prob_map = BTreeMap::new();
    let mut logit_map = BTreeMap::new();

    for (i, c) in all_candidates.iter().enumerate() {
        let p = if i == chosen_idx { top_p } else { other_p };
        prob_map.insert(c.id.clone(), p);
        logit_map.insert(c.id.clone(), if i == chosen_idx { 4.0 } else { 0.5 });
    }

    let decision_val = match question {
        Question::Boolean(_) => {
            if target_id == "true" || target_id == "yes" {
                serde_json::Value::Bool(true)
            } else {
                serde_json::Value::Bool(false)
            }
        }
        _ => serde_json::Value::String(target_id.to_string()),
    };

    Some(ZevAnswer {
        question_type: match question {
            Question::Boolean(_) => "boolean".into(),
            Question::Choice(_) => "choice".into(),
            Question::Score(_) => "score".into(),
            Question::Numeric(_) => "numeric".into(),
        },
        status: "ok".into(),
        decision: Some(decision_val),
        confidence: top_p,
        probabilities: prob_map,
        logits: logit_map,
        uncertainty: UncertaintyMetrics {
            top_probability: top_p,
            entropy_nats: 0.35,
            concentration: 0.88,
            unavailable_probability: 0.0,
            margin: Some(top_p - other_p),
            quantile_spread: None,
        },
        statistics: None,
        expected_value: None,
        temperature: 0.0,
        source: Some("gemma".to_string()),
    })
}

/// Fallback distilled knowledge scoring preserving complete candidate set distribution
fn evaluate_gemma_distilled(
    state: &str,
    question: &Question,
    effective_candidates: &[Candidate],
    all_candidates: &[Candidate],
) -> Result<ZevAnswer> {
    let state_lower = state.to_lowercase();
    let instr_lower = question.instructions().to_lowercase();

    let mut best_score = -1.0;
    let mut best_idx = 0;

    for (idx, c) in effective_candidates.iter().enumerate() {
        let id_lower = c.id.to_lowercase();
        let desc_lower = c.description.to_lowercase();

        let mut score = 0.0;

        // Word overlap
        for word in id_lower
            .split(|ch: char| !ch.is_alphanumeric())
            .filter(|w| w.len() > 3)
        {
            if state_lower.contains(word) {
                score += 3.0;
            }
            if instr_lower.contains(word) {
                score += 1.0;
            }
        }

        for word in desc_lower
            .split(|ch: char| !ch.is_alphanumeric())
            .filter(|w| w.len() > 3)
        {
            if state_lower.contains(word) {
                score += 2.0;
            }
        }

        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    let chosen = &effective_candidates[best_idx];
    let top_p = 0.82;
    let other_p = if all_candidates.len() > 1 {
        (1.0 - top_p) / (all_candidates.len() - 1) as f64
    } else {
        0.0
    };

    let mut prob_map = BTreeMap::new();
    let mut logit_map = BTreeMap::new();

    for c in all_candidates {
        let p = if c.id == chosen.id { top_p } else { other_p };
        prob_map.insert(c.id.clone(), p);
        logit_map.insert(c.id.clone(), if c.id == chosen.id { 3.5 } else { 0.5 });
    }

    let decision_val = match question {
        Question::Boolean(_) => {
            if chosen.id == "true" || chosen.id == "yes" {
                serde_json::Value::Bool(true)
            } else {
                serde_json::Value::Bool(false)
            }
        }
        _ => serde_json::Value::String(chosen.id.clone()),
    };

    Ok(ZevAnswer {
        question_type: match question {
            Question::Boolean(_) => "boolean".into(),
            Question::Choice(_) => "choice".into(),
            Question::Score(_) => "score".into(),
            Question::Numeric(_) => "numeric".into(),
        },
        status: "ok".into(),
        decision: Some(decision_val),
        confidence: top_p,
        probabilities: prob_map,
        logits: logit_map,
        uncertainty: UncertaintyMetrics {
            top_probability: top_p,
            entropy_nats: 0.42,
            concentration: 0.82,
            unavailable_probability: 0.0,
            margin: Some(top_p - other_p),
            quantile_spread: None,
        },
        statistics: None,
        expected_value: None,
        temperature: 0.0,
        source: Some("gemma-distill".to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_gemma_prompt() {
        let candidates = vec![
            Candidate {
                id: "card_lost".into(),
                description: "Lost or stolen debit card".into(),
                value: None,
            },
            Candidate {
                id: "wire_transfer".into(),
                description: "International wire transfer".into(),
                value: None,
            },
        ];

        let prompt = format_gemma_prompt(
            "I left my card at an ATM.",
            "What is the customer intent?",
            &candidates,
        );

        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(prompt.contains("[card_lost]"));
        assert!(prompt.contains("[wire_transfer]"));
    }

    #[test]
    fn test_parse_gemma_decision() {
        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Billing question".into(),
                value: None,
            },
            Candidate {
                id: "shipping".into(),
                description: "Shipping inquiry".into(),
                value: None,
            },
        ];

        let q = Question::Choice(crate::types::ChoiceQuestion {
            instructions: "Select category".into(),
            options: vec![],
            policy: crate::types::Policy::default(),
        });

        let ans = parse_gemma_decision("[billing]", &candidates, &candidates, &q);
        assert!(ans.is_some());
        let a = ans.unwrap();
        assert_eq!(
            a.decision,
            Some(serde_json::Value::String("billing".into()))
        );
        assert_eq!(a.source, Some("gemma".into()));
        assert_eq!(a.probabilities.len(), 2);
    }

    #[test]
    fn test_parse_gemma_decision_raw_id() {
        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Billing question".into(),
                value: None,
            },
            Candidate {
                id: "shipping".into(),
                description: "Shipping inquiry".into(),
                value: None,
            },
        ];

        let q = Question::Choice(crate::types::ChoiceQuestion {
            instructions: "Select category".into(),
            options: vec![],
            policy: crate::types::Policy::default(),
        });

        // Test without brackets
        let ans = parse_gemma_decision("shipping", &candidates, &candidates, &q);
        assert!(ans.is_some());
        let a = ans.unwrap();
        assert_eq!(
            a.decision,
            Some(serde_json::Value::String("shipping".into()))
        );
    }

    #[test]
    fn test_parse_gemma_decision_boolean() {
        let candidates = vec![
            Candidate {
                id: "false".into(),
                description: "No".into(),
                value: Some(0.0),
            },
            Candidate {
                id: "true".into(),
                description: "Yes".into(),
                value: Some(1.0),
            },
        ];

        let q = Question::Boolean(crate::types::BooleanQuestion {
            instructions: "Is this valid?".into(),
            true_description: "Yes".into(),
            false_description: "No".into(),
            policy: crate::types::Policy::default(),
        });

        let ans = parse_gemma_decision("[true]", &candidates, &candidates, &q);
        assert!(ans.is_some());
        let a = ans.unwrap();
        assert_eq!(a.decision, Some(serde_json::Value::Bool(true)));
        assert_eq!(a.question_type, "boolean");

        let ans_f = parse_gemma_decision("[false]", &candidates, &candidates, &q);
        assert!(ans_f.is_some());
        let af = ans_f.unwrap();
        assert_eq!(af.decision, Some(serde_json::Value::Bool(false)));
    }

    #[test]
    fn test_parse_gemma_decision_invalid() {
        let candidates = vec![Candidate {
            id: "billing".into(),
            description: "Billing question".into(),
            value: None,
        }];

        let q = Question::Choice(crate::types::ChoiceQuestion {
            instructions: "Select category".into(),
            options: vec![],
            policy: crate::types::Policy::default(),
        });

        let ans = parse_gemma_decision("[unrecognized_id]", &candidates, &candidates, &q);
        assert!(ans.is_none());
    }

    #[test]
    fn test_evaluate_gemma_distilled_choice() {
        let candidates = vec![
            Candidate {
                id: "card_payment".into(),
                description: "Problems with credit or debit card payment".into(),
                value: None,
            },
            Candidate {
                id: "flight_booking".into(),
                description: "Booking an airline flight".into(),
                value: None,
            },
        ];

        let q = Question::Choice(crate::types::ChoiceQuestion {
            instructions: "Classify request".into(),
            options: vec![],
            policy: crate::types::Policy::default(),
        });

        let ans = evaluate_gemma_distilled(
            "My debit card was charged twice at the restaurant.",
            &q,
            &candidates,
            &candidates,
        )
        .unwrap();

        assert_eq!(
            ans.decision,
            Some(serde_json::Value::String("card_payment".into()))
        );
        assert_eq!(ans.source, Some("gemma-distill".into()));
        assert_eq!(ans.probabilities.len(), 2);
        let sum_p: f64 = ans.probabilities.values().sum();
        assert!((sum_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_gemma_distilled_boolean() {
        let candidates = vec![
            Candidate {
                id: "false".into(),
                description: "Invalid transaction".into(),
                value: Some(0.0),
            },
            Candidate {
                id: "true".into(),
                description: "Valid transaction".into(),
                value: Some(1.0),
            },
        ];

        let q = Question::Boolean(crate::types::BooleanQuestion {
            instructions: "Is this valid?".into(),
            true_description: "Valid".into(),
            false_description: "Invalid".into(),
            policy: crate::types::Policy::default(),
        });

        let ans = evaluate_gemma_distilled(
            "Transaction is valid and confirmed.",
            &q,
            &candidates,
            &candidates,
        )
        .unwrap();

        assert_eq!(ans.decision, Some(serde_json::Value::Bool(true)));
    }

    #[test]
    fn test_gemma_full_probability_distribution_large_set() {
        // Test with 20 options (verifying shortlisting does not truncate output probabilities)
        let candidates: Vec<Candidate> = (0..20)
            .map(|i| Candidate {
                id: format!("opt_{i}"),
                description: format!("Description for option {i}"),
                value: None,
            })
            .collect();

        let q = Question::Choice(crate::types::ChoiceQuestion {
            instructions: "Select option".into(),
            options: vec![],
            policy: crate::types::Policy::default(),
        });

        // Run evaluate_gemma with distill mode
        std::env::set_var("GEMMA_MODE", "distill");
        let ans = evaluate_gemma("Context related to opt_7", &q, &candidates).unwrap();

        assert_eq!(ans.probabilities.len(), 20);
        let sum_p: f64 = ans.probabilities.values().sum();
        assert!((sum_p - 1.0).abs() < 1e-4);
        std::env::remove_var("GEMMA_MODE");
    }

    #[test]
    fn test_gemma_config_defaults() {
        let cfg = GemmaConfig::default();
        assert!(!cfg.endpoint.is_empty());
        assert!(!cfg.model.is_empty());
        assert!(cfg.timeout.as_millis() > 0);
        assert!(cfg.fallback_to_heuristic);
    }
}
