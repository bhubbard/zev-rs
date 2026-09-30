//! Neural backend integration for zev-rs using apfel-rs (Apple Intelligence / FoundationModels).
//!
//! Provides speculative neural fallback when fast SIMD heuristics encounter low-confidence
//! or ambiguous inputs.

#[cfg(feature = "neural")]
use crate::error::{Result, ZevError};
#[cfg(feature = "neural")]
use crate::types::{Candidate, Question, ZevAnswer};
#[cfg(feature = "neural")]
use apfel::{default_engine, BackendEngine, GenerateRequest};
#[cfg(feature = "neural")]
use std::sync::Arc;

#[cfg(feature = "neural")]
pub struct ApfelNeuralBackend {
    engine: Arc<dyn BackendEngine>,
}

#[cfg(feature = "neural")]
impl std::fmt::Debug for ApfelNeuralBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApfelNeuralBackend").finish_non_exhaustive()
    }
}

#[cfg(feature = "neural")]
impl Default for ApfelNeuralBackend {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(feature = "neural")]
impl ApfelNeuralBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initializes backend from `APFEL_ENGINE` ("foundation", "mlx", "mock") and `APFEL_MODEL` environment variables.
    pub fn from_env() -> Self {
        let engine_name = std::env::var("APFEL_ENGINE").ok();
        let model_name = std::env::var("APFEL_MODEL").ok();
        if engine_name.is_some() || model_name.is_some() {
            Self {
                engine: apfel::create_engine(engine_name.as_deref(), model_name.as_deref()),
            }
        } else {
            Self {
                engine: default_engine(),
            }
        }
    }

    /// Creates an Apfel neural backend with specific engine and model identifiers.
    pub fn with_engine_and_model(engine_name: Option<&str>, model_name: Option<&str>) -> Self {
        Self {
            engine: apfel::create_engine(engine_name, model_name),
        }
    }

    pub fn with_engine(engine: Arc<dyn BackendEngine>) -> Self {
        Self { engine }
    }

    /// Evaluates candidates using on-device Apple Intelligence FoundationModels or MLX
    pub fn evaluate_candidates(
        &self,
        state: &str,
        question: &Question,
        candidates: &[Candidate],
    ) -> Result<ZevAnswer> {
        crate::qos::elevate_thread_qos();
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
            top_p: None,
            max_tokens: Some(4),
            permissive: true,
            seed: Some(42),
            use_case: Some("content_tagging".into()),
        };

        let resp = self
            .engine
            .generate(&req)
            .map_err(|e| ZevError::Evaluation(format!("Apfel neural generation failed: {e}")))?;

        let raw = resp.content.trim();
        let parsed = parse_candidate_choice(raw, candidates);
        let matched_id =
            if parsed == "__insufficient__" || !candidates.iter().any(|c| c.id == parsed) {
                candidates
                    .first()
                    .map(|c| c.id.clone())
                    .unwrap_or_else(|| "none".to_string())
            } else {
                parsed
            };

        let mut probabilities = std::collections::BTreeMap::new();
        let mut logits = std::collections::BTreeMap::new();
        let top_p = 0.95;
        let other_p = if candidates.len() > 1 {
            (1.0 - top_p) / (candidates.len() - 1) as f64
        } else {
            0.0
        };

        for c in candidates {
            if c.id == matched_id {
                probabilities.insert(c.id.clone(), top_p);
                logits.insert(c.id.clone(), 3.0);
            } else {
                probabilities.insert(c.id.clone(), other_p);
                logits.insert(c.id.clone(), 0.0);
            }
        }

        let status = "ok".to_string();

        let uncertainty = crate::types::UncertaintyMetrics {
            top_probability: top_p,
            entropy_nats: 0.1,
            concentration: 0.9,
            margin: Some(top_p - other_p),
            unavailable_probability: 0.0,
            quantile_spread: None,
        };

        let q_type = match question {
            Question::Boolean(_) => "boolean",
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Numeric(_) => "numeric",
        };

        let decision_val = match question {
            Question::Boolean(_) => {
                if matched_id.eq_ignore_ascii_case("true") || matched_id.eq_ignore_ascii_case("yes")
                {
                    serde_json::Value::Bool(true)
                } else {
                    serde_json::Value::Bool(false)
                }
            }
            _ => serde_json::Value::String(matched_id),
        };

        Ok(ZevAnswer {
            question_type: q_type.to_string(),
            status,
            decision: Some(decision_val),
            confidence: top_p,
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
static SHARED_APFEL: std::sync::OnceLock<ApfelNeuralBackend> = std::sync::OnceLock::new();

#[cfg(feature = "neural")]
pub fn shared_apfel() -> &'static ApfelNeuralBackend {
    SHARED_APFEL.get_or_init(ApfelNeuralBackend::new)
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
            if let Some(c) = candidates
                .iter()
                .find(|c| c.id.eq_ignore_ascii_case(inside))
            {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BooleanQuestion, ChoiceQuestion, OptionDef, Policy};

    #[test]
    fn test_parse_candidate_choice_brackets_and_keywords() {
        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Invoices".into(),
                value: None,
            },
            Candidate {
                id: "technical".into(),
                description: "Crashes".into(),
                value: None,
            },
        ];

        // 1. Bracket matches
        assert_eq!(parse_candidate_choice("[billing]", &candidates), "billing");
        assert_eq!(
            parse_candidate_choice("Decision: [technical] please", &candidates),
            "technical"
        );
        assert_eq!(parse_candidate_choice("[BILLING]", &candidates), "billing");
        assert_eq!(
            parse_candidate_choice("[__insufficient__]", &candidates),
            "__insufficient__"
        );
        assert_eq!(
            parse_candidate_choice("[insufficient]", &candidates),
            "__insufficient__"
        );
        assert_eq!(
            parse_candidate_choice("[none]", &candidates),
            "__insufficient__"
        );

        // 2. Insufficient phrases
        assert_eq!(
            parse_candidate_choice("This input is unrelated to the context", &candidates),
            "__insufficient__"
        );
        assert_eq!(
            parse_candidate_choice("The query is out of domain", &candidates),
            "__insufficient__"
        );
        assert_eq!(
            parse_candidate_choice("none of the options fit", &candidates),
            "__insufficient__"
        );

        // 3. Numbered prefixes
        assert_eq!(parse_candidate_choice("1", &candidates), "billing");
        assert_eq!(parse_candidate_choice("1.", &candidates), "billing");
        assert_eq!(parse_candidate_choice("1. billing", &candidates), "billing");
        assert_eq!(
            parse_candidate_choice("2 technical support", &candidates),
            "technical"
        );
        assert_eq!(parse_candidate_choice("Option 1", &candidates), "billing");
        assert_eq!(parse_candidate_choice("Option 2", &candidates), "technical");

        // 4. Exact matches and substring
        assert_eq!(parse_candidate_choice("billing", &candidates), "billing");
        assert_eq!(
            parse_candidate_choice("TECHNICAL", &candidates),
            "technical"
        );
        assert_eq!(
            parse_candidate_choice("The selected category is billing today", &candidates),
            "billing"
        );

        // 5. Unmatched
        assert_eq!(
            parse_candidate_choice("completely unrelated banana fruit", &candidates),
            "__insufficient__"
        );
    }

    #[test]
    fn test_apfel_neural_backend_mock_choice_and_boolean() {
        let backend = ApfelNeuralBackend::with_engine_and_model(Some("mock"), None);
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("ApfelNeuralBackend"));

        let q_choice = Question::Choice(ChoiceQuestion {
            instructions: "Select category".into(),
            options: vec![
                OptionDef {
                    id: "billing".into(),
                    description: "Billing questions".into(),
                },
                OptionDef {
                    id: "tech".into(),
                    description: "Technical questions".into(),
                },
            ],
            policy: Policy::default(),
        });

        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Billing questions".into(),
                value: None,
            },
            Candidate {
                id: "tech".into(),
                description: "Technical questions".into(),
                value: None,
            },
        ];

        let ans = backend
            .evaluate_candidates(
                "Customer needs invoice payment help",
                &q_choice,
                &candidates,
            )
            .expect("Evaluation with mock engine must succeed");

        assert_eq!(ans.question_type, "choice");
        assert_eq!(ans.source.as_deref(), Some("neural"));
        assert!(
            ans.probabilities.contains_key("billing") || ans.probabilities.contains_key("tech")
        );

        // Boolean question
        let q_bool = Question::Boolean(BooleanQuestion {
            instructions: "Is this urgent?".into(),
            true_description: "Yes urgent".into(),
            false_description: "No routine".into(),
            policy: Policy::default(),
        });
        let bool_candidates = vec![
            Candidate {
                id: "true".into(),
                description: "Yes urgent".into(),
                value: None,
            },
            Candidate {
                id: "false".into(),
                description: "No routine".into(),
                value: None,
            },
        ];

        let bool_ans = backend
            .evaluate_candidates(
                "Production is down critical outage",
                &q_bool,
                &bool_candidates,
            )
            .expect("Boolean evaluation must succeed");

        assert_eq!(bool_ans.question_type, "boolean");
        assert!(bool_ans.decision.is_some());
    }

    #[test]
    fn test_apfel_neural_backend_shortlisting_many_candidates() {
        let backend = ApfelNeuralBackend::with_engine_and_model(Some("mock"), None);

        let mut options = Vec::new();
        let mut candidates = Vec::new();
        for i in 0..10 {
            let id = format!("dept_{i}");
            let desc = format!("Department number {i} for specific tasks");
            options.push(OptionDef {
                id: id.clone(),
                description: desc.clone(),
            });
            candidates.push(Candidate {
                id,
                description: desc,
                value: None,
            });
        }

        let q_many = Question::Choice(ChoiceQuestion {
            instructions: "Select department".into(),
            options,
            policy: Policy::default(),
        });

        let ans = backend
            .evaluate_candidates("Task for department number 3", &q_many, &candidates)
            .expect("Shortlisting evaluation must succeed");

        assert_eq!(ans.question_type, "choice");
    }

    #[test]
    fn test_apfel_neural_backend_from_env() {
        std::env::set_var("APFEL_ENGINE", "mock");
        std::env::set_var("APFEL_MODEL", "mock-model");
        let backend = ApfelNeuralBackend::from_env();
        std::env::remove_var("APFEL_ENGINE");
        std::env::remove_var("APFEL_MODEL");

        let q = Question::Choice(ChoiceQuestion {
            instructions: "Test question".into(),
            options: vec![OptionDef {
                id: "a".into(),
                description: "A".into(),
            }],
            policy: Policy::default(),
        });
        let candidates = vec![Candidate {
            id: "a".into(),
            description: "A".into(),
            value: None,
        }];
        let res = backend.evaluate_candidates("Testing state", &q, &candidates);
        assert!(res.is_ok());
    }

    #[test]
    fn test_apfel_neural_payload_properties_content_tagging() {
        let mock = Arc::new(apfel::backend::MockEngine::with_response("[billing]"));
        let backend = ApfelNeuralBackend::with_engine(mock.clone());

        let q = Question::Choice(ChoiceQuestion {
            instructions: "Classify incoming customer request".into(),
            options: vec![
                OptionDef {
                    id: "billing".into(),
                    description: "Invoices and subscription payments".into(),
                },
                OptionDef {
                    id: "technical".into(),
                    description: "Bug reports and API errors".into(),
                },
            ],
            policy: Policy::default(),
        });
        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Invoices and subscription payments".into(),
                value: None,
            },
            Candidate {
                id: "technical".into(),
                description: "Bug reports and API errors".into(),
                value: None,
            },
        ];

        let ans = backend
            .evaluate_candidates("Customer was double charged on invoice #5512", &q, &candidates)
            .expect("Evaluation should succeed");

        assert_eq!(ans.decision, Some(serde_json::Value::String("billing".into())));
        assert_eq!(ans.question_type, "choice");

        // Verify exact payload forwarded to apfel engine
        let last_req = mock.last_request().expect("Engine must receive GenerateRequest");
        assert_eq!(last_req.use_case, Some("content_tagging".to_string()));
        assert_eq!(last_req.temperature, Some(0.0));
        assert_eq!(last_req.top_p, None);
        assert_eq!(last_req.max_tokens, Some(4));
        assert_eq!(last_req.seed, Some(42));
        assert!(last_req.permissive);
        assert!(last_req.prompt.contains("billing"));
        assert!(last_req.prompt.contains("Customer was double charged"));
    }

    #[test]
    fn test_parse_candidate_choice_edge_cases() {
        let candidates = vec![
            Candidate {
                id: "billing".into(),
                description: "Billing issues".into(),
                value: None,
            },
            Candidate {
                id: "hardware".into(),
                description: "Device broken".into(),
                value: None,
            },
        ];

        // Bracket with trailing punctuation
        assert_eq!(super::parse_candidate_choice("The answer is [billing].", &candidates), "billing");
        // Bracket with leading text
        assert_eq!(super::parse_candidate_choice("Decision: [hardware]", &candidates), "hardware");
        // Insufficient context
        assert_eq!(super::parse_candidate_choice("[__insufficient__]", &candidates), "__insufficient__");
        // Keyword fallback without brackets
        assert_eq!(super::parse_candidate_choice("This seems like a billing matter.", &candidates), "billing");
        // Unmatched fallback
        assert_eq!(super::parse_candidate_choice("Random unrelated chatter", &candidates), "__insufficient__");
    }

    #[test]
    fn test_apfel_neural_backend_score_and_numeric() {
        let mock = Arc::new(apfel::backend::MockEngine::with_response("[5]"));
        let backend = ApfelNeuralBackend::with_engine(mock);

        let q_score = Question::Score(crate::types::ScoreQuestion {
            instructions: "Rate satisfaction 1-5".into(),
            levels: vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
            policy: Policy::default(),
        });
        let candidates = vec![
            Candidate {
                id: "1".into(),
                description: "Poor".into(),
                value: Some(1.0),
            },
            Candidate {
                id: "5".into(),
                description: "Great".into(),
                value: Some(5.0),
            },
        ];

        let ans = backend
            .evaluate_candidates("Amazing service, five stars!", &q_score, &candidates)
            .expect("Score evaluation should succeed");

        assert_eq!(ans.question_type, "score");
        assert_eq!(ans.decision, Some(serde_json::Value::String("5".into())));

        // Test Numeric question as well
        let q_num = Question::Numeric(crate::types::NumericQuestion {
            instructions: "Count total items".into(),
            unit: "items".into(),
            anchors: Vec::new(),
            policy: Policy::default(),
        });
        let ans_num = backend
            .evaluate_candidates("Counted five items", &q_num, &candidates)
            .expect("Numeric evaluation should succeed");
        assert_eq!(ans_num.question_type, "numeric");
        assert_eq!(ans_num.decision, Some(serde_json::Value::String("5".into())));
    }
}
