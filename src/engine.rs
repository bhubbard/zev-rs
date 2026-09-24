use std::collections::BTreeMap;
use std::time::Instant;
use crate::calibration::resolve_temperature;
use crate::decoding::{decode_decision, generate_candidates};
use crate::error::Result;
use crate::order_invariant::compute_order_invariant_logits;
use crate::preprocessor::preprocess_state;
use crate::shortlist::shortlist_options;
use crate::types::{
    BooleanQuestion, ChoiceQuestion, ExecutionTiming, OptionDef, Policy, Question,
    ScoreQuestion, SystemOneRequest, SystemOneResponse, WireQuestion, WireUsage, ZevAnswer,
    ZevRequest, ZevResponse, DEFAULT_MODEL, MAX_SLOTS,
};

pub struct ZevEngine {
    pub default_temperature: f64,
}

impl Default for ZevEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ZevEngine {
    pub fn new(temperature: Option<f64>) -> Self {
        let temp = resolve_temperature(temperature).unwrap_or(2.179078721266035);
        Self {
            default_temperature: temp,
        }
    }

    /// Evaluates a native ZevRequest with complete features
    pub fn evaluate(&self, req: &ZevRequest) -> Result<ZevResponse> {
        let start = Instant::now();

        let state_raw = if let Some(s) = req.state.as_str() {
            s.to_string()
        } else {
            serde_json::to_string(&req.state)?
        };

        // 1. Text Preprocessing & Temporal Grounding
        let preprocessed_state = preprocess_state(&state_raw, req.enable_temporal_facts);
        let eval_start = Instant::now();

        let temp = resolve_temperature(req.temperature.or(Some(self.default_temperature)))?;
        let mut answers = BTreeMap::new();

        // 2. Parallel / Multi-Task Question Scoring
        for (key, question) in &req.questions {
            question.validate(key)?;

            // Shortlisting if choice options exceed MAX_SLOTS
            let final_question = match question {
                Question::Choice(c) if c.options.len() > MAX_SLOTS - 1 => {
                    let max_keep = if c.policy.allow_abstain { MAX_SLOTS - 1 } else { MAX_SLOTS };
                    let shortlisted = shortlist_options(&c.options, &preprocessed_state, max_keep);
                    Question::Choice(ChoiceQuestion {
                        instructions: c.instructions.clone(),
                        options: shortlisted,
                        policy: c.policy.clone(),
                    })
                }
                _ => question.clone(),
            };

            let candidates = generate_candidates(&final_question);

            // 3. 100% Order-Invariant Isolated Logit Scoring
            let logits = compute_order_invariant_logits(&preprocessed_state, &candidates);

            // 4. Calibrated Decoding & Moment Statistics
            let answer = decode_decision(&final_question, &candidates, &logits, temp)?;
            answers.insert(key.clone(), answer);
        }

        let eval_micros = eval_start.elapsed().as_secs_f64() * 1_000_000.0;
        let total_micros = start.elapsed().as_secs_f64() * 1_000_000.0;

        Ok(ZevResponse {
            model: req.model.clone().unwrap_or_else(|| DEFAULT_MODEL.into()),
            answers,
            execution: ExecutionTiming {
                total_micros,
                eval_micros,
                shared_prefix_tokens: preprocessed_state.len() / 4,
            },
        })
    }

    /// Evaluates a TypeSafe SystemOneRequest, ensuring 100% wire-protocol drop-in compatibility
    pub fn evaluate_system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        let mut native_questions = BTreeMap::new();
        let mut options_order: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let wire_policy = Policy {
            allow_abstain: false,
            max_unavailable_probability: 0.5,
            min_top_probability: 0.0,
        };

        for (key, q) in &req.questions {
            let nq = match q {
                WireQuestion::Noul(n) => {
                    let mut true_desc = "Yes. Supports affirmative answer.".to_string();
                    let mut false_desc = "No. Supports negative answer.".to_string();

                    if let Some(ref c) = n.criteria {
                        if let Some(ref t) = c.true_criterion {
                            true_desc = format!("Yes. {}", t.as_str().unwrap_or(&t.to_string()));
                        }
                        if let Some(ref f) = c.false_criterion {
                            false_desc = format!("No. {}", f.as_str().unwrap_or(&f.to_string()));
                        }
                    }

                    Question::Boolean(BooleanQuestion {
                        instructions: n.instructions.as_str().unwrap_or(&n.instructions.to_string()).to_string(),
                        true_description: true_desc,
                        false_description: false_desc,
                        policy: wire_policy.clone(),
                    })
                }
                WireQuestion::Choice(c) => {
                    let mut opts = Vec::new();
                    let mut names = Vec::new();

                    for (name, detail) in &c.criteria {
                        names.push(name.clone());
                        let desc = match detail {
                            Some(d) => format!("{}: {}", name, d.as_str().unwrap_or(&d.to_string())),
                            None => name.clone(),
                        };
                        opts.push(OptionDef {
                            id: name.clone(),
                            description: desc,
                        });
                    }

                    options_order.insert(key.clone(), names);
                    Question::Choice(ChoiceQuestion {
                        instructions: c.instructions.as_str().unwrap_or(&c.instructions.to_string()).to_string(),
                        options: opts,
                        policy: wire_policy.clone(),
                    })
                }
                WireQuestion::Score(s) => {
                    let levels: Vec<String> = s
                        .criteria
                        .iter()
                        .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                        .collect();

                    Question::Score(ScoreQuestion {
                        instructions: s.instructions.as_str().unwrap_or(&s.instructions.to_string()).to_string(),
                        levels,
                        policy: wire_policy.clone(),
                    })
                }
            };
            native_questions.insert(key.clone(), nq);
        }

        let zev_req = ZevRequest {
            state: req.state.clone(),
            questions: native_questions,
            model: Some(req.model.clone()),
            temperature: Some(self.default_temperature),
            enable_temporal_facts: true,
        };

        let zev_resp = self.evaluate(&zev_req)?;
        let mut wire_answers = BTreeMap::new();
        let mut input_tokens = zev_resp.execution.shared_prefix_tokens;

        for (key, q) in &req.questions {
            if let Some(ans) = zev_resp.answers.get(key) {
                input_tokens += 20; // branch overhead
                let val = match q {
                    WireQuestion::Noul(_) => {
                        let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);
                        serde_json::json!({
                            "type": "noul",
                            "noul": p_true,
                        })
                    }
                    WireQuestion::Choice(_) => {
                        let choice_str = match &ans.decision {
                            Some(serde_json::Value::String(s)) => s.clone(),
                            _ => "".into(),
                        };
                        serde_json::json!({
                            "type": "choice",
                            "choice": choice_str,
                            "probabilities": ans.probabilities,
                            "confidence": ans.confidence,
                        })
                    }
                    WireQuestion::Score(s) => {
                        let mut legend = BTreeMap::new();
                        for (i, lvl) in s.criteria.iter().enumerate() {
                            legend.insert(i.to_string(), lvl.as_str().unwrap_or(&lvl.to_string()).to_string());
                        }
                        serde_json::json!({
                            "type": "score",
                            "score": ans.expected_value.unwrap_or(0.0),
                            "legend": legend,
                            "probabilities": ans.probabilities,
                            "confidence": ans.confidence,
                        })
                    }
                };
                wire_answers.insert(key.clone(), val);
            }
        }

        Ok(SystemOneResponse {
            model: req.model.clone(),
            answers: wire_answers,
            usage: WireUsage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }

    /// Fast confidence gate helper
    pub fn confidence_gate(&self, state: &str, question: Question, threshold: f64) -> Result<(bool, ZevAnswer)> {
        let mut questions = BTreeMap::new();
        questions.insert("gate".into(), question);
        let req = ZevRequest {
            state: serde_json::json!(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: true,
        };
        let resp = self.evaluate(&req)?;
        let ans = resp.answers.get("gate").cloned().unwrap();
        let pass = ans.confidence >= threshold && ans.status == "ok";
        Ok((pass, ans))
    }

    /// Fast route helper
    pub fn route(&self, state: &str, routes: BTreeMap<String, String>) -> Result<(String, f64)> {
        let options = routes
            .into_iter()
            .map(|(id, desc)| OptionDef { id, description: desc })
            .collect();
        let question = Question::Choice(ChoiceQuestion {
            instructions: "Route to the best destination".into(),
            options,
            policy: Policy { allow_abstain: false, ..Default::default() },
        });
        let mut questions = BTreeMap::new();
        questions.insert("route".into(), question);
        let req = ZevRequest {
            state: serde_json::json!(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
        };
        let resp = self.evaluate(&req)?;
        let ans = resp.answers.get("route").unwrap();
        let choice = match &ans.decision {
            Some(serde_json::Value::String(s)) => s.clone(),
            _ => "".into(),
        };
        let prob = ans.probabilities.get(&choice).copied().unwrap_or(0.0);
        Ok((choice, prob))
    }
}
