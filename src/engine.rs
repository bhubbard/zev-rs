use std::collections::BTreeMap;
use std::time::Instant;
use crate::calibration::resolve_temperature;
use crate::decoding::{decode_decision, generate_candidates};
use crate::error::Result;
use crate::order_invariant::{compute_order_invariant_logits_with_context, PremiseContext};
use crate::preprocessor::preprocess_state;
use crate::shortlist::shortlist_options;
use crate::types::{
    ChoiceQuestion, ExecutionTiming, OptionDef, Policy, Question,
    SystemOneRequest, SystemOneResponse, WireQuestion, WireUsage, ZevAnswer,
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

        let state_borrowed: std::borrow::Cow<str> = match &req.state {
            serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(serde_json::to_string(other)?),
        };

        // 1. Text Preprocessing & Temporal Grounding
        let preprocessed_state = preprocess_state(&state_borrowed, req.enable_temporal_facts);
        let eval_start = Instant::now();

        // Pre-tokenize premise context once for all questions
        let ctx = PremiseContext::new(&preprocessed_state);

        let temp = resolve_temperature(req.temperature.or(Some(self.default_temperature)))?;
        let mut answers = BTreeMap::new();

        // 2. Parallel / Multi-Task Question Scoring
        for (key, question) in &req.questions {
            question.validate(key)?;

            // Shortlisting if choice options exceed MAX_SLOTS
            let shortlisted_storage;
            let final_question: &Question = match question {
                Question::Choice(c) if c.options.len() > MAX_SLOTS - 1 => {
                    let max_keep = if c.policy.allow_abstain { MAX_SLOTS - 1 } else { MAX_SLOTS };
                    let shortlisted = shortlist_options(&c.options, &preprocessed_state, max_keep);
                    shortlisted_storage = Some(Question::Choice(ChoiceQuestion {
                        instructions: c.instructions.clone(),
                        options: shortlisted,
                        policy: c.policy.clone(),
                    }));
                    shortlisted_storage.as_ref().unwrap()
                }
                _ => question,
            };

            let candidates = generate_candidates(final_question);

            // 3. 100% Order-Invariant Isolated Logit Scoring reusing pre-tokenized context
            let logits = compute_order_invariant_logits_with_context(&ctx, &candidates);

            // 4. Calibrated Decoding & Moment Statistics
            let answer = decode_decision(final_question, &candidates, &logits, temp)?;
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
        let state_borrowed: std::borrow::Cow<str> = match &req.state {
            serde_json::Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(serde_json::to_string(other)?),
        };

        let preprocessed_state = preprocess_state(&state_borrowed, true);
        let ctx = PremiseContext::new(&preprocessed_state);
        let temp = self.default_temperature;

        let mut wire_answers = BTreeMap::new();
        let mut input_tokens = preprocessed_state.len() / 4;

        let mut logits_buf = [0.0; 16];
        let mut probs_buf = [0.0; 16];

        for (key, q) in &req.questions {
            input_tokens += 20; // branch overhead
            match q {
                WireQuestion::Noul(n) => {
                    let true_desc = if let Some(ref c) = n.criteria {
                        if let Some(ref t) = c.true_criterion {
                            t.as_str().unwrap_or("")
                        } else {
                            "Yes. Supports affirmative answer."
                        }
                    } else {
                        "Yes. Supports affirmative answer."
                    };
                    let false_desc = if let Some(ref c) = n.criteria {
                        if let Some(ref f) = c.false_criterion {
                            f.as_str().unwrap_or("")
                        } else {
                            "No. Supports negative answer."
                        }
                    } else {
                        "No. Supports negative answer."
                    };

                    logits_buf[0] = ctx.score_candidate_raw("false", false_desc);
                    logits_buf[1] = ctx.score_candidate_raw("true", true_desc);
                    crate::calibration::scaled_softmax_slice(&logits_buf[..2], temp, &mut probs_buf[..2])?;

                    let p_true = probs_buf[1];
                    wire_answers.insert(key.clone(), serde_json::json!({
                        "type": "noul",
                        "noul": p_true,
                    }));
                }
                WireQuestion::Choice(c) => {
                    let n = c.criteria.len();
                    let mut prob_map = BTreeMap::new();
                    let mut best_id = "";
                    let mut best_prob = -1.0;
                    let mut second_prob = -1.0;

                    if n <= 16 {
                        let mut idx = 0;
                        for (name, detail) in &c.criteria {
                            let desc = detail.as_ref().and_then(|d| d.as_str()).unwrap_or(name.as_str());
                            logits_buf[idx] = ctx.score_candidate_raw(name, desc);
                            idx += 1;
                        }
                        crate::calibration::scaled_softmax_slice(&logits_buf[..n], temp, &mut probs_buf[..n])?;

                        for ((name, _), &p) in c.criteria.iter().zip(probs_buf[..n].iter()) {
                            prob_map.insert(name.clone(), p);
                            if p > best_prob {
                                second_prob = best_prob;
                                best_prob = p;
                                best_id = name.as_str();
                            } else if p > second_prob {
                                second_prob = p;
                            }
                        }
                    } else {
                        let mut l_vec = Vec::with_capacity(n);
                        for (name, detail) in &c.criteria {
                            let desc = detail.as_ref().and_then(|d| d.as_str()).unwrap_or(name.as_str());
                            l_vec.push(ctx.score_candidate_raw(name, desc));
                        }
                        let mut p_vec = vec![0.0; n];
                        crate::calibration::scaled_softmax_slice(&l_vec, temp, &mut p_vec)?;

                        for ((name, _), &p) in c.criteria.iter().zip(p_vec.iter()) {
                            prob_map.insert(name.clone(), p);
                            if p > best_prob {
                                second_prob = best_prob;
                                best_prob = p;
                                best_id = name.as_str();
                            } else if p > second_prob {
                                second_prob = p;
                            }
                        }
                    }

                    let confidence = if second_prob < 0.0 { best_prob } else { (best_prob - second_prob).clamp(0.0, 1.0) };

                    wire_answers.insert(key.clone(), serde_json::json!({
                        "type": "choice",
                        "choice": best_id,
                        "probabilities": prob_map,
                        "confidence": confidence,
                    }));
                }
                WireQuestion::Score(s) => {
                    let n = s.criteria.len();
                    let mut legend = BTreeMap::new();
                    let mut prob_map = BTreeMap::new();
                    let mut expected_val = 0.0;
                    let mut best_prob = -1.0;
                    let mut second_prob = -1.0;

                    const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

                    if n <= 16 {
                        for (idx, lvl) in s.criteria.iter().enumerate() {
                            let desc = lvl.as_str().unwrap_or("");
                            let id_str = if idx < 10 { DIGITS[idx] } else { "" };
                            logits_buf[idx] = ctx.score_candidate_raw(id_str, desc);
                        }
                        crate::calibration::scaled_softmax_slice(&logits_buf[..n], temp, &mut probs_buf[..n])?;

                        for (idx, (lvl, &p)) in s.criteria.iter().zip(probs_buf[..n].iter()).enumerate() {
                            let id_str = if idx < 10 { DIGITS[idx].to_string() } else { idx.to_string() };
                            let desc = lvl.as_str().unwrap_or(&lvl.to_string()).to_string();
                            legend.insert(id_str.clone(), desc);
                            prob_map.insert(id_str, p);
                            expected_val += idx as f64 * p;

                            if p > best_prob {
                                second_prob = best_prob;
                                best_prob = p;
                            } else if p > second_prob {
                                second_prob = p;
                            }
                        }
                    } else {
                        let mut l_vec = Vec::with_capacity(n);
                        for (idx, lvl) in s.criteria.iter().enumerate() {
                            let desc = lvl.as_str().unwrap_or("");
                            let id_str = if idx < 10 { DIGITS[idx] } else { "" };
                            l_vec.push(ctx.score_candidate_raw(id_str, desc));
                        }
                        let mut p_vec = vec![0.0; n];
                        crate::calibration::scaled_softmax_slice(&l_vec, temp, &mut p_vec)?;

                        for (idx, (lvl, &p)) in s.criteria.iter().zip(p_vec.iter()).enumerate() {
                            let id_str = if idx < 10 { DIGITS[idx].to_string() } else { idx.to_string() };
                            let desc = lvl.as_str().unwrap_or(&lvl.to_string()).to_string();
                            legend.insert(id_str.clone(), desc);
                            prob_map.insert(id_str, p);
                            expected_val += idx as f64 * p;

                            if p > best_prob {
                                second_prob = best_prob;
                                best_prob = p;
                            } else if p > second_prob {
                                second_prob = p;
                            }
                        }
                    }

                    let confidence = if second_prob < 0.0 { best_prob } else { (best_prob - second_prob).clamp(0.0, 1.0) };

                    wire_answers.insert(key.clone(), serde_json::json!({
                        "type": "score",
                        "score": expected_val,
                        "legend": legend,
                        "probabilities": prob_map,
                        "confidence": confidence,
                    }));
                }
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
