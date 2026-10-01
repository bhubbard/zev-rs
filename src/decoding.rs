use crate::abstain::is_abstain_candidate;
use crate::calibration::scaled_softmax_slice;
use crate::error::{Result, ZevError};
use crate::types::{
    Candidate, DecisionStatistics, Question, UncertaintyMetrics, ZevAnswer, ABOVE, BELOW, UNKNOWN,
};
use std::collections::BTreeMap;

pub fn generate_candidates(question: &Question) -> Vec<Candidate> {
    let mut list = match question {
        Question::Boolean(b) => vec![
            Candidate {
                id: "false".into(),
                description: b.false_description.clone(),
                value: Some(0.0),
            },
            Candidate {
                id: "true".into(),
                description: b.true_description.clone(),
                value: Some(1.0),
            },
        ],
        Question::Choice(c) => c
            .options
            .iter()
            .map(|o| Candidate {
                id: o.id.clone(),
                description: o.description.clone(),
                value: None,
            })
            .collect(),
        Question::Score(s) => s
            .levels
            .iter()
            .enumerate()
            .map(|(i, lvl)| Candidate {
                id: i.to_string(),
                description: lvl.clone(),
                value: Some(i as f64),
            })
            .collect(),
        Question::Numeric(n) => {
            let mut items = Vec::new();
            for (i, a) in n.anchors.iter().enumerate() {
                items.push(Candidate {
                    id: i.to_string(),
                    description: format!("Approximately {} {}: {}", a.value, n.unit, a.description),
                    value: Some(a.value),
                });
            }
            items.push(Candidate {
                id: BELOW.into(),
                description: format!(
                    "The value is below {} {}.",
                    n.anchors.first().unwrap().value,
                    n.unit
                ),
                value: None,
            });
            items.push(Candidate {
                id: ABOVE.into(),
                description: format!(
                    "The value is above {} {}.",
                    n.anchors.last().unwrap().value,
                    n.unit
                ),
                value: None,
            });
            items
        }
    };

    if question.policy().allow_abstain {
        let has_explicit_abstain = match question {
            Question::Choice(c) => c
                .options
                .iter()
                .any(|o| is_abstain_candidate(&o.id, &o.description)),
            _ => false,
        };
        if !has_explicit_abstain {
            list.push(Candidate {
                id: UNKNOWN.into(),
                description: "Cannot determine the answer: evidence is contradictory or missing."
                    .into(),
                value: None,
            });
        }
    }

    list
}

pub fn summarize_moments(values: &[f64], probs: &[f64]) -> DecisionStatistics {
    let mean: f64 = values.iter().zip(probs.iter()).map(|(&v, &p)| v * p).sum();
    let variance: f64 = values
        .iter()
        .zip(probs.iter())
        .map(|(&v, &p)| p * (v - mean).powi(2))
        .sum();

    let quantile = |q: f64| -> f64 {
        let mut cumulative = 0.0;
        for (&v, &p) in values.iter().zip(probs.iter()) {
            cumulative += p;
            if cumulative >= q {
                return v;
            }
        }
        *values.last().unwrap_or(&0.0)
    };

    let mut q_map = BTreeMap::new();
    q_map.insert("p10".into(), quantile(0.1));
    q_map.insert("p90".into(), quantile(0.9));

    DecisionStatistics {
        mean,
        stddev: variance.sqrt(),
        median: quantile(0.5),
        quantiles: q_map,
    }
}

/// Applies RLCD-inspired ordinal smoothing to a probability distribution over an ordered scale.
///
/// For ordinal levels $0, \dots, K-1$, sharp categorical predictions often assign zero mass to
/// adjacent levels, creating cliff-like risk profiles. Ordinal smoothing applies a tridiagonal kernel:
///
/// `\tilde{p}_i = (1 - 2\alpha) p_i + \alpha p_{i-1} + \alpha p_{i+1}` for interior levels,
/// `\tilde{p}_0 = (1 - \alpha) p_0 + \alpha p_1`,
/// `\tilde{p}_{K-1} = (1 - \alpha) p_{K-1} + \alpha p_{K-2}`.
///
/// This kernel is strictly mass-conserving: `\sum \tilde{p}_i = \sum p_i = 1.0`.
pub fn smooth_ordinal_probabilities(probs: &[f64], alpha: f64) -> Vec<f64> {
    let k = probs.len();
    if k <= 1 || alpha <= 0.0 {
        return probs.to_vec();
    }
    let alpha = alpha.clamp(0.0, 0.49);
    let mut smoothed = vec![0.0; k];

    if k == 2 {
        smoothed[0] = (1.0 - alpha) * probs[0] + alpha * probs[1];
        smoothed[1] = alpha * probs[0] + (1.0 - alpha) * probs[1];
        return smoothed;
    }

    // Boundary at i = 0
    smoothed[0] = (1.0 - alpha) * probs[0] + alpha * probs[1];

    // Interior points
    for i in 1..k - 1 {
        smoothed[i] = (1.0 - 2.0 * alpha) * probs[i] + alpha * probs[i - 1] + alpha * probs[i + 1];
    }

    // Boundary at i = k - 1
    smoothed[k - 1] = (1.0 - alpha) * probs[k - 1] + alpha * probs[k - 2];

    smoothed
}

pub fn decode_decision(
    question: &Question,
    candidates: &[Candidate],
    logits: &[f64],
    temperature: f64,
) -> Result<ZevAnswer> {
    if logits.len() != candidates.len() {
        return Err(ZevError::DecodingError(format!(
            "Logit count ({}) != candidate count ({})",
            logits.len(),
            candidates.len()
        )));
    }

    let n = candidates.len();
    let mut probs_buf = [0.0; 128];
    let mut probs_vec;
    let probs: &[f64] = if n <= 128 {
        scaled_softmax_slice(logits, temperature, &mut probs_buf[..n])?;
        &probs_buf[..n]
    } else {
        probs_vec = vec![0.0; n];
        scaled_softmax_slice(logits, temperature, &mut probs_vec)?;
        &probs_vec
    };

    let mut prob_map = BTreeMap::new();
    let mut logit_map = BTreeMap::new();

    for (c, (&p, &l)) in candidates.iter().zip(probs.iter().zip(logits.iter())) {
        prob_map.insert(c.id.clone(), p);
        logit_map.insert(c.id.clone(), l);
    }

    let is_unavailable = |c: &Candidate| -> bool {
        c.id == UNKNOWN
            || c.id == BELOW
            || c.id == ABOVE
            || is_abstain_candidate(&c.id, &c.description)
    };

    let unavailable_prob: f64 = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| is_unavailable(c))
        .map(|(_, &p)| p)
        .sum();

    let available_prob: f64 = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| !is_unavailable(c))
        .map(|(_, &p)| p)
        .sum();

    let top_idx = (0..candidates.len())
        .max_by(|&a, &b| {
            let pa = probs[a];
            let pb = probs[b];
            pa.partial_cmp(&pb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| candidates[b].id.cmp(&candidates[a].id))
        })
        .unwrap_or(0);

    let winner = &candidates[top_idx];
    let top_prob = probs[top_idx];

    let entropy_nats: f64 = probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln())
        .sum();

    let concentration = if probs.len() > 1 {
        (1.0 - (entropy_nats / (probs.len() as f64).ln())).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let policy = question.policy();
    let mut status = "ok".to_string();

    let margin = if probs.len() >= 2 {
        let mut top1 = f64::NEG_INFINITY;
        let mut top2 = f64::NEG_INFINITY;
        for &p in probs {
            if p > top1 {
                top2 = top1;
                top1 = p;
            } else if p > top2 {
                top2 = p;
            }
        }
        top1 - top2
    } else {
        1.0
    };

    // Entropy-scaled dynamic confidence threshold:
    let effective_min_top_prob = if policy.min_top_probability > 0.0 {
        let uniform_prior = 1.0 / (candidates.len().max(1) as f64);
        let n_scale = (4.0 / (candidates.len() as f64 + 3.0)).sqrt();
        (policy.min_top_probability * n_scale).max(uniform_prior * 1.5)
    } else {
        0.0
    };

    let below_prob = prob_map.get(BELOW).copied().unwrap_or(0.0);
    let above_prob = prob_map.get(ABOVE).copied().unwrap_or(0.0);
    let unknown_prob = prob_map.get(UNKNOWN).copied().unwrap_or(0.0);
    let out_of_range_prob = below_prob + above_prob;

    // Out-of-range detection (R4):
    // If the argmax/majority probability mass is on __below_range__ or __above_range__,
    // the status must be reported as out_of_range, NOT ok (independent of allow_abstain).
    if winner.id == BELOW || winner.id == ABOVE || out_of_range_prob > available_prob {
        status = "out_of_range".into();
    } else if policy.allow_abstain {
        if is_unavailable(winner) || unavailable_prob >= policy.max_unavailable_probability {
            status = if out_of_range_prob > unknown_prob {
                "out_of_range".into()
            } else {
                "insufficient_evidence".into()
            };
        } else if (policy.min_top_probability > 0.0 && top_prob < effective_min_top_prob)
            || (policy.min_top_probability > 0.0
                && candidates.len() > 2
                && margin < 0.10
                && top_prob < effective_min_top_prob + 0.10)
        {
            status = "uncertain".into();
        }
    }

    let valid_cands: Vec<(&Candidate, f64)> = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| !is_unavailable(c))
        .map(|(c, &p)| (c, p))
        .collect();

    let cond_probs: Option<Vec<f64>> = if available_prob > 0.0 {
        Some(
            valid_cands
                .iter()
                .map(|(_, p)| p / available_prob)
                .collect(),
        )
    } else {
        None
    };

    let q_type_str = match question {
        Question::Boolean(_) => "boolean",
        Question::Choice(_) => "choice",
        Question::Score(_) => "score",
        Question::Numeric(_) => "numeric",
    };

    let mut answer = ZevAnswer {
        question_type: q_type_str.into(),
        status: status.clone(),
        decision: None,
        confidence: top_prob,
        probabilities: prob_map,
        logits: logit_map,
        uncertainty: UncertaintyMetrics {
            top_probability: top_prob,
            entropy_nats,
            concentration,
            unavailable_probability: unavailable_prob,
            margin: Some(margin),
            quantile_spread: None,
        },
        statistics: None,
        expected_value: None,
        temperature,
        source: Some("native".to_string()),
    };

    // Pre-flight sanity guardrails on numeric question anchors
    if let Question::Numeric(n) = question {
        let anchor_vals: Vec<f64> = n.anchors.iter().map(|a| a.value).collect();
        let guardrail = crate::types::check_numeric_guardrails(
            &anchor_vals,
            &crate::types::NumericGuardrailConfig::default(),
        );
        if guardrail.should_abstain {
            status = "insufficient_evidence".into();
            answer.status = status.clone();
        }
    }

    // Compute moments and quantile_spread for continuous/ordinal questions if valid candidates exist
    if matches!(question, Question::Score(_) | Question::Numeric(_)) {
        let values: Vec<f64> = valid_cands.iter().filter_map(|(c, _)| c.value).collect();
        if let Some(ref cp) = cond_probs {
            if !values.is_empty() {
                let effective_cp = if let Question::Score(s) = question {
                    if let Some(alpha) = s.ordinal_smoothing {
                        smooth_ordinal_probabilities(cp, alpha)
                    } else {
                        cp.clone()
                    }
                } else {
                    cp.clone()
                };
                let stats = summarize_moments(&values, &effective_cp);
                let spread = (stats.quantiles["p90"] - stats.quantiles["p10"]).max(0.0);
                answer.uncertainty.quantile_spread = Some(spread);
                if status == "ok" {
                    answer.expected_value = Some(stats.mean);
                    answer.decision = Some(serde_json::json!(stats.mean));
                    answer.statistics = Some(stats);
                }
            }
        }
    }

    if status == "ok" {
        match question {
            Question::Boolean(_) => {
                answer.decision = Some(serde_json::Value::Bool(winner.id == "true"));
            }
            Question::Choice(_) => {
                answer.decision = Some(serde_json::Value::String(winner.id.clone()));
            }
            Question::Score(_) | Question::Numeric(_) => {}
        }
    } else if status == "insufficient_evidence"
        && winner.id != UNKNOWN
        && winner.id != BELOW
        && winner.id != ABOVE
        && matches!(question, Question::Choice(_))
    {
        answer.decision = Some(serde_json::Value::String(winner.id.clone()));
    }

    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChoiceQuestion, OptionDef, Policy};

    #[test]
    fn test_decoding_mismatched_lengths() {
        let q = Question::Choice(ChoiceQuestion {
            instructions: "test".into(),
            options: vec![
                OptionDef {
                    id: "a".into(),
                    description: "A".into(),
                },
                OptionDef {
                    id: "b".into(),
                    description: "B".into(),
                },
            ],
            policy: Default::default(),
        });
        let cands = generate_candidates(&q);
        assert!(decode_decision(&q, &cands, &[1.0], 1.0).is_err());
    }

    #[test]
    fn test_decoding_large_option_buffer() {
        let options: Vec<OptionDef> = (0..130)
            .map(|i| OptionDef {
                id: format!("opt_{i}"),
                description: format!("option {i}"),
            })
            .collect();
        let q = Question::Choice(ChoiceQuestion {
            instructions: "test".into(),
            options,
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });
        let cands = generate_candidates(&q);
        let logits = vec![0.5; cands.len()];
        let ans = decode_decision(&q, &cands, &logits, 1.0).unwrap();
        assert_eq!(ans.status, "ok");
    }

    #[test]
    fn test_decoding_uncertain_status_margin() {
        let q = Question::Choice(ChoiceQuestion {
            instructions: "test".into(),
            options: vec![
                OptionDef {
                    id: "a".into(),
                    description: "Option A".into(),
                },
                OptionDef {
                    id: "b".into(),
                    description: "Option B".into(),
                },
                OptionDef {
                    id: "c".into(),
                    description: "Option C".into(),
                },
            ],
            policy: Policy {
                allow_abstain: true,
                min_top_probability: 0.35,
                max_unavailable_probability: 0.9,
                max_slots: None,
            },
        });
        let cands = generate_candidates(&q);
        // Tie logits producing narrow margin (<0.10)
        let logits = vec![1.01, 1.00, 0.99, -5.0];
        let ans = decode_decision(&q, &cands, &logits, 1.0).unwrap();
        assert_eq!(ans.status, "uncertain");
    }

    #[test]
    fn test_summarize_moments_edge_quantile() {
        let values = [1.0, 2.0];
        let probs = [0.0, 0.0];
        let stats = summarize_moments(&values, &probs);
        assert_eq!(stats.median, 2.0);
    }

    #[test]
    fn test_numeric_out_of_range_without_abstain_p07() {
        use crate::types::{Anchor, NumericQuestion};

        // P07: Numeric question with allow_abstain: false
        let q = Question::Numeric(NumericQuestion {
            instructions: "Estimate valuation".into(),
            unit: "M_USD".into(),
            anchors: vec![
                Anchor {
                    value: 0.0,
                    description: "Seed stage".into(),
                },
                Anchor {
                    value: 100.0,
                    description: "Growth stage".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });
        let candidates = generate_candidates(&q);
        let mut logits = vec![0.0; candidates.len()];
        let above_idx = candidates.iter().position(|c| c.id == ABOVE).unwrap();
        logits[above_idx] = 10.0; // ABOVE heavily dominates

        let ans = decode_decision(&q, &candidates, &logits, 1.0).unwrap();
        assert_eq!(ans.status, "out_of_range");
        assert!(ans.decision.is_none());
        assert!(ans.expected_value.is_none());
    }

    #[test]
    fn test_numeric_below_range_without_abstain() {
        use crate::types::{Anchor, NumericQuestion};

        let q = Question::Numeric(NumericQuestion {
            instructions: "Estimate valuation".into(),
            unit: "M_USD".into(),
            anchors: vec![
                Anchor {
                    value: 10.0,
                    description: "Seed".into(),
                },
                Anchor {
                    value: 50.0,
                    description: "Series A".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });
        let candidates = generate_candidates(&q);
        let mut logits = vec![0.0; candidates.len()];
        let below_idx = candidates.iter().position(|c| c.id == BELOW).unwrap();
        logits[below_idx] = 10.0; // BELOW heavily dominates

        let ans = decode_decision(&q, &candidates, &logits, 1.0).unwrap();
        assert_eq!(ans.status, "out_of_range");
        assert!(ans.decision.is_none());
        assert!(ans.expected_value.is_none());
    }

    #[test]
    fn test_numeric_majority_out_of_range_without_abstain() {
        use crate::types::{Anchor, NumericQuestion};

        let q = Question::Numeric(NumericQuestion {
            instructions: "Estimate valuation".into(),
            unit: "M_USD".into(),
            anchors: vec![
                Anchor {
                    value: 10.0,
                    description: "Seed".into(),
                },
                Anchor {
                    value: 50.0,
                    description: "Series A".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });
        let candidates = generate_candidates(&q);
        let mut logits = vec![0.0; candidates.len()];
        let below_idx = candidates.iter().position(|c| c.id == BELOW).unwrap();
        let above_idx = candidates.iter().position(|c| c.id == ABOVE).unwrap();
        // Combined below and above mass is majority
        logits[below_idx] = 3.0;
        logits[above_idx] = 3.0;

        let ans = decode_decision(&q, &candidates, &logits, 1.0).unwrap();
        assert_eq!(ans.status, "out_of_range");
        assert!(ans.decision.is_none());
        assert!(ans.expected_value.is_none());
    }

    #[test]
    fn test_smooth_ordinal_probabilities() {
        // Mass conservation on 4 ordinal levels
        let original = vec![0.0, 1.0, 0.0, 0.0]; // Spike at index 1
        let smoothed = smooth_ordinal_probabilities(&original, 0.15);
        let sum: f64 = smoothed.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // Mass dispersed to neighbors: index 0 gets 0.15, index 2 gets 0.15, index 1 retains 0.70
        assert!((smoothed[0] - 0.15).abs() < 1e-6);
        assert!((smoothed[1] - 0.70).abs() < 1e-6);
        assert!((smoothed[2] - 0.15).abs() < 1e-6);
        assert_eq!(smoothed[3], 0.0);

        // Boundary test (spike at 0)
        let orig_b0 = vec![1.0, 0.0, 0.0];
        let smooth_b0 = smooth_ordinal_probabilities(&orig_b0, 0.10);
        assert!((smooth_b0.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((smooth_b0[0] - 0.90).abs() < 1e-6);
        assert!((smooth_b0[1] - 0.10).abs() < 1e-6);
        assert_eq!(smooth_b0[2], 0.0);

        // 2 elements
        let orig_2 = vec![0.8, 0.2];
        let smooth_2 = smooth_ordinal_probabilities(&orig_2, 0.10);
        assert!((smooth_2.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // Zero / negative alpha or single element
        assert_eq!(smooth_ordinal_probabilities(&[1.0], 0.2), vec![1.0]);
        assert_eq!(smooth_ordinal_probabilities(&[0.5, 0.5], 0.0), vec![0.5, 0.5]);
    }

    #[test]
    fn test_score_decoding_with_ordinal_smoothing() {
        use crate::types::ScoreQuestion;

        let q_sharp = Question::Score(ScoreQuestion {
            instructions: "Rate severity".into(),
            levels: vec!["Low".into(), "Medium".into(), "High".into()],
            ordinal_smoothing: None,
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });

        let q_smooth = Question::Score(ScoreQuestion {
            instructions: "Rate severity".into(),
            levels: vec!["Low".into(), "Medium".into(), "High".into()],
            ordinal_smoothing: Some(0.20),
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });

        let candidates = generate_candidates(&q_sharp);
        // Sharp prediction favoring index 2 ("High")
        let logits = vec![-5.0, -5.0, 10.0];

        let ans_sharp = decode_decision(&q_sharp, &candidates, &logits, 1.0).unwrap();
        let ans_smooth = decode_decision(&q_smooth, &candidates, &logits, 1.0).unwrap();

        // Smoothed expected value pulls slightly inward from the boundary
        let ev_sharp = ans_sharp.expected_value.unwrap();
        let ev_smooth = ans_smooth.expected_value.unwrap();
        assert!(ev_smooth < ev_sharp);
        assert!(ev_smooth > 1.5);
    }

    #[test]
    fn test_ordinal_smoothing_symmetry_and_expectation_conservation() {
        // Symmetric 5-point distribution: [0.05, 0.20, 0.50, 0.20, 0.05]
        let original = vec![0.05, 0.20, 0.50, 0.20, 0.05];
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let smoothed = smooth_ordinal_probabilities(&original, 0.15);

        // 1. Mass conservation
        let sum: f64 = smoothed.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);

        // 2. Exact symmetry preservation: p[0] == p[4] and p[1] == p[3]
        assert!((smoothed[0] - smoothed[4]).abs() < 1e-9);
        assert!((smoothed[1] - smoothed[3]).abs() < 1e-9);

        // 3. Expected value conservation on symmetric distribution:
        // E[X] = 2.0 before and after smoothing
        let stats_orig = summarize_moments(&values, &original);
        let stats_smooth = summarize_moments(&values, &smoothed);
        assert!((stats_orig.mean - 2.0).abs() < 1e-9);
        assert!((stats_smooth.mean - 2.0).abs() < 1e-9);

        // Variance should increase slightly due to diffusion
        assert!(stats_smooth.stddev > stats_orig.stddev);
    }

    #[test]
    fn test_ordinal_smoothing_monotonicity_preservation() {
        // Monotonic ramp: [0.10, 0.20, 0.30, 0.40]
        let ramp = vec![0.10, 0.20, 0.30, 0.40];
        let smoothed = smooth_ordinal_probabilities(&ramp, 0.10);

        assert!((smoothed.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // Monotonic order must remain strictly increasing
        for w in smoothed.windows(2) {
            assert!(w[0] < w[1], "Ordinal smoothing must preserve monotonic rank order");
        }
    }

    #[test]
    fn test_score_question_builder_pattern() {
        use crate::types::ScoreQuestion;
        let q = ScoreQuestion::new("Rate severity", vec!["Low".into(), "High".into()])
            .with_ordinal_smoothing(0.12);

        assert_eq!(q.instructions, "Rate severity");
        assert_eq!(q.levels.len(), 2);
        assert_eq!(q.ordinal_smoothing, Some(0.12));
    }
}
