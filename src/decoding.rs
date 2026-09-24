use std::collections::BTreeMap;
use crate::calibration::scaled_softmax_slice;
use crate::error::{Result, ZevError};
use crate::types::{
    Candidate, DecisionStatistics, Question, UncertaintyMetrics, ZevAnswer,
    ABOVE, BELOW, UNKNOWN,
};

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
                description: format!("The value is below {} {}.", n.anchors.first().unwrap().value, n.unit),
                value: None,
            });
            items.push(Candidate {
                id: ABOVE.into(),
                description: format!("The value is above {} {}.", n.anchors.last().unwrap().value, n.unit),
                value: None,
            });
            items
        }
    };

    if question.policy().allow_abstain {
        list.push(Candidate {
            id: UNKNOWN.into(),
            description: "Cannot determine the answer: evidence is contradictory or missing.".into(),
            value: None,
        });
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

    let unavailable_ids = [UNKNOWN, BELOW, ABOVE];
    let unavailable_prob: f64 = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| unavailable_ids.contains(&c.id.as_str()))
        .map(|(_, &p)| p)
        .sum();

    let available_prob: f64 = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| !unavailable_ids.contains(&c.id.as_str()))
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

    let mut sorted_probs: Vec<f64> = probs.to_vec();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = if sorted_probs.len() >= 2 {
        sorted_probs[0] - sorted_probs[1]
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

    if policy.allow_abstain {
        if unavailable_ids.contains(&winner.id.as_str()) || unavailable_prob >= policy.max_unavailable_probability {
            let below_prob = prob_map.get(BELOW).copied().unwrap_or(0.0);
            let above_prob = prob_map.get(ABOVE).copied().unwrap_or(0.0);
            let unknown_prob = prob_map.get(UNKNOWN).copied().unwrap_or(0.0);

            status = if (below_prob + above_prob) > unknown_prob {
                "out_of_range".into()
            } else {
                "insufficient_evidence".into()
            };
        } else if (policy.min_top_probability > 0.0 && top_prob < effective_min_top_prob)
            || (policy.min_top_probability > 0.0 && candidates.len() > 2 && margin < 0.10 && top_prob < effective_min_top_prob + 0.10)
        {
            status = "uncertain".into();
        }
    }

    let valid_cands: Vec<(&Candidate, f64)> = candidates
        .iter()
        .zip(probs.iter())
        .filter(|(c, _)| !unavailable_ids.contains(&c.id.as_str()))
        .map(|(c, &p)| (c, p))
        .collect();

    let cond_probs: Option<Vec<f64>> = if available_prob > 0.0 {
        Some(valid_cands.iter().map(|(_, p)| p / available_prob).collect())
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
        confidence: concentration,
        probabilities: prob_map,
        logits: logit_map,
        uncertainty: UncertaintyMetrics {
            top_probability: top_prob,
            entropy_nats,
            concentration,
            unavailable_probability: unavailable_prob,
        },
        statistics: None,
        expected_value: None,
        temperature,
    };

    if status == "ok" {
        match question {
            Question::Boolean(_) => {
                answer.decision = Some(serde_json::Value::Bool(winner.id == "true"));
            }
            Question::Choice(_) => {
                answer.decision = Some(serde_json::Value::String(winner.id.clone()));
            }
            Question::Score(_) | Question::Numeric(_) => {
                let values: Vec<f64> = valid_cands.iter().filter_map(|(c, _)| c.value).collect();
                if let Some(ref cp) = cond_probs {
                    if !values.is_empty() {
                        let stats = summarize_moments(&values, cp);
                        answer.expected_value = Some(stats.mean);
                        answer.decision = Some(serde_json::json!(stats.mean));
                        answer.statistics = Some(stats);
                    }
                }
            }
        }
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
                OptionDef { id: "a".into(), description: "A".into() },
                OptionDef { id: "b".into(), description: "B".into() },
            ],
            policy: Default::default(),
        });
        let cands = generate_candidates(&q);
        assert!(decode_decision(&q, &cands, &[1.0], 1.0).is_err());
    }

    #[test]
    fn test_decoding_large_option_buffer() {
        let options: Vec<OptionDef> = (0..130)
            .map(|i| OptionDef { id: format!("opt_{i}"), description: format!("option {i}") })
            .collect();
        let q = Question::Choice(ChoiceQuestion {
            instructions: "test".into(),
            options,
            policy: Policy { allow_abstain: false, ..Default::default() },
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
                OptionDef { id: "a".into(), description: "Option A".into() },
                OptionDef { id: "b".into(), description: "Option B".into() },
                OptionDef { id: "c".into(), description: "Option C".into() },
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
}

