use std::collections::BTreeMap;
use crate::calibration::scaled_softmax;
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

    let probs = scaled_softmax(logits, temperature)?;

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

    let top_idx = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
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

    if unavailable_ids.contains(&winner.id.as_str()) || unavailable_prob >= policy.max_unavailable_probability {
        let below_prob = prob_map.get(BELOW).copied().unwrap_or(0.0);
        let above_prob = prob_map.get(ABOVE).copied().unwrap_or(0.0);
        let unknown_prob = prob_map.get(UNKNOWN).copied().unwrap_or(0.0);

        status = if (below_prob + above_prob) > unknown_prob {
            "out_of_range".into()
        } else {
            "insufficient_evidence".into()
        };
    } else if top_prob < policy.min_top_probability {
        status = "uncertain".into();
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
