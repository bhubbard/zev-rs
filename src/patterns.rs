//! Composable System One decision patterns: confidence gating, direct routing, and composite scoring.
//!
//! Synthesized from Von (`von-rs`) and Rizzo Flow (`rizzo-flow-rs`).

use crate::error::{Result, ZevError};
use crate::types::ZevAnswer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Answers whose calibrated confidence meets or exceeds the threshold.
    pub automatic: BTreeMap<String, ZevAnswer>,
    /// Answers requiring human escalation or secondary verification.
    pub escalate: BTreeMap<String, ZevAnswer>,
}

/// Selective automation pattern: splits decision answers into automatic execution vs. human escalation
/// based on a calibrated confidence threshold in [0.0, 1.0].
pub fn confidence_gate(
    answers: &BTreeMap<String, ZevAnswer>,
    threshold: f64,
) -> Result<GateResult> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(ZevError::Internal(format!(
            "threshold must be in [0.0, 1.0], got {threshold}"
        )));
    }

    let mut automatic = BTreeMap::new();
    let mut escalate = BTreeMap::new();

    for (qid, ans) in answers {
        if ans.confidence >= threshold && ans.status == "ok" {
            automatic.insert(qid.clone(), ans.clone());
        } else {
            escalate.insert(qid.clone(), ans.clone());
        }
    }

    Ok(GateResult {
        automatic,
        escalate,
    })
}

/// Executes a choice decision and dispatches directly to the matching route closure.
pub fn route_decision<H, D, R>(
    answer: &ZevAnswer,
    routes: &HashMap<String, H>,
    default: Option<D>,
    min_confidence: f64,
) -> Result<R>
where
    H: Fn(&str) -> R,
    D: Fn(&str) -> R,
{
    let choice_str = match &answer.decision {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Bool(b)) => b.to_string(),
        Some(other) => other.to_string(),
        None => {
            return Err(ZevError::Internal(
                "Cannot route decision: answer has no decision target".into(),
            ))
        }
    };

    if answer.confidence >= min_confidence && answer.status == "ok" {
        if let Some(handler) = routes.get(&choice_str) {
            return Ok(handler(&choice_str));
        }
    }

    if let Some(def_handler) = default {
        return Ok(def_handler(&choice_str));
    }

    Err(ZevError::Internal(format!(
        "No matching route handler for choice '{choice_str}' with confidence {:.3}",
        answer.confidence
    )))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeScoreResult {
    /// Combined normalized score in [0.0, 1.0].
    pub composite_score: f64,
    /// Normalized individual components contributing to the composite.
    pub components: BTreeMap<String, f64>,
}

/// Combines multiple Score and Boolean (noul) questions into a unified normalized risk index in [0, 1].
///
/// Score questions are normalized by their max level scale, while boolean questions use true-probability.
pub fn composite_score(
    answers: &BTreeMap<String, ZevAnswer>,
    weights: Option<&HashMap<String, f64>>,
) -> Result<CompositeScoreResult> {
    let mut components = BTreeMap::new();

    for (qid, ans) in answers {
        match ans.question_type.as_str() {
            "noul" | "boolean" => {
                let prob_true = ans
                    .probabilities
                    .get("true")
                    .copied()
                    .or_else(|| ans.probabilities.get("1").copied())
                    .unwrap_or(ans.confidence);
                components.insert(qid.clone(), prob_true.clamp(0.0, 1.0));
            }
            "score" => {
                if let Some(ev) = ans.expected_value {
                    let max_level = (ans.probabilities.len().saturating_sub(1)).max(1) as f64;
                    components.insert(qid.clone(), (ev / max_level).clamp(0.0, 1.0));
                } else if let Some(serde_json::Value::Number(n)) = &ans.decision {
                    if let Some(v) = n.as_f64() {
                        let max_level = (ans.probabilities.len().saturating_sub(1)).max(1) as f64;
                        components.insert(qid.clone(), (v / max_level).clamp(0.0, 1.0));
                    }
                }
            }
            _ => {} // Choice questions excluded from numeric composite
        }
    }

    if components.is_empty() {
        return Err(ZevError::Internal(
            "No score or boolean questions found for composite score".into(),
        ));
    }

    let default_weights = HashMap::new();
    let w_map = weights.unwrap_or(&default_weights);

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for (qid, &val) in &components {
        let weight = w_map.get(qid).copied().unwrap_or(1.0);
        if weight > 0.0 {
            weighted_sum += val * weight;
            total_weight += weight;
        }
    }

    let composite_score = if total_weight > 0.0 {
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(CompositeScoreResult {
        composite_score,
        components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::UncertaintyMetrics;

    fn mock_answer(qtype: &str, decision: &str, conf: f64) -> ZevAnswer {
        let mut probs = BTreeMap::new();
        probs.insert(decision.to_string(), conf);
        ZevAnswer {
            question_type: qtype.into(),
            status: "ok".into(),
            decision: Some(serde_json::json!(decision)),
            confidence: conf,
            probabilities: probs,
            logits: BTreeMap::new(),
            uncertainty: UncertaintyMetrics {
                top_probability: conf,
                entropy_nats: 0.1,
                concentration: 0.9,
                unavailable_probability: 0.0,
                margin: Some(0.8),
                quantile_spread: None,
            },
            statistics: None,
            expected_value: None,
            temperature: 1.0,
            source: Some("mock".into()),
        }
    }

    #[test]
    fn test_confidence_gate() {
        let mut answers = BTreeMap::new();
        answers.insert("high".into(), mock_answer("choice", "approve", 0.95));
        answers.insert("low".into(), mock_answer("choice", "review", 0.60));

        let gate = confidence_gate(&answers, 0.85).unwrap();
        assert_eq!(gate.automatic.len(), 1);
        assert!(gate.automatic.contains_key("high"));
        assert_eq!(gate.escalate.len(), 1);
        assert!(gate.escalate.contains_key("low"));
    }

    #[test]
    fn test_route_decision() {
        fn handle_billing(_: &str) -> &'static str {
            "handled_billing"
        }
        fn handle_support(_: &str) -> &'static str {
            "handled_support"
        }
        fn handle_default(_: &str) -> &'static str {
            "default_handler"
        }

        let ans = mock_answer("choice", "billing", 0.92);
        let mut routes: HashMap<String, fn(&str) -> &'static str> = HashMap::new();
        routes.insert("billing".to_string(), handle_billing);
        routes.insert("support".to_string(), handle_support);

        let res = route_decision(&ans, &routes, Some(handle_default), 0.80).unwrap();
        assert_eq!(res, "handled_billing");

        // Below threshold triggers default
        let res_low = route_decision(&ans, &routes, Some(handle_default), 0.99).unwrap();
        assert_eq!(res_low, "default_handler");
    }

    #[test]
    fn test_composite_score() {
        let mut answers = BTreeMap::new();
        let mut noul_ans = mock_answer("noul", "true", 0.8);
        noul_ans.probabilities.insert("true".into(), 0.8);
        noul_ans.probabilities.insert("false".into(), 0.2);

        let mut score_ans = mock_answer("score", "3", 0.9);
        score_ans.expected_value = Some(3.0);
        score_ans.probabilities.insert("0".into(), 0.0);
        score_ans.probabilities.insert("1".into(), 0.0);
        score_ans.probabilities.insert("2".into(), 0.1);
        score_ans.probabilities.insert("3".into(), 0.8);
        score_ans.probabilities.insert("4".into(), 0.1); // max level 4 => 3.0 / 4.0 = 0.75

        answers.insert("risk_flag".into(), noul_ans);
        answers.insert("severity".into(), score_ans);

        let comp = composite_score(&answers, None).unwrap();
        // (0.8 + 0.75) / 2 = 0.775
        assert!((comp.composite_score - 0.775).abs() < 1e-4);
    }
}
