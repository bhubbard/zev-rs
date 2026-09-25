use crate::error::{Result, ZevError};
use crate::types::{
    BooleanQuestion, ChoiceQuestion, OptionDef, Policy, Question, ScoreQuestion, ZevAnswer,
    MODEL_ALIAS,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// -------------------------------------------------------------------------------------------------
// TypeSafe / OpenJev Compatibility Wire Types
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireNoulCriteria {
    #[serde(rename = "true", default)]
    pub true_criterion: Option<serde_json::Value>,
    #[serde(rename = "false", default)]
    pub false_criterion: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireNoulQuestion {
    pub instructions: serde_json::Value,
    #[serde(default)]
    pub criteria: Option<WireNoulCriteria>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireChoiceQuestion {
    pub instructions: serde_json::Value,
    pub criteria: BTreeMap<String, Option<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireScoreQuestion {
    pub instructions: serde_json::Value,
    pub criteria: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WireQuestion {
    Noul(WireNoulQuestion),
    Choice(WireChoiceQuestion),
    Score(WireScoreQuestion),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: serde_json::Value,
    #[serde(default = "default_wire_model")]
    pub model: String,
    pub questions: BTreeMap<String, WireQuestion>,
}

fn default_wire_model() -> String {
    MODEL_ALIAS.into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, serde_json::Value>,
    pub usage: WireUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Strongly-typed Wire Answer representations for SystemOne JSON compatibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum WireAnswer {
    Noul {
        noul: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    Score {
        score: f64,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
}

impl TryFrom<&WireQuestion> for Question {
    type Error = ZevError;

    fn try_from(wire: &WireQuestion) -> Result<Self> {
        wire_to_question(wire)
    }
}

impl TryFrom<WireQuestion> for Question {
    type Error = ZevError;

    fn try_from(wire: WireQuestion) -> Result<Self> {
        wire_to_question(&wire)
    }
}

/// Converts a wire question specification into the unified Question domain
pub fn wire_to_question(wire: &WireQuestion) -> Result<Question> {
    match wire {
        WireQuestion::Noul(n) => {
            let instr = n.instructions.as_str().unwrap_or("").to_string();
            let true_desc = n
                .criteria
                .as_ref()
                .and_then(|c| c.true_criterion.as_ref())
                .and_then(|t| t.as_str())
                .unwrap_or("Yes. Supports affirmative answer.");
            let false_desc = n
                .criteria
                .as_ref()
                .and_then(|c| c.false_criterion.as_ref())
                .and_then(|f| f.as_str())
                .unwrap_or("No. Supports negative answer.");
            Ok(Question::Boolean(BooleanQuestion {
                instructions: instr,
                true_description: true_desc.to_string(),
                false_description: false_desc.to_string(),
                policy: Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            }))
        }
        WireQuestion::Choice(c) => {
            let instr = c.instructions.as_str().unwrap_or("").to_string();
            let mut options = Vec::with_capacity(c.criteria.len());
            for (name, detail) in &c.criteria {
                let desc = detail
                    .as_ref()
                    .and_then(|d| d.as_str())
                    .unwrap_or(name.as_str())
                    .to_string();
                options.push(OptionDef {
                    id: name.clone(),
                    description: desc,
                });
            }
            Ok(Question::Choice(ChoiceQuestion {
                instructions: instr,
                options,
                policy: Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            }))
        }
        WireQuestion::Score(s) => {
            let instr = s.instructions.as_str().unwrap_or("").to_string();
            let levels: Vec<String> = s
                .criteria
                .iter()
                .map(|lvl| {
                    if lvl.is_null() {
                        String::new()
                    } else if let Some(s) = lvl.as_str() {
                        s.to_string()
                    } else {
                        lvl.to_string()
                    }
                })
                .collect();
            Ok(Question::Score(ScoreQuestion {
                instructions: instr,
                levels,
                policy: Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            }))
        }
    }
}

/// Converts a evaluated core ZevAnswer back into the wire WireAnswer
pub fn wire_answer_from_zev_answer(wire_q: &WireQuestion, ans: &ZevAnswer) -> WireAnswer {
    match wire_q {
        WireQuestion::Noul(_) => {
            let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);
            WireAnswer::Noul {
                noul: p_true,
                confidence: None,
                source: ans.source.clone(),
            }
        }
        WireQuestion::Choice(_) => {
            let choice = match &ans.decision {
                Some(serde_json::Value::String(s)) => s.clone(),
                _ => ans
                    .probabilities
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k.clone())
                    .unwrap_or_default(),
            };
            WireAnswer::Choice {
                choice,
                probabilities: ans.probabilities.clone(),
                confidence: ans.confidence,
                source: ans.source.clone(),
            }
        }
        WireQuestion::Score(s) => {
            let mut legend = BTreeMap::new();
            for (idx, lvl) in s.criteria.iter().enumerate() {
                let id_str = idx.to_string();
                let desc = if lvl.is_null() {
                    String::new()
                } else if let Some(s) = lvl.as_str() {
                    s.to_string()
                } else {
                    lvl.to_string()
                };
                legend.insert(id_str, desc);
            }
            let score = ans.expected_value.unwrap_or(0.0);
            WireAnswer::Score {
                score,
                legend,
                probabilities: ans.probabilities.clone(),
                confidence: ans.confidence,
                source: ans.source.clone(),
            }
        }
    }
}

/// Converts a core ZevAnswer into a wire-compatible serde_json::Value
pub fn wire_value_from_zev_answer(
    wire_q: &WireQuestion,
    ans: &ZevAnswer,
) -> Result<serde_json::Value> {
    let wire_ans = wire_answer_from_zev_answer(wire_q, ans);
    Ok(serde_json::to_value(wire_ans)?)
}
