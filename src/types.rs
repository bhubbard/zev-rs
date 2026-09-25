use crate::error::{Result, ZevError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_MODEL: &str = "zev-apex-v1";
pub const MODEL_ALIAS: &str = "zev-latest";
pub const MAX_SLOTS: usize = 26;
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_STATE_BYTES: usize = 2 * 1024 * 1024; // 2MB
pub const DEFAULT_CALIBRATED_TEMPERATURE: f64 = 2.179078721266035;

pub const UNKNOWN: &str = "__insufficient__";
pub const BELOW: &str = "__below_range__";
pub const ABOVE: &str = "__above_range__";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    #[serde(default = "default_allow_abstain")]
    pub allow_abstain: bool,
    #[serde(default = "default_max_unavailable_prob")]
    pub max_unavailable_probability: f64,
    #[serde(default = "default_min_top_prob")]
    pub min_top_probability: f64,
    #[serde(default)]
    pub max_slots: Option<usize>,
}

fn default_allow_abstain() -> bool {
    true
}
fn default_max_unavailable_prob() -> f64 {
    0.5
}
fn default_min_top_prob() -> f64 {
    0.0
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            allow_abstain: default_allow_abstain(),
            max_unavailable_probability: default_max_unavailable_prob(),
            min_top_probability: default_min_top_prob(),
            max_slots: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionDef {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub value: f64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanQuestion {
    pub instructions: String,
    #[serde(default = "default_true_desc")]
    pub true_description: String,
    #[serde(default = "default_false_desc")]
    pub false_description: String,
    #[serde(default)]
    pub policy: Policy,
}

fn default_true_desc() -> String {
    "Yes. The context provides strong affirmative evidence.".into()
}

fn default_false_desc() -> String {
    "No. The context contradicts or does not support the premise.".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    pub instructions: String,
    pub options: Vec<OptionDef>,
    #[serde(default)]
    pub policy: Policy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    pub instructions: String,
    pub levels: Vec<String>,
    #[serde(default)]
    pub policy: Policy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericQuestion {
    pub instructions: String,
    pub unit: String,
    pub anchors: Vec<Anchor>,
    #[serde(default)]
    pub policy: Policy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Boolean(BooleanQuestion),
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Numeric(NumericQuestion),
}

impl Question {
    pub fn policy(&self) -> &Policy {
        match self {
            Question::Boolean(q) => &q.policy,
            Question::Choice(q) => &q.policy,
            Question::Score(q) => &q.policy,
            Question::Numeric(q) => &q.policy,
        }
    }

    pub fn instructions(&self) -> &str {
        match self {
            Question::Boolean(q) => &q.instructions,
            Question::Choice(q) => &q.instructions,
            Question::Score(q) => &q.instructions,
            Question::Numeric(q) => &q.instructions,
        }
    }

    pub fn validate(&self, key: &str) -> Result<()> {
        let reserved = if self.policy().allow_abstain { 1 } else { 0 };
        let slot_limit = self.policy().max_slots.unwrap_or(MAX_SLOTS);
        match self {
            Question::Boolean(_) => Ok(()),
            Question::Choice(c) => {
                if c.options.len() < 2 {
                    return Err(ZevError::InvalidRequest(format!(
                        "{key}: choice requires at least 2 options"
                    )));
                }
                if c.options.len() + reserved > slot_limit {
                    return Err(ZevError::SlotLimitExceeded(format!(
                        "{key}: options ({}) + reserved ({reserved}) exceed slot limit ({slot_limit})",
                        c.options.len()
                    )));
                }
                Ok(())
            }
            Question::Score(s) => {
                if s.levels.len() < 2 {
                    return Err(ZevError::InvalidRequest(format!(
                        "{key}: score requires at least 2 levels"
                    )));
                }
                if s.levels.len() + reserved > slot_limit {
                    return Err(ZevError::SlotLimitExceeded(format!(
                        "{key}: levels ({}) + reserved ({reserved}) exceed slot limit ({slot_limit})",
                        s.levels.len()
                    )));
                }
                Ok(())
            }
            Question::Numeric(n) => {
                if n.anchors.len() < 2 {
                    return Err(ZevError::InvalidRequest(format!(
                        "{key}: numeric requires at least 2 anchors"
                    )));
                }
                if n.anchors.len() + reserved + 2 > slot_limit {
                    return Err(ZevError::SlotLimitExceeded(format!(
                        "{key}: anchors exceed slot capacity"
                    )));
                }
                for pair in n.anchors.windows(2) {
                    if pair[1].value <= pair[0].value {
                        return Err(ZevError::InvalidRequest(format!(
                            "{key}: numeric anchors must be strictly increasing"
                        )));
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub description: String,
    pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStatistics {
    pub mean: f64,
    pub stddev: f64,
    pub median: f64,
    pub quantiles: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyMetrics {
    pub top_probability: f64,
    pub entropy_nats: f64,
    pub concentration: f64,
    pub unavailable_probability: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZevAnswer {
    #[serde(rename = "type")]
    pub question_type: String,
    pub status: String,
    pub decision: Option<serde_json::Value>,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub logits: BTreeMap<String, f64>,
    pub uncertainty: UncertaintyMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistics: Option<DecisionStatistics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    pub temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZevRequest {
    pub state: serde_json::Value,
    pub questions: BTreeMap<String, Question>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default = "default_enable_temporal")]
    pub enable_temporal_facts: bool,
}

fn default_enable_temporal() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTiming {
    pub total_micros: f64,
    pub eval_micros: f64,
    pub shared_prefix_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZevResponse {
    pub model: String,
    pub answers: BTreeMap<String, ZevAnswer>,
    pub execution: ExecutionTiming,
}

// -------------------------------------------------------------------------------------------------
// TypeSafe / OpenJev Compatibility Wire Types
// -------------------------------------------------------------------------------------------------

pub use crate::wire::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_instructions_and_defaults() {
        let b: BooleanQuestion = serde_json::from_str(r#"{"instructions": "is active"}"#).unwrap();
        assert_eq!(
            b.true_description,
            "Yes. The context provides strong affirmative evidence."
        );
        assert_eq!(
            b.false_description,
            "No. The context contradicts or does not support the premise."
        );

        let q_bool = Question::Boolean(b);
        assert_eq!(q_bool.instructions(), "is active");

        let q_choice = Question::Choice(ChoiceQuestion {
            instructions: "pick one".into(),
            options: vec![
                OptionDef {
                    id: "a".into(),
                    description: "desc a".into(),
                },
                OptionDef {
                    id: "b".into(),
                    description: "desc b".into(),
                },
            ],
            policy: Default::default(),
        });
        assert_eq!(q_choice.instructions(), "pick one");

        let q_score = Question::Score(ScoreQuestion {
            instructions: "rate quality".into(),
            levels: vec!["bad".into(), "good".into()],
            policy: Default::default(),
        });
        assert_eq!(q_score.instructions(), "rate quality");

        let q_num = Question::Numeric(NumericQuestion {
            instructions: "estimate temp".into(),
            unit: "F".into(),
            anchors: vec![
                Anchor {
                    value: 0.0,
                    description: "freezing".into(),
                },
                Anchor {
                    value: 100.0,
                    description: "boiling".into(),
                },
            ],
            policy: Default::default(),
        });
        assert_eq!(q_num.instructions(), "estimate temp");
    }

    #[test]
    fn test_score_validation_edge_cases() {
        let q_few = Question::Score(ScoreQuestion {
            instructions: "rate".into(),
            levels: vec!["only_one".into()],
            policy: Default::default(),
        });
        assert!(q_few.validate("test").is_err());

        let q_many = Question::Score(ScoreQuestion {
            instructions: "rate".into(),
            levels: (0..30).map(|i| format!("lvl_{i}")).collect(),
            policy: Policy {
                allow_abstain: true,
                ..Default::default()
            },
        });
        assert!(q_many.validate("test").is_err());
    }

    #[test]
    fn test_numeric_validation_edge_cases() {
        let q_few = Question::Numeric(NumericQuestion {
            instructions: "num".into(),
            unit: "x".into(),
            anchors: vec![Anchor {
                value: 1.0,
                description: "one".into(),
            }],
            policy: Default::default(),
        });
        assert!(q_few.validate("test").is_err());

        let q_unordered = Question::Numeric(NumericQuestion {
            instructions: "num".into(),
            unit: "x".into(),
            anchors: vec![
                Anchor {
                    value: 10.0,
                    description: "ten".into(),
                },
                Anchor {
                    value: 5.0,
                    description: "five".into(),
                },
            ],
            policy: Default::default(),
        });
        assert!(q_unordered.validate("test").is_err());

        let q_many = Question::Numeric(NumericQuestion {
            instructions: "num".into(),
            unit: "x".into(),
            anchors: (0..30)
                .map(|i| Anchor {
                    value: i as f64,
                    description: format!("a{i}"),
                })
                .collect(),
            policy: Policy {
                allow_abstain: true,
                ..Default::default()
            },
        });
        assert!(q_many.validate("test").is_err());
    }
}
