use crate::engine::ZevEngine;
use crate::error::Result;
use crate::types::{ChoiceQuestion, OptionDef, Policy, Question, ZevRequest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// A row in a tabular dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TabularRow {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl TabularRow {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A batch of tabular rows ready for zero-token AI-SQL operator evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TabularBatch {
    pub rows: Vec<TabularRow>,
}

impl TabularBatch {
    pub fn new(rows: Vec<TabularRow>) -> Self {
        Self { rows }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn push(&mut self, row: TabularRow) {
        self.rows.push(row);
    }
}

/// A filter predicate for tabular rows (similar to Quail's AI_FILTER / BigQuery AI.IF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabularFilterPredicate {
    pub instructions: String,
    pub positive_criterion: String,
    pub negative_criterion: String,
    pub threshold: f64,
}

impl TabularFilterPredicate {
    pub fn new(
        instructions: impl Into<String>,
        positive_criterion: impl Into<String>,
        negative_criterion: impl Into<String>,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            positive_criterion: positive_criterion.into(),
            negative_criterion: negative_criterion.into(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

/// Execution report detailing throughput, wall time, and evaluated items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExecutionReport {
    pub input_rows: usize,
    pub output_rows: usize,
    pub elapsed_microseconds: u128,
    pub rows_per_second: f64,
    pub evaluated_pairs: usize,
}

/// High-performance tabular execution engine for batch AI operations.
pub struct TabularEngine {
    zev: ZevEngine,
}

impl Default for TabularEngine {
    fn default() -> Self {
        Self::new(ZevEngine::default())
    }
}

impl TabularEngine {
    pub fn new(zev: ZevEngine) -> Self {
        Self { zev }
    }

    pub fn engine(&self) -> &ZevEngine {
        &self.zev
    }

    /// Evaluates AI_FILTER over a tabular batch, returning surviving rows.
    pub fn filter_batch(
        &self,
        batch: &TabularBatch,
        predicate: &TabularFilterPredicate,
    ) -> Result<(TabularBatch, BatchExecutionReport)> {
        let t0 = Instant::now();
        let mut surviving = Vec::new();

        for row in &batch.rows {
            let mut questions = BTreeMap::new();
            questions.insert(
                "filter".to_string(),
                Question::Choice(ChoiceQuestion {
                    instructions: predicate.instructions.clone(),
                    options: vec![
                        OptionDef {
                            id: "match".to_string(),
                            description: predicate.positive_criterion.clone(),
                        },
                        OptionDef {
                            id: "reject".to_string(),
                            description: predicate.negative_criterion.clone(),
                        },
                    ],
                    policy: Policy {
                        allow_abstain: false,
                        ..Default::default()
                    },
                }),
            );

            let req = ZevRequest {
                state: serde_json::Value::String(row.text.clone()),
                questions,
                model: None,
                temperature: None,
                enable_temporal_facts: false,
            };

            let resp = self.zev.evaluate(&req)?;
            if let Some(ans) = resp.answers.get("filter") {
                let p_match = ans.probabilities.get("match").copied().unwrap_or(0.0);
                if p_match >= predicate.threshold {
                    surviving.push(row.clone());
                }
            }
        }

        let elapsed = t0.elapsed().as_micros();
        let rows_per_sec = if elapsed > 0 {
            (batch.len() as f64) / (elapsed as f64 / 1_000_000.0)
        } else {
            0.0
        };

        let report = BatchExecutionReport {
            input_rows: batch.len(),
            output_rows: surviving.len(),
            elapsed_microseconds: elapsed,
            rows_per_second: rows_per_sec,
            evaluated_pairs: batch.len(),
        };

        Ok((TabularBatch::new(surviving), report))
    }

    /// Evaluates AI.SCORE (continuous 0.0-1.0 probability) across tabular rows.
    pub fn score_batch(
        &self,
        batch: &TabularBatch,
        instructions: &str,
        positive_criterion: &str,
        negative_criterion: &str,
    ) -> Result<(Vec<(String, f64)>, BatchExecutionReport)> {
        let t0 = Instant::now();
        let mut scores = Vec::with_capacity(batch.len());

        for row in &batch.rows {
            let mut questions = BTreeMap::new();
            questions.insert(
                "score".to_string(),
                Question::Choice(ChoiceQuestion {
                    instructions: instructions.to_string(),
                    options: vec![
                        OptionDef {
                            id: "positive".to_string(),
                            description: positive_criterion.to_string(),
                        },
                        OptionDef {
                            id: "negative".to_string(),
                            description: negative_criterion.to_string(),
                        },
                    ],
                    policy: Policy {
                        allow_abstain: false,
                        ..Default::default()
                    },
                }),
            );

            let req = ZevRequest {
                state: serde_json::Value::String(row.text.clone()),
                questions,
                model: None,
                temperature: None,
                enable_temporal_facts: false,
            };

            let resp = self.zev.evaluate(&req)?;
            let prob = resp
                .answers
                .get("score")
                .and_then(|ans| ans.probabilities.get("positive").copied())
                .unwrap_or(0.0);

            scores.push((row.id.clone(), prob));
        }

        let elapsed = t0.elapsed().as_micros();
        let rows_per_sec = if elapsed > 0 {
            (batch.len() as f64) / (elapsed as f64 / 1_000_000.0)
        } else {
            0.0
        };

        let report = BatchExecutionReport {
            input_rows: batch.len(),
            output_rows: scores.len(),
            elapsed_microseconds: elapsed,
            rows_per_second: rows_per_sec,
            evaluated_pairs: batch.len(),
        };

        Ok((scores, report))
    }

    /// Evaluates AI.ROUTE across a tabular batch, assigning destinations with confidence.
    pub fn route_batch(
        &self,
        batch: &TabularBatch,
        routes: &BTreeMap<String, String>,
    ) -> Result<(Vec<(String, String, f64)>, BatchExecutionReport)> {
        let t0 = Instant::now();
        let mut routed = Vec::with_capacity(batch.len());

        for row in &batch.rows {
            let (dest, conf) = self.zev.route(&row.text, routes.clone())?;
            routed.push((row.id.clone(), dest, conf));
        }

        let elapsed = t0.elapsed().as_micros();
        let rows_per_sec = if elapsed > 0 {
            (batch.len() as f64) / (elapsed as f64 / 1_000_000.0)
        } else {
            0.0
        };

        let report = BatchExecutionReport {
            input_rows: batch.len(),
            output_rows: routed.len(),
            elapsed_microseconds: elapsed,
            rows_per_second: rows_per_sec,
            evaluated_pairs: batch.len() * routes.len(),
        };

        Ok((routed, report))
    }

    /// Evaluates an AI_JOIN (asymmetric cross product) between Anchor rows and Partner criteria.
    ///
    /// For each (anchor, partner) pair, evaluates match condition. Returns matching pairs.
    pub fn join_batch(
        &self,
        anchors: &TabularBatch,
        partners: &TabularBatch,
        instructions: &str,
        threshold: f64,
    ) -> Result<(Vec<(String, String, f64)>, BatchExecutionReport)> {
        let t0 = Instant::now();
        let mut matches = Vec::new();
        let total_pairs = anchors.len() * partners.len();

        for anchor in &anchors.rows {
            for partner in &partners.rows {
                let combined_state = format!("State: {}\nCandidate: {}", anchor.text, partner.text);
                let mut questions = BTreeMap::new();
                questions.insert(
                    "match".to_string(),
                    Question::Choice(ChoiceQuestion {
                        instructions: instructions.to_string(),
                        options: vec![
                            OptionDef {
                                id: "match".to_string(),
                                description: "The candidate criterion applies to the state".to_string(),
                            },
                            OptionDef {
                                id: "mismatch".to_string(),
                                description: "The candidate criterion does not apply".to_string(),
                            },
                        ],
                        policy: Policy {
                            allow_abstain: false,
                            ..Default::default()
                        },
                    }),
                );

                let req = ZevRequest {
                    state: serde_json::Value::String(combined_state),
                    questions,
                    model: None,
                    temperature: None,
                    enable_temporal_facts: true,
                };

                let resp = self.zev.evaluate(&req)?;
                if let Some(ans) = resp.answers.get("match") {
                    let prob = ans.probabilities.get("match").copied().unwrap_or(0.0);
                    if prob >= threshold {
                        matches.push((anchor.id.clone(), partner.id.clone(), prob));
                    }
                }
            }
        }

        let elapsed = t0.elapsed().as_micros();
        let rows_per_sec = if elapsed > 0 {
            (total_pairs as f64) / (elapsed as f64 / 1_000_000.0)
        } else {
            0.0
        };

        let report = BatchExecutionReport {
            input_rows: anchors.len(),
            output_rows: matches.len(),
            elapsed_microseconds: elapsed,
            rows_per_second: rows_per_sec,
            evaluated_pairs: total_pairs,
        };

        Ok((matches, report))
    }
}
