use crate::engine::ZevEngine;
use crate::error::Result;
use crate::types::{ChoiceQuestion, OptionDef, Policy, Question, ZevRequest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// Stage type in an execution cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageKind {
    /// Zero-cost rule/heuristic check (e.g. keyword/precondition)
    RuleCheck,
    /// Fast SIMD probabilistic filter (e.g. toxicity / intent gate)
    SimdFilter,
    /// Detailed multi-class or neural verifier
    DetailedEvaluator,
}

/// A stage in an early-rejection predicate cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeStage {
    pub name: String,
    pub kind: StageKind,
    pub instructions: String,
    pub positive_criterion: String,
    pub negative_criterion: String,
    pub threshold: f64,
    pub estimated_cost_micros: f64,
    pub estimated_selectivity: f64, // Fraction expected to pass (0.0 - 1.0)
    #[serde(default = "default_consecutive_required")]
    pub consecutive_required: usize,
}

fn default_consecutive_required() -> usize {
    1
}

impl CascadeStage {
    pub fn new(
        name: impl Into<String>,
        kind: StageKind,
        instructions: impl Into<String>,
        positive_criterion: impl Into<String>,
        negative_criterion: impl Into<String>,
    ) -> Self {
        let default_cost = match kind {
            StageKind::RuleCheck => 0.5,
            StageKind::SimdFilter => 5.0,
            StageKind::DetailedEvaluator => 50.0,
        };

        Self {
            name: name.into(),
            kind,
            instructions: instructions.into(),
            positive_criterion: positive_criterion.into(),
            negative_criterion: negative_criterion.into(),
            threshold: 0.5,
            estimated_cost_micros: default_cost,
            estimated_selectivity: 0.5,
            consecutive_required: 1,
        }
    }

    pub fn with_cost(mut self, cost_micros: f64) -> Self {
        self.estimated_cost_micros = cost_micros;
        self
    }

    pub fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.estimated_selectivity = selectivity;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_consecutive_required(mut self, consecutive: usize) -> Self {
        self.consecutive_required = consecutive.max(1);
        self
    }
}

/// Summary report of a cascaded evaluation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeReport {
    pub passed: bool,
    pub completed_stages: usize,
    pub total_stages: usize,
    pub short_circuited: bool,
    pub elapsed_micros: u128,
    pub stage_results: BTreeMap<String, f64>,
}

/// A planner and executor that orders predicates to minimize total execution cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateCascade {
    pub stages: Vec<CascadeStage>,
}

impl PredicateCascade {
    pub fn new(stages: Vec<CascadeStage>) -> Self {
        let mut cascade = Self { stages };
        cascade.optimize_order();
        cascade
    }

    /// Optimizes stage order using Quail-style cost / selectivity modeling.
    /// Stages that reject the most rows with the lowest compute cost run first.
    pub fn optimize_order(&mut self) {
        // Sort by cost-benefit ratio: Cost / (1 - Selectivity)
        // Lower is better: cheap stages that reject many candidates run earliest.
        self.stages.sort_by(|a, b| {
            let reject_a = (1.0 - a.estimated_selectivity).max(0.001);
            let reject_b = (1.0 - b.estimated_selectivity).max(0.001);
            let score_a = a.estimated_cost_micros / reject_a;
            let score_b = b.estimated_cost_micros / reject_b;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Evaluates the cascade on a state with early-rejection short-circuiting.
    pub fn evaluate(&self, engine: &ZevEngine, state: &str) -> Result<CascadeReport> {
        let t0 = Instant::now();
        let mut stage_results = BTreeMap::new();
        let mut short_circuited = false;
        let mut passed = true;
        let mut completed_stages = 0;

        for stage in &self.stages {
            completed_stages += 1;
            let mut questions = BTreeMap::new();
            questions.insert(
                stage.name.clone(),
                Question::Choice(ChoiceQuestion {
                    instructions: stage.instructions.clone(),
                    options: vec![
                        OptionDef {
                            id: "match".to_string(),
                            description: stage.positive_criterion.clone(),
                        },
                        OptionDef {
                            id: "reject".to_string(),
                            description: stage.negative_criterion.clone(),
                        },
                    ],
                    policy: Policy {
                        allow_abstain: false,
                        ..Default::default()
                    },
                }),
            );

            let req = ZevRequest {
                state: serde_json::Value::String(state.to_string()),
                questions,
                model: None,
                temperature: None,
                enable_temporal_facts: false,
            };

            let resp = engine.evaluate(&req)?;
            let p_match = resp
                .answers
                .get(&stage.name)
                .and_then(|ans| ans.probabilities.get("match").copied())
                .unwrap_or(0.0);

            stage_results.insert(stage.name.clone(), p_match);

            if p_match < stage.threshold {
                passed = false;
                short_circuited = completed_stages < self.stages.len();
                break;
            }
        }

        Ok(CascadeReport {
            passed,
            completed_stages,
            total_stages: self.stages.len(),
            short_circuited,
            elapsed_micros: t0.elapsed().as_micros(),
            stage_results,
        })
    }

    /// Evaluates the cascade on a stream step, maintaining consecutive match streaks.
    /// A stage with `consecutive_required > 1` requires `consecutive_required` consecutive
    /// positive matches before triggering a stage pass.
    pub fn evaluate_stream_step(
        &self,
        engine: &ZevEngine,
        state: &str,
        streaks: &mut BTreeMap<String, usize>,
    ) -> Result<CascadeReport> {
        let t0 = Instant::now();
        let mut stage_results = BTreeMap::new();
        let mut short_circuited = false;
        let mut passed = true;
        let mut completed_stages = 0;

        for stage in &self.stages {
            completed_stages += 1;
            let mut questions = BTreeMap::new();
            questions.insert(
                stage.name.clone(),
                Question::Choice(ChoiceQuestion {
                    instructions: stage.instructions.clone(),
                    options: vec![
                        OptionDef {
                            id: "match".to_string(),
                            description: stage.positive_criterion.clone(),
                        },
                        OptionDef {
                            id: "reject".to_string(),
                            description: stage.negative_criterion.clone(),
                        },
                    ],
                    policy: Policy {
                        allow_abstain: false,
                        ..Default::default()
                    },
                }),
            );

            let req = ZevRequest {
                state: serde_json::Value::String(state.to_string()),
                questions,
                model: None,
                temperature: None,
                enable_temporal_facts: false,
            };

            let resp = engine.evaluate(&req)?;
            let p_match = resp
                .answers
                .get(&stage.name)
                .and_then(|ans| ans.probabilities.get("match").copied())
                .unwrap_or(0.0);

            stage_results.insert(stage.name.clone(), p_match);

            let is_match = p_match >= stage.threshold;
            let streak = streaks.entry(stage.name.clone()).or_insert(0);
            if is_match {
                *streak += 1;
            } else {
                *streak = 0;
            }

            let effective_pass = *streak >= stage.consecutive_required;

            if !effective_pass {
                passed = false;
                short_circuited = completed_stages < self.stages.len();
                break;
            }
        }

        Ok(CascadeReport {
            passed,
            completed_stages,
            total_stages: self.stages.len(),
            short_circuited,
            elapsed_micros: t0.elapsed().as_micros(),
            stage_results,
        })
    }
}

/// Stateful runner that manages consecutive match streaks across sequential inputs.
/// Ported from TimesFM-rs consecutive_steps threshold policy gating.
#[derive(Debug, Clone)]
pub struct SequentialCascadeRunner {
    pub cascade: PredicateCascade,
    pub streaks: BTreeMap<String, usize>,
}

impl SequentialCascadeRunner {
    pub fn new(cascade: PredicateCascade) -> Self {
        Self {
            cascade,
            streaks: BTreeMap::new(),
        }
    }

    pub fn evaluate_step(&mut self, engine: &ZevEngine, state: &str) -> Result<CascadeReport> {
        self.cascade.evaluate_stream_step(engine, state, &mut self.streaks)
    }

    pub fn reset_streaks(&mut self) {
        self.streaks.clear();
    }
}
