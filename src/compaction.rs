//! Agent Tool-Call Compactor (Fast-Jev-Compaction Protocol)
//!
//! Evaluates whether tool executions and their outputs in an AI agent's
//! conversation history should be preserved in full, truncated, or dropped,
//! operating in microseconds with zero token costs and zero model weights.

use crate::engine::DecisionEngine;
use crate::error::Result;
use crate::types::{ChoiceQuestion, OptionDef, Policy, Question, ZevRequest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Record of a tool execution within an agent conversation trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub arguments: String,
    pub result: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub execution_order: usize,
}

/// Recommended compaction action for a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCompactionAction {
    /// Keep the tool call and entire result verbatim.
    KeepFull,
    /// Keep the tool call, but truncate the result (first and last N lines).
    KeepTruncated,
    /// Keep the invocation arguments, but drop the entire result body.
    DropResult,
    /// Drop both the tool call and the result entirely.
    DropAll,
}

/// Outcome of compaction evaluation for a single tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCompactionDecision {
    pub action: ToolCompactionAction,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub reason: String,
}

/// Agent context compaction engine.
#[derive(Default)]
pub struct ToolCompactor {
    engine: DecisionEngine,
}

impl ToolCompactor {
    pub fn new() -> Self {
        Self {
            engine: DecisionEngine::new(),
        }
    }

    /// Evaluates a tool call record against the ongoing conversation goal.
    pub fn evaluate(
        &self,
        conversation_goal: &str,
        tool: &ToolCallRecord,
    ) -> Result<ToolCompactionDecision> {
        let state = format!(
            "Goal: {}\nTool: {}\nInput: {}\nPayload: {}\nExit Status: {}",
            conversation_goal,
            tool.tool_name,
            tool.arguments,
            if tool.result.len() > 1000 {
                &tool.result[..1000]
            } else {
                &tool.result
            },
            if tool.is_error {
                "error failure"
            } else {
                "success"
            }
        );

        let mut questions = BTreeMap::new();
        questions.insert(
            "compaction".to_string(),
            Question::Choice(ChoiceQuestion {
                instructions: "Decide whether to keep or compact this tool execution in the agent context window.".into(),
                options: vec![
                    OptionDef {
                        id: "keep_full".into(),
                        description: "keep full data payload, calculation, or final answer needed for customer invoice or user goal".into(),
                    },
                    OptionDef {
                        id: "keep_truncated".into(),
                        description: "keep truncated summary of verbose build log, compile output, or multiline listing".into(),
                    },
                    OptionDef {
                        id: "drop_result".into(),
                        description: "drop payload of quiet side-effect command like mkdir or rm where execution confirmation suffices".into(),
                    },
                    OptionDef {
                        id: "drop_all".into(),
                        description: "drop entirely superseded attempt, failed command, or transient error".into(),
                    },
                ],
                policy: Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            }),
        );

        let req = ZevRequest {
            state: serde_json::Value::String(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
        };

        let resp = self.engine.eval(&req)?;
        let ans = resp.answers.get("compaction").ok_or_else(|| {
            crate::error::ZevError::DecodingError("Missing compaction answer".into())
        })?;

        let winner_id = match &ans.decision {
            Some(serde_json::Value::String(s)) => s.as_str(),
            _ => ans
                .probabilities
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.as_str())
                .unwrap_or("keep_full"),
        };

        let action = match winner_id {
            "keep_full" => ToolCompactionAction::KeepFull,
            "keep_truncated" => ToolCompactionAction::KeepTruncated,
            "drop_result" => ToolCompactionAction::DropResult,
            "drop_all" => ToolCompactionAction::DropAll,
            _ => ToolCompactionAction::KeepFull,
        };

        let reason = match action {
            ToolCompactionAction::KeepFull => "Tool output is deemed vital to current context.",
            ToolCompactionAction::KeepTruncated => {
                "Tool output is verbose log/listing; head/tail truncation recommended."
            }
            ToolCompactionAction::DropResult => {
                "Execution confirmed; result content is disposable."
            }
            ToolCompactionAction::DropAll => {
                "Intermediate or superseded step; safe to evict completely."
            }
        }
        .to_string();

        Ok(ToolCompactionDecision {
            action,
            confidence: ans.confidence,
            probabilities: ans.probabilities.clone(),
            reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_compaction_keep_final_output() {
        let compactor = ToolCompactor::new();
        let tool = ToolCallRecord {
            tool_name: "calculate_tax".into(),
            arguments: "{\"amount\": 1000, \"state\": \"CA\"}".into(),
            result: "{\"tax\": 72.50, \"total\": 1072.50}".into(),
            is_error: false,
            execution_order: 1,
        };
        let decision = compactor
            .evaluate("Calculate final invoice total for customer", &tool)
            .expect("evaluation failed");
        assert_eq!(decision.action, ToolCompactionAction::KeepFull);
    }

    #[test]
    fn test_tool_compaction_truncate_verbose_log() {
        let compactor = ToolCompactor::new();
        let tool = ToolCallRecord {
            tool_name: "run_build".into(),
            arguments: "cargo build --release".into(),
            result: "Compiling 400 crates...\nwarning: unused variable...\nFinished release profile in 42s".into(),
            is_error: false,
            execution_order: 2,
        };
        let decision = compactor
            .evaluate("Verify compilation and summarize build warnings", &tool)
            .expect("evaluation failed");
        assert!(
            decision.action == ToolCompactionAction::KeepTruncated
                || decision.action == ToolCompactionAction::KeepFull
        );
    }
}
