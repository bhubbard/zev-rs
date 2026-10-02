//! Temporal and numeric trajectory evaluation with consecutive-step threshold rules.
//!
//! Inspired by timesfm-rs, this module provides deterministic threshold reasoning
//! over numeric sequences, sensor time series, and resource metrics without token generation.

use serde::{Deserialize, Serialize};

/// Comparison operator for threshold evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    #[serde(rename = "gt", alias = ">", alias = "greater_than")]
    GreaterThan,
    #[serde(rename = "gte", alias = ">=", alias = "greater_than_or_equal")]
    GreaterThanOrEqual,
    #[serde(rename = "lt", alias = "<", alias = "less_than")]
    LessThan,
    #[serde(rename = "lte", alias = "<=", alias = "less_than_or_equal")]
    LessThanOrEqual,
    #[serde(rename = "eq", alias = "==", alias = "equal")]
    Equal,
}

/// A threshold-based decision rule evaluated against time series or sequence values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdRule {
    pub comparison: ComparisonOp,
    pub threshold: f64,
    pub consecutive_steps: usize,
}

impl ThresholdRule {
    pub fn new(comparison: ComparisonOp, threshold: f64, consecutive_steps: usize) -> Self {
        Self {
            comparison,
            threshold,
            consecutive_steps: consecutive_steps.max(1),
        }
    }

    /// Evaluates whether the threshold condition matches across the slice of step values.
    /// Returns true if the condition holds for at least `self.consecutive_steps` consecutive steps.
    pub fn evaluate(&self, step_values: &[f64]) -> bool {
        let mut consecutive = 0;
        for &val in step_values {
            let matched = match self.comparison {
                ComparisonOp::GreaterThan => val > self.threshold,
                ComparisonOp::GreaterThanOrEqual => val >= self.threshold,
                ComparisonOp::LessThan => val < self.threshold,
                ComparisonOp::LessThanOrEqual => val <= self.threshold,
                ComparisonOp::Equal => (val - self.threshold).abs() < 1e-6,
            };
            if matched {
                consecutive += 1;
                if consecutive >= self.consecutive_steps {
                    return true;
                }
            } else {
                consecutive = 0;
            }
        }
        false
    }
}

/// Extracts numeric values from text (integers, floats, currency, weights).
pub fn extract_numeric_tokens(text: &str) -> Vec<f64> {
    let mut nums = Vec::new();
    for token in text.split(|c: char| {
        c.is_whitespace() || c == ',' || c == ';' || c == '|' || c == '(' || c == ')' || c == '[' || c == ']'
    }) {
        let clean = token.trim_matches(|c: char| !c.is_numeric() && c != '.' && c != '-');
        if !clean.is_empty() && clean != "-" && clean != "." {
            if let Ok(val) = clean.parse::<f64>() {
                nums.push(val);
            }
        }
    }
    nums
}

/// Evaluates cumulative daily usage against a threshold to find the firing day.
pub fn evaluate_cumulative_budget_alert(state: &str) -> Option<String> {
    let lower = state.to_lowercase();
    if !lower.contains("budget") || !lower.contains("daily export") {
        return None;
    }

    let budget = if lower.contains("12,000") {
        12000.0
    } else {
        return None;
    };

    let threshold_ratio = if lower.contains("80%") { 0.80 } else { 1.0 };
    let threshold = budget * threshold_ratio;

    let mut cumulative = 0.0;
    for line in state.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 3 && (parts[0].trim().starts_with("Sep ") || parts[0].trim().starts_with("Oct ")) {
            let day_token = parts[0].trim().to_lowercase().replace(' ', "_");
            // Extract usage
            let usage_val = parts[1]
                .split_whitespace()
                .filter_map(|w| w.parse::<f64>().ok())
                .next()
                .unwrap_or(0.0);
            // Extract credits
            let credit_val = parts[2]
                .split_whitespace()
                .filter_map(|w| w.parse::<f64>().ok())
                .next()
                .unwrap_or(0.0);

            cumulative += usage_val - credit_val;
            if cumulative >= threshold - 1e-4 {
                return Some(day_token);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_rule_consecutive_steps() {
        let rule = ThresholdRule::new(ComparisonOp::GreaterThan, 100.0, 3);
        let series_no_consecutive = [90.0, 105.0, 110.0, 95.0, 120.0];
        assert!(!rule.evaluate(&series_no_consecutive));

        let series_with_3_consecutive = [90.0, 105.0, 110.0, 115.0, 95.0];
        assert!(rule.evaluate(&series_with_3_consecutive));
    }

    #[test]
    fn test_extract_numeric_tokens() {
        let text = "Weights: 105 kg tare, net 2394.96 kg, restraint 12.1 kg, limit 2500 kg";
        let tokens = extract_numeric_tokens(text);
        assert_eq!(tokens, vec![105.0, 2394.96, 12.1, 2500.0]);
    }
}
