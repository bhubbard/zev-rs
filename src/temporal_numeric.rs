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

fn clean_num_token(s: &str) -> &str {
    s.trim_matches(|c: char| !c.is_ascii_digit() && c != '-')
        .trim_end_matches('.')
}

/// Extracts numeric values from text (integers, floats, currency, weights, thousand-separated amounts).
pub fn extract_numeric_tokens(text: &str) -> Vec<f64> {
    let mut nums = Vec::new();
    for token in text.split(|c: char| {
        c.is_whitespace() || c == ';' || c == '|' || c == '(' || c == ')' || c == '[' || c == ']'
    }) {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check if token contains thousands separators like "1,250.00" or "$12,000"
        if trimmed.contains(',') {
            let without_commas: String = trimmed.chars().filter(|&c| c != ',').collect();
            let clean = clean_num_token(&without_commas);
            if !clean.is_empty() && clean != "-" && clean != "." {
                if let Ok(val) = clean.parse::<f64>() {
                    let parts: Vec<&str> = trimmed.split(',').collect();
                    let is_thousand_fmt = parts.len() > 1
                        && parts.iter().enumerate().all(|(idx, p)| {
                            let p_clean = clean_num_token(p);
                            if idx == 0 {
                                !p_clean.is_empty() && p_clean.len() <= 3
                            } else if idx == parts.len() - 1 && p_clean.contains('.') {
                                p_clean
                                    .split('.')
                                    .next()
                                    .map(|s| s.len() == 3)
                                    .unwrap_or(false)
                            } else {
                                p_clean.len() == 3
                            }
                        });

                    if is_thousand_fmt {
                        nums.push(val);
                        continue;
                    }
                }
            }

            // Fallback: treat comma as delimiter for comma-separated items
            for sub in trimmed.split(',') {
                let clean = clean_num_token(sub);
                if !clean.is_empty() && clean != "-" && clean != "." {
                    if let Ok(val) = clean.parse::<f64>() {
                        nums.push(val);
                    }
                }
            }
        } else {
            let clean = clean_num_token(trimmed);
            if !clean.is_empty() && clean != "-" && clean != "." {
                if let Ok(val) = clean.parse::<f64>() {
                    nums.push(val);
                }
            }
        }
    }
    nums
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

        let currency_text = "Price is $1,250.00 with credit limit 12,000 USD and fee of $5.50.";
        let cur_tokens = extract_numeric_tokens(currency_text);
        assert_eq!(cur_tokens, vec![1250.0, 12000.0, 5.5]);

        let list_text = "Values: 10,20,30 and 40, 50";
        let list_tokens = extract_numeric_tokens(list_text);
        assert_eq!(list_tokens, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }
}
