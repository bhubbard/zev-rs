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

use chrono::NaiveDate;
use regex::Regex;
use std::sync::LazyLock;

static RE_DURATION_POLICY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:within|allow(?:s|ed)?\s+up\s+to|valid\s+for|window\s+of|limit\s+of|policy\s+of|has\s+a)\s*(\d+)\s*(day|week|month|year)s?|(\d+)-(day|week|month|year)\s*(?:return|warranty|grace\s+period|trial|window|policy|limit|deadline)")
        .expect("valid duration policy regex")
});

static RE_ISO_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").expect("valid iso date regex")
});

static RE_TEXT_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b")
        .expect("valid text date regex")
});

/// Evaluates whether elapsed durations between dates mentioned in text exceed or satisfy stated policy windows.
/// Returns descriptive findings for warranty, return, grace period, and SLA questions.
pub fn resolve_temporal_duration_constraints(text: &str) -> Vec<String> {
    let mut policies = Vec::new();

    for cap in RE_DURATION_POLICY.captures_iter(text) {
        let (num_str, unit_str) = if let (Some(n), Some(u)) = (cap.get(1), cap.get(2)) {
            (n.as_str(), u.as_str())
        } else if let (Some(n), Some(u)) = (cap.get(3), cap.get(4)) {
            (n.as_str(), u.as_str())
        } else {
            continue;
        };

        if let Ok(num) = num_str.parse::<i64>() {
            let days = match unit_str.to_lowercase().as_str() {
                "day" => num,
                "week" => num * 7,
                "month" => num * 30,
                "year" => num * 365,
                _ => continue,
            };
            policies.push((days, format!("{} {}", num, unit_str)));
        }
    }

    if policies.is_empty() {
        return Vec::new();
    }

    // Extract all unique dates
    let mut found_dates: Vec<(String, NaiveDate)> = Vec::new();
    for cap in RE_ISO_DATE.captures_iter(text) {
        let raw = cap.get(0).unwrap().as_str();
        if let Ok(d) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
            if !found_dates.iter().any(|(r, _)| *r == raw) {
                found_dates.push((raw.to_string(), d));
            }
        }
    }

    for cap in RE_TEXT_DATE.captures_iter(text) {
        let raw = cap.get(0).unwrap().as_str();
        let month_str = &cap[1];
        let day_str = &cap[2];
        let year_str = &cap[3];
        let date_str = format!("{month_str} {day_str}, {year_str}");
        if let Ok(d) = NaiveDate::parse_from_str(&date_str, "%B %d, %Y") {
            if !found_dates.iter().any(|(r, _)| *r == raw) {
                found_dates.push((raw.to_string(), d));
            }
        }
    }

    if found_dates.len() < 2 {
        return Vec::new();
    }

    found_dates.sort_by_key(|&(_, d)| d);

    let mut findings = Vec::new();
    for i in 0..found_dates.len() {
        for j in (i + 1)..found_dates.len() {
            let (raw_i, date_i) = &found_dates[i];
            let (raw_j, date_j) = &found_dates[j];
            let elapsed = (*date_j - *date_i).num_days();

            if elapsed <= 0 {
                continue;
            }

            for &(policy_days, ref policy_name) in &policies {
                if elapsed > policy_days {
                    findings.push(format!(
                        "[TEMPORAL CONSTRAINT]: Elapsed duration between {} and {} is {} days, which exceeds the stated {} limit (EXCEEDS_POLICY_DURATION: TRUE, WITHIN_WINDOW: FALSE).",
                        raw_i, raw_j, elapsed, policy_name
                    ));
                } else {
                    findings.push(format!(
                        "[TEMPORAL CONSTRAINT]: Elapsed duration between {} and {} is {} days, which is within the stated {} limit (EXCEEDS_POLICY_DURATION: FALSE, WITHIN_WINDOW: TRUE).",
                        raw_i, raw_j, elapsed, policy_name
                    ));
                }
            }
        }
    }

    findings
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

    #[test]
    fn test_resolve_temporal_duration_constraints() {
        let text_expired = "Store has a 30-day return policy. Order delivered on 2023-01-10. Customer requested return on 2023-02-25.";
        let res_expired = resolve_temporal_duration_constraints(text_expired);
        assert!(!res_expired.is_empty());
        assert!(res_expired[0].contains("EXCEEDS_POLICY_DURATION: TRUE"));

        let text_valid = "Covered under a 1-year warranty. Purchased on 2022-03-01. Defect reported on 2022-08-15.";
        let res_valid = resolve_temporal_duration_constraints(text_valid);
        assert!(!res_valid.is_empty());
        assert!(res_valid[0].contains("EXCEEDS_POLICY_DURATION: FALSE"));
    }
}
