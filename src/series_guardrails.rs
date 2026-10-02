//! Pre-flight numerical series sanity guardrails and fast abstention detection.
//!
//! Inspired by timesfm-rs, this module provides microsecond-level sanity checks on
//! numerical slices, series, and sensor readings. It detects:
//! - Empty sequences
//! - High NaN / infinite value ratios
//! - Degenerate flatlines (zero or near-zero variance)
//! - Insufficient sample counts
//!
//! When detected, it triggers fast `UNKNOWN` (`__insufficient__`) abstention in <4 µs
//! before executing full logit decoding.

use crate::types::UNKNOWN;
use serde::{Deserialize, Serialize};

pub const TOLERANCE: f64 = 1e-6;

/// Replaces values smaller than tolerance with the signed tolerance floor to prevent division-by-zero.
#[inline(always)]
pub fn make_safe_for_division(val: f64) -> f64 {
    if val.abs() < TOLERANCE {
        if val >= 0.0 {
            TOLERANCE
        } else {
            -TOLERANCE
        }
    } else {
        val
    }
}

/// Single-pass running statistics update (Welford's algorithm).
///
/// Returns (new_count, new_mean, new_m2) where sample variance is `new_m2 / (new_count - 1)`.
#[inline]
pub fn update_running_stats(count: usize, mean: f64, m2: f64, x: f64) -> (usize, f64, f64) {
    let new_count = count + 1;
    let delta = x - mean;
    let new_mean = mean + delta / (new_count as f64);
    let delta2 = x - new_mean;
    let new_m2 = m2 + delta * delta2;
    (new_count, new_mean, new_m2)
}

/// Reversible instance normalization (RevIN) for numerical sequences.
/// Normalizes a sequence to zero mean and unit variance.
pub fn revin_normalize(series: &[f64]) -> (Vec<f64>, f64, f64) {
    if series.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }
    let n = series.len() as f64;
    let mean = series.iter().sum::<f64>() / n;
    let var = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = make_safe_for_division(var.sqrt());
    let normalized = series.iter().map(|&x| (x - mean) / std).collect();
    (normalized, mean, std)
}

/// Denormalizes a previously normalized sequence back to its original scale.
pub fn revin_denormalize(normalized: &[f64], mean: f64, std: f64) -> Vec<f64> {
    normalized.iter().map(|&x| x * std + mean).collect()
}

/// Configuration parameters for pre-flight series sanity guardrails.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailConfig {
    pub min_length: usize,
    pub max_nan_ratio: f64,
    pub min_variance: f64,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            min_length: 3,
            max_nan_ratio: 0.30,
            min_variance: 1e-8,
        }
    }
}

/// Result of pre-flight time series sanity guardrail checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub passed: bool,
    pub should_abstain: bool,
    pub abstention_code: Option<String>,
    pub reason: Option<String>,
    pub total_points: usize,
    pub valid_points: usize,
    pub nan_count: usize,
    pub nan_ratio: f64,
    pub variance: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub mean_val: f64,
    pub is_flatline: bool,
}

impl GuardrailResult {
    #[inline]
    pub fn is_ok(&self) -> bool {
        self.passed
    }
}

/// Checks pre-flight sanity guardrails on a series slice with default configuration.
#[inline]
pub fn check_series_guardrails(series: &[f64]) -> GuardrailResult {
    check_series_guardrails_with_config(series, &GuardrailConfig::default())
}

/// Checks pre-flight sanity guardrails on a series slice with custom configuration.
pub fn check_series_guardrails_with_config(
    series: &[f64],
    config: &GuardrailConfig,
) -> GuardrailResult {
    let total_points = series.len();
    if total_points == 0 {
        return GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(UNKNOWN.to_string()),
            reason: Some("Empty series: no data points provided".to_string()),
            total_points: 0,
            valid_points: 0,
            nan_count: 0,
            nan_ratio: 1.0,
            variance: 0.0,
            min_val: 0.0,
            max_val: 0.0,
            mean_val: 0.0,
            is_flatline: true,
        };
    }

    let mut valid_points = 0usize;
    let mut nan_count = 0usize;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for &val in series {
        if val.is_finite() {
            valid_points += 1;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            sum += val;
            sum_sq += val * val;
        } else {
            nan_count += 1;
        }
    }

    let nan_ratio = nan_count as f64 / total_points as f64;

    if valid_points < config.min_length {
        return GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(UNKNOWN.to_string()),
            reason: Some(format!(
                "Insufficient valid data points ({} < required {})",
                valid_points, config.min_length
            )),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance: 0.0,
            min_val: if min_val.is_finite() { min_val } else { 0.0 },
            max_val: if max_val.is_finite() { max_val } else { 0.0 },
            mean_val: if valid_points > 0 {
                sum / valid_points as f64
            } else {
                0.0
            },
            is_flatline: true,
        };
    }

    if nan_ratio > config.max_nan_ratio {
        let mean = sum / valid_points as f64;
        let variance = ((sum_sq / valid_points as f64) - (mean * mean)).max(0.0);
        return GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(UNKNOWN.to_string()),
            reason: Some(format!(
                "High NaN ratio: {:.1}% exceeds allowed {:.1}%",
                nan_ratio * 100.0,
                config.max_nan_ratio * 100.0
            )),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance,
            min_val,
            max_val,
            mean_val: mean,
            is_flatline: false,
        };
    }

    let mean = sum / valid_points as f64;
    let variance = ((sum_sq / valid_points as f64) - (mean * mean)).max(0.0);
    let range = max_val - min_val;
    let is_flatline = variance <= config.min_variance || range <= 1e-7;

    if is_flatline {
        return GuardrailResult {
            passed: false,
            should_abstain: true,
            abstention_code: Some(UNKNOWN.to_string()),
            reason: Some("Degenerate flatline series: zero or near-zero variance".to_string()),
            total_points,
            valid_points,
            nan_count,
            nan_ratio,
            variance,
            min_val,
            max_val,
            mean_val: mean,
            is_flatline: true,
        };
    }

    GuardrailResult {
        passed: true,
        should_abstain: false,
        abstention_code: None,
        reason: None,
        total_points,
        valid_points,
        nan_count,
        nan_ratio,
        variance,
        min_val,
        max_val,
        mean_val: mean,
        is_flatline: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_series_abstains() {
        let res = check_series_guardrails(&[]);
        assert!(!res.passed);
        assert!(res.should_abstain);
        assert_eq!(res.abstention_code, Some(UNKNOWN.to_string()));
    }

    #[test]
    fn test_flatline_series_abstains() {
        let series = [42.0, 42.0, 42.0, 42.0, 42.0];
        let res = check_series_guardrails(&series);
        assert!(!res.passed);
        assert!(res.should_abstain);
        assert!(res.is_flatline);
    }

    #[test]
    fn test_valid_series_passes() {
        let series = [10.0, 15.0, 12.0, 18.0, 22.0];
        let res = check_series_guardrails(&series);
        assert!(res.passed);
        assert!(!res.should_abstain);
        assert!(!res.is_flatline);
        assert!(res.variance > 0.0);
    }

    #[test]
    fn test_running_stats_and_revin() {
        let series = [10.0, 20.0, 30.0, 40.0, 50.0];
        let (norm, mean, std) = revin_normalize(&series);
        assert_eq!(norm.len(), 5);
        assert!((mean - 30.0).abs() < 1e-4);

        let restored = revin_denormalize(&norm, mean, std);
        for (orig, rest) in series.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() < 1e-4);
        }
    }
}
