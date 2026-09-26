use crate::error::{Result, ZevError};
use crate::types::DEFAULT_CALIBRATED_TEMPERATURE;
use serde::{Deserialize, Serialize};

/// Question-type specific calibrated temperatures.
/// Based on empirical calibration findings from Decider (Mapika) and Jev:
/// - choice: 1.48 (multi-class categorical routing)
/// - boolean: 2.22 (binary noul)
/// - score: 1.38 (ordinal ratings / Likert scale)
/// - numeric: 1.25 (anchor regression)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TypeTemperatureConfig {
    pub choice: f64,
    pub boolean: f64,
    pub score: f64,
    pub numeric: f64,
}

impl Default for TypeTemperatureConfig {
    fn default() -> Self {
        Self {
            choice: 1.48,
            boolean: 2.22,
            score: 1.38,
            numeric: 1.25,
        }
    }
}

impl TypeTemperatureConfig {
    pub fn new(choice: f64, boolean: f64, score: f64, numeric: f64) -> Self {
        Self {
            choice,
            boolean,
            score,
            numeric,
        }
    }

    pub fn get_temperature(&self, q_type: &str) -> f64 {
        match q_type.to_lowercase().as_str() {
            "choice" | "routing" => self.choice,
            "boolean" | "noul" => self.boolean,
            "score" | "ordinal" => self.score,
            "numeric" => self.numeric,
            _ => DEFAULT_CALIBRATED_TEMPERATURE,
        }
    }
}

pub fn resolve_temperature(user_temp: Option<f64>) -> Result<f64> {
    let t = user_temp.unwrap_or(DEFAULT_CALIBRATED_TEMPERATURE);
    if !t.is_finite() || t <= 0.0 {
        return Err(ZevError::CalibrationError(
            "Temperature must be a positive finite float".into(),
        ));
    }
    Ok(t)
}

pub fn scaled_softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    if logits.is_empty() {
        return Err(ZevError::DecodingError(
            "Logits array cannot be empty".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ZevError::CalibrationError(
            "Temperature must be positive and finite".into(),
        ));
    }

    let mut out = vec![0.0; logits.len()];
    scaled_softmax_slice(logits, temperature, &mut out)?;
    Ok(out)
}

#[inline(always)]
pub fn scaled_softmax_slice(logits: &[f64], temperature: f64, out: &mut [f64]) -> Result<()> {
    if logits.is_empty() || logits.len() != out.len() {
        return Err(ZevError::DecodingError(
            "Logits array cannot be empty".into(),
        ));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ZevError::CalibrationError(
            "Temperature must be positive and finite".into(),
        ));
    }

    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let inv_temp = 1.0 / temperature;
    let mut sum = 0.0;
    for (i, &x) in logits.iter().enumerate() {
        let w = ((x - max_logit) * inv_temp).exp();
        out[i] = w;
        sum += w;
    }

    if sum <= 0.0 || !sum.is_finite() {
        return Err(ZevError::DecodingError(
            "Softmax normalization encountered non-finite sum".into(),
        ));
    }

    let inv_sum = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
    Ok(())
}

/// Returns the family-adapted calibrated temperature (Hopper protocol).
/// Applies specialized temperature multipliers by question family.
pub fn family_calibrated_temperature(family: &str, base_temp: f64) -> f64 {
    match family {
        "intent" | "routing" => (base_temp * 0.90).max(1.0),
        "policy" | "long_policy" => base_temp * 1.20,
        "noul" | "boolean" => base_temp * 0.95,
        "trap" | "adversarial" => base_temp * 1.15,
        "score" | "ordinal" => base_temp * 1.05,
        _ => base_temp,
    }
}

/// Multi-candidate margin temperature dampening (Maisa djev protocol).
/// If the margin between the top two logits is below `margin_threshold`,
/// smoothly softens the temperature to prevent overconfidence on knife-edge ties.
#[inline]
pub fn dampen_temperature_by_margin(logits: &[f64], base_temp: f64, margin_threshold: f64) -> f64 {
    if logits.len() < 2 || margin_threshold <= 0.0 {
        return base_temp;
    }
    let mut top1 = f64::NEG_INFINITY;
    let mut top2 = f64::NEG_INFINITY;
    for &x in logits {
        if x > top1 {
            top2 = top1;
            top1 = x;
        } else if x > top2 {
            top2 = x;
        }
    }
    if top2.is_finite() {
        let margin = (top1 - top2).max(0.0);
        if margin < margin_threshold {
            let factor = 1.0 + 0.35 * (1.0 - margin / margin_threshold);
            return base_temp * factor;
        }
    }
    base_temp
}

/// Computes Expected Calibration Error (ECE) across M equal-width bins
pub fn compute_ece(confidences: &[f64], accuracies: &[bool], num_bins: usize) -> f64 {
    if confidences.is_empty() || confidences.len() != accuracies.len() || num_bins == 0 {
        return 0.0;
    }

    let bin_size = 1.0 / num_bins as f64;
    let mut ece = 0.0;
    let total_n = confidences.len() as f64;

    for i in 0..num_bins {
        let bin_lower = i as f64 * bin_size;
        let bin_upper = (i + 1) as f64 * bin_size;

        let mut in_bin_count = 0;
        let mut correct_count = 0;
        let mut sum_conf = 0.0;

        for (&conf, &acc) in confidences.iter().zip(accuracies.iter()) {
            if (conf >= bin_lower && conf < bin_upper)
                || (i == num_bins - 1 && conf >= bin_lower && conf <= 1.0)
            {
                in_bin_count += 1;
                sum_conf += conf;
                if acc {
                    correct_count += 1;
                }
            }
        }

        if in_bin_count > 0 {
            let bin_acc = correct_count as f64 / in_bin_count as f64;
            let bin_avg_conf = sum_conf / in_bin_count as f64;
            let weight = in_bin_count as f64 / total_n;
            ece += weight * (bin_acc - bin_avg_conf).abs();
        }
    }

    ece
}

/// Fits temperature T minimizing NLL using golden-section search
pub fn fit_temperature(
    pairs: &[(Vec<f64>, usize)],
    min_t: f64,
    max_t: f64,
    max_iters: usize,
) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;

    let mut a = min_t;
    let mut b = max_t;

    let loss = |t: f64| -> f64 {
        let mut nll = 0.0;
        for (logits, target_idx) in pairs {
            if let Ok(probs) = scaled_softmax(logits, t) {
                let p = probs.get(*target_idx).copied().unwrap_or(1e-12).max(1e-12);
                nll -= p.ln();
            }
        }
        nll / pairs.len() as f64
    };

    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = loss(c);
    let mut fd = loss(d);

    for _ in 0..max_iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = loss(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = loss(d);
        }
        if (b - a).abs() < 1e-5 {
            break;
        }
    }

    (a + b) / 2.0
}

/// Fits optimal calibration temperatures individually per question type
/// (Choice, Boolean, Score, Numeric) to minimize Expected Calibration Error.
/// Inspired by Decider's multi-type calibration architecture.
pub fn fit_temperatures_by_type(
    samples_by_type: &std::collections::HashMap<String, Vec<(Vec<f64>, usize)>>,
    min_t: f64,
    max_t: f64,
    max_iters: usize,
) -> TypeTemperatureConfig {
    let mut config = TypeTemperatureConfig::default();
    if let Some(pairs) = samples_by_type.get("choice") {
        if !pairs.is_empty() {
            config.choice = fit_temperature(pairs, min_t, max_t, max_iters);
        }
    }
    if let Some(pairs) = samples_by_type
        .get("boolean")
        .or_else(|| samples_by_type.get("noul"))
    {
        if !pairs.is_empty() {
            config.boolean = fit_temperature(pairs, min_t, max_t, max_iters);
        }
    }
    if let Some(pairs) = samples_by_type
        .get("score")
        .or_else(|| samples_by_type.get("ordinal"))
    {
        if !pairs.is_empty() {
            config.score = fit_temperature(pairs, min_t, max_t, max_iters);
        }
    }
    if let Some(pairs) = samples_by_type.get("numeric") {
        if !pairs.is_empty() {
            config.numeric = fit_temperature(pairs, min_t, max_t, max_iters);
        }
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_temperature_config() {
        let config = TypeTemperatureConfig::default();
        assert_eq!(config.get_temperature("choice"), 1.48);
        assert_eq!(config.get_temperature("routing"), 1.48);
        assert_eq!(config.get_temperature("boolean"), 2.22);
        assert_eq!(config.get_temperature("noul"), 2.22);
        assert_eq!(config.get_temperature("score"), 1.38);
        assert_eq!(config.get_temperature("ordinal"), 1.38);
        assert_eq!(config.get_temperature("numeric"), 1.25);
    }

    #[test]
    fn test_fit_temperatures_by_type() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "choice".to_string(),
            vec![(vec![2.0, 0.5], 0), (vec![0.1, 2.5], 1)],
        );
        map.insert(
            "score".to_string(),
            vec![(vec![3.0, 1.0, 0.2], 0), (vec![0.2, 1.5, 3.2], 2)],
        );

        let fitted = fit_temperatures_by_type(&map, 0.5, 4.0, 15);
        assert!(fitted.choice >= 0.5 && fitted.choice <= 4.0);
        assert!(fitted.score >= 0.5 && fitted.score <= 4.0);
        // Untrained types retain defaults
        assert_eq!(fitted.boolean, 2.22);
        assert_eq!(fitted.numeric, 1.25);
    }

    #[test]
    fn test_calibration_errors() {
        assert!(resolve_temperature(Some(-1.0)).is_err());
        assert!(resolve_temperature(Some(f64::NAN)).is_err());

        assert!(scaled_softmax(&[], 1.0).is_err());
        assert!(scaled_softmax(&[1.0, 2.0], -1.0).is_err());

        let mut out = vec![0.0];
        assert!(scaled_softmax_slice(&[1.0, 2.0], 1.0, &mut out).is_err());
        let mut out2 = vec![0.0, 0.0];
        assert!(scaled_softmax_slice(&[1.0, 2.0], 0.0, &mut out2).is_err());

        assert_eq!(compute_ece(&[], &[], 5), 0.0);
        assert_eq!(compute_ece(&[0.9], &[true, false], 5), 0.0);
        assert_eq!(compute_ece(&[0.9], &[true], 0), 0.0);
    }

    #[test]
    fn test_hopper_family_calibrated_temperature() {
        let base = 2.0;
        assert!(family_calibrated_temperature("intent", base) < base);
        assert!(family_calibrated_temperature("policy", base) > base);
        assert!(family_calibrated_temperature("trap", base) > base);
        assert_eq!(family_calibrated_temperature("unknown", base), base);
    }

    #[test]
    fn test_djev_margin_temperature_dampening() {
        let base = 2.0;
        // Large margin (1.0) -> no dampening
        assert_eq!(dampen_temperature_by_margin(&[5.0, 4.0], base, 0.4), base);

        // Near tie (margin 0.05 < 0.4) -> dampened (higher temperature)
        let dampened = dampen_temperature_by_margin(&[5.05, 5.0], base, 0.4);
        assert!(dampened > base);
    }
}
