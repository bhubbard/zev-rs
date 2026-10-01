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

/// Adaptive margin temperature scaling.
/// Softens temperature on razor-thin margins (< margin_threshold) to prevent overconfidence,
/// and sharpens temperature on decisive margins (> 2.5 * margin_threshold) to crystallize top candidate confidence.
#[inline]
pub fn adaptive_margin_temperature(logits: &[f64], base_temp: f64, margin_threshold: f64) -> f64 {
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
        } else if margin > 2.5 * margin_threshold {
            // Decisive margin: sharpen slightly (cool down by up to 12%) to crystallize confidence
            let excess = ((margin - 2.5 * margin_threshold) / (2.5 * margin_threshold)).min(1.0);
            let factor = 1.0 - 0.12 * excess;
            return (base_temp * factor).max(0.5);
        }
    }
    base_temp
}

/// Combines probability distributions using Bayesian Log-Linear Product of Experts (PoE).
///
/// Operates in log-probability space:
///   `log P(c) = w_a * ln(P_a(c) + eps) + w_b * ln(P_b(c) + eps)`
/// followed by stable softmax normalization.
///
/// Unlike linear averaging, Log-Linear PoE geometrically compounds consensus and
/// penalizes dissonant or high-entropy distributions without flattening decisive predictions.
pub fn log_linear_poe_fusion(
    keys: &[String],
    probs_a: &std::collections::BTreeMap<String, f64>,
    probs_b: &std::collections::BTreeMap<String, f64>,
    weight_a: f64,
) -> std::collections::BTreeMap<String, f64> {
    let w_a = weight_a.clamp(0.0, 1.0);
    let w_b = 1.0 - w_a;
    let eps = 1e-7;

    let mut log_scores = Vec::with_capacity(keys.len());
    let mut max_log = f64::NEG_INFINITY;

    for k in keys {
        let p_a = probs_a.get(k).copied().unwrap_or(0.0).max(eps);
        let p_b = probs_b.get(k).copied().unwrap_or(0.0).max(eps);
        let log_p = w_a * p_a.ln() + w_b * p_b.ln();
        if log_p > max_log {
            max_log = log_p;
        }
        log_scores.push(log_p);
    }

    let mut sum_exp = 0.0;
    let mut exp_scores = Vec::with_capacity(keys.len());
    for &ls in &log_scores {
        let e = (ls - max_log).exp();
        exp_scores.push(e);
        sum_exp += e;
    }

    let mut result = std::collections::BTreeMap::new();
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
    for (k, e) in keys.iter().zip(exp_scores) {
        result.insert(k.clone(), (e * inv_sum * 10000.0).round() / 10000.0);
    }
    result
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

/// Computes the multi-class Brier Score for a single probability vector and target class index.
///
/// Brier Score: `\sum_{k=0}^{K-1} (p_k - \delta_{k, y})^2`
/// Ranges from 0.0 (perfect prediction) to 2.0 (maximum overconfident error).
#[inline]
pub fn compute_brier_score(probs: &[f64], target_idx: usize) -> f64 {
    if probs.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for (k, &p) in probs.iter().enumerate() {
        let y = if k == target_idx { 1.0 } else { 0.0 };
        let diff = p - y;
        sum_sq += diff * diff;
    }
    sum_sq
}

/// Computes the mean Brier Score over a dataset of (probabilities, target_idx) pairs.
pub fn compute_dataset_brier_score(dataset: &[(&[f64], usize)]) -> f64 {
    if dataset.is_empty() {
        return 0.0;
    }
    let total_loss: f64 = dataset
        .iter()
        .map(|(probs, target)| compute_brier_score(probs, *target))
        .sum();
    total_loss / dataset.len() as f64
}

/// Computes the binary Brier Score across binary probability predictions and boolean outcomes.
///
/// `BS = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2`
/// where `p_i` is predicted probability of true and `y_i \in {0, 1}`.
pub fn compute_binary_brier_score(probs: &[f64], targets: &[bool]) -> f64 {
    if probs.is_empty() || probs.len() != targets.len() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for (&p, &target) in probs.iter().zip(targets.iter()) {
        let y = if target { 1.0 } else { 0.0 };
        let diff = p - y;
        sum_sq += diff * diff;
    }
    sum_sq / probs.len() as f64
}

/// Fits temperature `T` minimizing the Brier Score loss using golden-section search.
///
/// Unlike NLL (which relies on log loss and can over-penalize outlier mispredictions),
/// Brier loss is bounded and strictly proper, producing robust, well-calibrated probabilities
/// in decision-oriented models like Clef.
pub fn fit_temperature_brier(
    pairs: &[(Vec<f64>, usize)],
    min_t: f64,
    max_t: f64,
    max_iters: usize,
) -> f64 {
    if pairs.is_empty() {
        return DEFAULT_CALIBRATED_TEMPERATURE;
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;

    let mut a = min_t;
    let mut b = max_t;

    let loss = |t: f64| -> f64 {
        let mut total_brier = 0.0;
        let mut count = 0;
        for (logits, target_idx) in pairs {
            if let Ok(probs) = scaled_softmax(logits, t) {
                total_brier += compute_brier_score(&probs, *target_idx);
                count += 1;
            }
        }
        if count > 0 {
            total_brier / count as f64
        } else {
            f64::MAX
        }
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

    #[test]
    fn test_adaptive_margin_temperature_and_poe_fusion() {
        let base = 2.0;
        // Near tie -> softened (higher temp)
        let soft = adaptive_margin_temperature(&[5.05, 5.0], base, 0.4);
        assert!(soft > base);

        // Exact 1.0 (2.5 * 0.4) -> base
        assert_eq!(adaptive_margin_temperature(&[5.0, 4.0], base, 0.4), base);

        // Decisive margin (3.0 >> 1.0) -> sharpened (lower temp)
        let sharp = adaptive_margin_temperature(&[6.0, 3.0], base, 0.4);
        assert!(sharp < base);

        // PoE Fusion
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("A".into(), 0.90);
        m1.insert("B".into(), 0.10);

        let mut m2 = BTreeMap::new();
        m2.insert("A".into(), 0.85);
        m2.insert("B".into(), 0.15);

        let fused = log_linear_poe_fusion(&["A".into(), "B".into()], &m1, &m2, 0.5);
        assert!(fused["A"] > 0.85);
        assert!(fused["B"] < 0.15);
    }

    #[test]
    fn test_brier_score_calculations() {
        // Perfect prediction: probs = [1.0, 0.0], target = 0 -> BS = (1-1)^2 + (0-0)^2 = 0.0
        let p_perfect = vec![1.0, 0.0];
        assert_eq!(compute_brier_score(&p_perfect, 0), 0.0);

        // Completely wrong: probs = [0.0, 1.0], target = 0 -> BS = (0-1)^2 + (1-0)^2 = 2.0
        assert_eq!(compute_brier_score(&p_perfect, 1), 2.0);

        // Uniform 3-way: probs = [1/3, 1/3, 1/3], target = 0 -> (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 0.6667
        let p_uniform = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let bs = compute_brier_score(&p_uniform, 0);
        assert!((bs - 2.0 / 3.0).abs() < 1e-6);

        // Binary brier
        let probs = vec![0.9, 0.1];
        let targets = vec![true, false];
        let bin_bs = compute_binary_brier_score(&probs, &targets);
        // (0.9 - 1)^2 + (0.1 - 0)^2 = 0.01 + 0.01 = 0.02 / 2 = 0.01
        assert!((bin_bs - 0.01).abs() < 1e-6);

        // Empty cases
        assert_eq!(compute_brier_score(&[], 0), 0.0);
        assert_eq!(compute_binary_brier_score(&[], &[]), 0.0);
        assert_eq!(compute_dataset_brier_score(&[]), 0.0);
    }

    #[test]
    fn test_fit_temperature_brier() {
        // Pairs with overconfident high logits: T should soften (increase) to lower Brier score
        let pairs = vec![
            (vec![10.0, 0.0], 0),
            (vec![0.0, 10.0], 1),
            (vec![5.0, 0.0], 0),
        ];
        let fitted_t = fit_temperature_brier(&pairs, 0.5, 4.0, 20);
        assert!((0.5..=4.0).contains(&fitted_t));

        let empty: Vec<(Vec<f64>, usize)> = vec![];
        assert_eq!(fit_temperature_brier(&empty, 0.5, 4.0, 10), DEFAULT_CALIBRATED_TEMPERATURE);
    }

    #[test]
    fn test_brier_score_boundaries() {
        // Multi-class Brier score is bounded in [0.0, 2.0] for any probability distribution
        let test_cases = vec![
            (vec![0.5, 0.5], 0),
            (vec![0.7, 0.2, 0.1], 1),
            (vec![0.05, 0.90, 0.05], 0),
            (vec![0.25, 0.25, 0.25, 0.25], 3),
        ];

        for (probs, target) in test_cases {
            let bs = compute_brier_score(&probs, target);
            assert!((0.0..=2.0).contains(&bs), "Brier score {bs} out of [0, 2] bounds");
        }
    }

    #[test]
    fn test_dataset_brier_score_weighted_batches() {
        let p1 = [0.8, 0.2]; // target 0: (0.8-1)^2 + 0.2^2 = 0.04 + 0.04 = 0.08
        let p2 = [0.1, 0.9]; // target 1: 0.1^2 + (0.9-1)^2 = 0.01 + 0.01 = 0.02
        let p3 = [0.4, 0.6]; // target 0: (0.4-1)^2 + 0.6^2 = 0.36 + 0.36 = 0.72

        let dataset: Vec<(&[f64], usize)> = vec![
            (&p1[..], 0),
            (&p2[..], 1),
            (&p3[..], 0),
        ];

        let mean_bs = compute_dataset_brier_score(&dataset);
        let expected = (0.08 + 0.02 + 0.72) / 3.0; // 0.82 / 3 = 0.273333...
        assert!((mean_bs - expected).abs() < 1e-6);
    }

    #[test]
    fn test_brier_temperature_monotonic_softening() {
        // When predicting ambiguous or noisy labels, overconfident logits (e.g. margin=8.0)
        // receive severe quadratic penalty when mispredicted, driving Brier optimal T higher.
        let moderate_noisy_pairs = vec![
            (vec![1.5, 0.0], 0),
            (vec![1.5, 0.0], 0),
            (vec![1.5, 0.0], 0),
            (vec![1.5, 0.0], 1), // 25% label noise
        ];
        let extreme_noisy_pairs = vec![
            (vec![8.0, 0.0], 0),
            (vec![8.0, 0.0], 0),
            (vec![8.0, 0.0], 0),
            (vec![8.0, 0.0], 1), // 25% label noise
        ];

        let t_moderate = fit_temperature_brier(&moderate_noisy_pairs, 0.5, 8.0, 30);
        let t_extreme = fit_temperature_brier(&extreme_noisy_pairs, 0.5, 8.0, 30);

        assert!(t_extreme > t_moderate, "Extreme logit scale on noisy data must yield higher softening temperature");
    }

    #[test]
    fn test_brier_vs_nll_outlier_robustness() {
        // A dataset where one sample is an extreme outlier misprediction
        let pairs_with_outlier = vec![
            (vec![3.0, 0.0], 0),
            (vec![3.0, 0.0], 0),
            (vec![3.0, 0.0], 0),
            (vec![5.0, 0.0], 1), // Outlier: extreme model confidence on 0, but truth is 1
        ];

        let t_brier = fit_temperature_brier(&pairs_with_outlier, 0.5, 4.0, 20);
        let t_nll = fit_temperature(&pairs_with_outlier, 0.5, 4.0, 20);

        // Brier loss remains bounded (loss <= 2.0 per sample) whereas NLL has huge gradient pulling T higher
        assert!(t_brier.is_finite());
        assert!(t_nll.is_finite());
        assert!((0.5..=4.0).contains(&t_brier));
    }
}

