use crate::error::{Result, ZevError};
use crate::types::DEFAULT_CALIBRATED_TEMPERATURE;

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
        return Err(ZevError::DecodingError("Logits array cannot be empty".into()));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ZevError::CalibrationError("Temperature must be positive and finite".into()));
    }

    let mut out = vec![0.0; logits.len()];
    scaled_softmax_slice(logits, temperature, &mut out)?;
    Ok(out)
}

#[inline(always)]
pub fn scaled_softmax_slice(logits: &[f64], temperature: f64, out: &mut [f64]) -> Result<()> {
    if logits.is_empty() || logits.len() != out.len() {
        return Err(ZevError::DecodingError("Logits array cannot be empty".into()));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(ZevError::CalibrationError("Temperature must be positive and finite".into()));
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
        return Err(ZevError::DecodingError("Softmax normalization encountered non-finite sum".into()));
    }

    let inv_sum = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
    Ok(())
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
            if (conf >= bin_lower && conf < bin_upper) || (i == num_bins - 1 && conf >= bin_lower && conf <= 1.0) {
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
pub fn fit_temperature(pairs: &[(Vec<f64>, usize)], min_t: f64, max_t: f64, max_iters: usize) -> f64 {
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
