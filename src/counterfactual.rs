//! Counterfactual Shadow Inversion Probing
//!
//! Evaluates candidate robustness by projecting prompts through a counterfactual "shadow state"
//! where modal verbs, permissions, eligibility, and temporal polarities are inverted.
//!
//! Genuine policy-grounded decisions exhibit strong counterfactual divergence:
//! their probability collapses when the rules are reversed. Spurious false positives
//! (caused by raw keyword overlap or trap distractors) remain invariant to rule inversion,
//! allowing the engine to mathematically identify and penalize them at microsecond speeds.

use regex::Regex;
use std::sync::LazyLock;

/// Numerically stable softmax with temperature scaling.
pub fn compute_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = temperature.max(1e-4);
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
    exps.into_iter().map(|e| e / sum).collect()
}

static RE_POLARITY_TERMS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(is strictly ineligible|is not eligible|is eligible|eligible for|ineligible for|approved|denied|unauthorized|authorized|covered under|excluded from|violating policy|within policy|exceeding|within|prohibited|permitted|suspended|active|invalid|valid|non-compliant|compliant)\b")
        .expect("valid polarity terms regex")
});

/// Synthesizes a counterfactual shadow inversion of the text by inverting modal rules and polarities in a single pass.
pub fn generate_counterfactual_shadow(text: &str) -> String {
    RE_POLARITY_TERMS
        .replace_all(text, |caps: &regex::Captures| -> std::borrow::Cow<'static, str> {
            let lower = caps[1].to_lowercase();
            let repl: &'static str = match lower.as_str() {
                "is eligible" => "is strictly ineligible",
                "is not eligible" => "is fully eligible",
                "is strictly ineligible" => "is eligible",
                "eligible for" => "disqualified from",
                "ineligible for" => "eligible for",
                "approved" => "denied",
                "denied" => "approved",
                "authorized" => "unauthorized",
                "unauthorized" => "authorized",
                "covered under" => "excluded from",
                "excluded from" => "covered under",
                "within policy" => "violating policy",
                "violating policy" => "within policy",
                "within" => "exceeding",
                "exceeding" => "within",
                "permitted" => "prohibited",
                "prohibited" => "permitted",
                "active" => "suspended",
                "suspended" => "active",
                "valid" => "invalid",
                "invalid" => "valid",
                "compliant" => "non-compliant",
                "non-compliant" => "compliant",
                _ => "",
            };
            if repl.is_empty() {
                std::borrow::Cow::Owned(caps[0].to_string())
            } else {
                std::borrow::Cow::Borrowed(repl)
            }
        })
        .into_owned()
}

/// Computes counterfactual sensitivity metrics between original logits and shadow logits.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualResult {
    /// Symmetric KL divergence between original and shadow probability distributions.
    pub symmetric_kl: f64,
    /// Per-candidate sensitivity: P_orig(c) - P_shadow(c). Positive means candidate is sensitive to policy.
    pub candidate_sensitivities: Vec<f64>,
    /// Flags whether the distribution is invariant to counterfactual inversion (sign of spurious attractor).
    pub is_spurious_attractor: bool,
    /// Calibrated logits after penalizing spurious invariances.
    pub calibrated_logits: Vec<f64>,
}

/// Evaluates counterfactual divergence between original logits and shadow logits.
/// Penalizes candidates that remain confident even when policy conditions are inverted.
pub fn evaluate_counterfactual_divergence(
    original_logits: &[f64],
    shadow_logits: &[f64],
    temperature: f64,
) -> CounterfactualResult {
    let p_orig = compute_softmax(original_logits, temperature);
    let p_shadow = compute_softmax(shadow_logits, temperature);

    let n = p_orig.len().min(p_shadow.len());
    let mut sym_kl = 0.0;
    let mut sensitivities = Vec::with_capacity(n);
    let mut calibrated_logits = original_logits.to_vec();

    for i in 0..n {
        let p_o = p_orig[i].max(1e-8);
        let p_s = p_shadow[i].max(1e-8);

        // Symmetric KL divergence component: 0.5 * (p_o - p_s) * ln(p_o / p_s)
        let kl_term = 0.5 * (p_o - p_s) * (p_o.ln() - p_s.ln());
        sym_kl += kl_term;

        let delta = p_o - p_s;
        sensitivities.push(delta);

        // If candidate i has high probability in BOTH original and shadow states,
        // it is a spurious lexical attractor that ignores policy constraints.
        if p_o > 0.40 && p_s > 0.35 {
            // Apply counterfactual invariance penalty
            let penalty = 0.85 * (p_s - 0.20).max(0.0);
            calibrated_logits[i] -= penalty;
        }
    }

    let is_spurious_attractor = sym_kl < 0.08 && n >= 2;

    CounterfactualResult {
        symmetric_kl: sym_kl,
        candidate_sensitivities: sensitivities,
        is_spurious_attractor,
        calibrated_logits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_counterfactual_shadow() {
        let text = "Customer is eligible for refund under policy. Transaction is approved and active.";
        let shadow = generate_counterfactual_shadow(text);
        assert!(shadow.contains("ineligible"));
        assert!(shadow.contains("denied"));
        assert!(shadow.contains("suspended"));
    }

    #[test]
    fn test_counterfactual_divergence_penalizes_spurious_attractor() {
        // Candidate 0 has high probability in both original and shadow
        let orig_logits = vec![2.0, 0.5];
        let shadow_logits = vec![1.8, 0.6]; // Remains top candidate even in shadow!

        let res = evaluate_counterfactual_divergence(&orig_logits, &shadow_logits, 1.0);
        assert!(res.calibrated_logits[0] < orig_logits[0], "Spurious candidate must be penalized");
    }

    #[test]
    fn test_counterfactual_divergence_rewards_policy_grounded_decision() {
        // Candidate 0 is high in original, collapses in shadow
        let orig_logits = vec![2.5, 0.2];
        let shadow_logits = vec![0.1, 2.3]; // Shadow flips to candidate 1!

        let res = evaluate_counterfactual_divergence(&orig_logits, &shadow_logits, 1.0);
        assert!(!res.is_spurious_attractor);
        assert!(res.candidate_sensitivities[0] > 0.3);
    }
}
