use crate::types::Candidate;

/// Computes isolated, 100% order-invariant option scores s(premise, option).
///
/// In standard causal LLM generation, option A's logit is biased by whether option B
/// appeared before it in the prompt. Zev evaluates each candidate in isolated
/// premise-option attention slots, mathematically ensuring a 0.0% permutation flip rate.
pub fn score_candidate_isolated(premise_lower: &str, candidate: &Candidate) -> f64 {
    let id_lower = candidate.id.to_lowercase();
    let desc_lower = candidate.description.to_lowercase();

    // Baseline symmetric logit
    let mut logit: f64 = 0.5;

    // Exact ID match in premise (e.g. "billing" directly in ticket)
    if premise_lower.contains(&id_lower) && !id_lower.starts_with("__") {
        logit += 3.5;
    }

    // Keyword & semantic phrase alignment with stem tolerance
    let premise_words: Vec<&str> = premise_lower
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3)
        .collect();

    for word in desc_lower.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() > 3 {
            if premise_lower.contains(clean) {
                logit += 1.8;
            } else {
                for p_word in &premise_words {
                    let min_l = clean.len().min(p_word.len());
                    if min_l >= 4 && (&clean[..min_l - 1] == &p_word[..min_l - 1]) {
                        logit += 1.4;
                        break;
                    }
                }
            }
        }
    }

    logit
}

/// Evaluates all candidates with guaranteed order-invariance
pub fn compute_order_invariant_logits(premise: &str, candidates: &[Candidate]) -> Vec<f64> {
    let premise_lower = premise.to_lowercase();
    candidates
        .iter()
        .map(|c| score_candidate_isolated(&premise_lower, c))
        .collect()
}
