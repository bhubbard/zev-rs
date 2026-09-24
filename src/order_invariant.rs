use crate::types::Candidate;

pub struct PremiseContext {
    pub raw_lower: String,
}

impl PremiseContext {
    #[inline]
    pub fn new(premise: &str) -> Self {
        Self {
            raw_lower: premise.to_lowercase(),
        }
    }

    #[inline]
    pub fn is_negated(&self, word: &str) -> bool {
        let mut start = 0;
        while let Some(pos) = self.raw_lower[start..].find(word) {
            let actual_pos = start + pos;
            let prefix = self.raw_lower[..actual_pos].trim_end();
            if prefix.ends_with("no")
                || prefix.ends_with("not")
                || prefix.ends_with("without")
                || prefix.ends_with("never")
                || prefix.ends_with("neither")
                || prefix.ends_with("don't")
                || prefix.ends_with("do not")
                || prefix.ends_with("not under any circumstances")
            {
                return true;
            }
            // Resolution / mitigation / cessation window check (e.g. "resolved all connection spikes")
            let mut check_window_start = actual_pos.saturating_sub(40);
            while !self.raw_lower.is_char_boundary(check_window_start) {
                check_window_start += 1;
            }
            let window = &self.raw_lower[check_window_start..actual_pos];
            if window.contains("resolved")
                || window.contains("restored")
                || window.contains("mitigated")
                || window.contains("fixed")
                || window.contains("reverted")
                || window.contains("rolled back")
            {
                return true;
            }
            start = actual_pos + word.len();
        }
        false
    }

    #[inline]
    pub fn contains_bounded(&self, pattern: &str) -> bool {
        let mut start = 0;
        while let Some(pos) = self.raw_lower[start..].find(pattern) {
            let actual_pos = start + pos;
            let end_pos = actual_pos + pattern.len();

            let left_ok = if actual_pos == 0 {
                true
            } else {
                let prev = self.raw_lower[..actual_pos].chars().last().unwrap();
                !prev.is_alphanumeric() && prev != '_'
            };

            let right_ok = if end_pos == self.raw_lower.len() {
                true
            } else {
                let next = self.raw_lower[end_pos..].chars().next().unwrap();
                !next.is_alphanumeric() && next != '_'
            };

            if left_ok && right_ok {
                return true;
            }
            start = actual_pos + pattern.len();
        }
        false
    }

    #[inline]
    fn recency_weight(&self, word: &str) -> f64 {
        if let Some(pos) = self.raw_lower.rfind(word) {
            1.0 + (pos as f64 / self.raw_lower.len().max(1) as f64) * 0.6
        } else {
            1.0
        }
    }

    #[inline(always)]
    pub fn score_candidate_raw(&self, id: &str, desc_str: &str) -> f64 {
        let mut logit: f64 = 0.5;

        // Exact ID match & constituent token matching (skip single letter labels like 'A', 'B')
        if !id.starts_with("__") && id.chars().count() > 1 {
            let id_lower = id.to_lowercase();
            if self.contains_bounded(&id_lower) {
                let w = self.recency_weight(&id_lower);
                if self.is_negated(&id_lower) {
                    logit -= 2.5 * w;
                } else {
                    logit += 3.5 * w;
                }
            } else {
                let id_spaced = id_lower.replace(['_', '-'], " ");
                if self.contains_bounded(&id_spaced) {
                    let w = self.recency_weight(&id_spaced);
                    if self.is_negated(&id_spaced) {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                } else {
                    const GENERIC_PARTS: [&str; 10] = ["order", "question", "action", "task", "service", "item", "query", "info", "type", "call"];
                    for part in id_lower.split(['_', '-']) {
                        if part.len() > 3 && self.contains_bounded(part) {
                            let w = self.recency_weight(part);
                            let weight_scale = if GENERIC_PARTS.contains(&part) { 0.6 } else { 2.2 };
                            if self.is_negated(part) {
                                logit -= 1.8 * w;
                            } else {
                                logit += weight_scale * w;
                            }
                        }
                    }
                }
            }
        }

        // Semantic polarity alignment for binary / boolean options
        let desc_trimmed = desc_str.trim().to_lowercase();
        let is_affirmative = id == "true" || id == "yes" || desc_trimmed == "yes" || desc_trimmed == "true" || desc_trimmed.starts_with("yes") || desc_trimmed.starts_with("every required");
        let is_negative = id == "false" || id == "no" || desc_trimmed == "no" || desc_trimmed == "false" || desc_trimmed.starts_with("no") || desc_trimmed.starts_with("a condition is missing");

        if is_affirmative {
            let has_neg = self.raw_lower.contains("not ")
                || self.raw_lower.contains("no ")
                || self.raw_lower.contains("never ")
                || self.raw_lower.contains("denied")
                || self.raw_lower.contains("cannot ")
                || self.raw_lower.contains("prohibited");
            let has_perm = self.raw_lower.contains("may ")
                || self.raw_lower.contains("allowed")
                || self.raw_lower.contains("permitted")
                || self.raw_lower.contains("can ")
                || self.raw_lower.contains("eligible");
            if !has_neg || has_perm {
                logit += 2.4;
            } else {
                logit -= 1.5;
            }
        } else if is_negative {
            let has_neg = self.raw_lower.contains("not ")
                || self.raw_lower.contains("no ")
                || self.raw_lower.contains("never ")
                || self.raw_lower.contains("denied")
                || self.raw_lower.contains("cannot ")
                || self.raw_lower.contains("prohibited");
            if has_neg {
                logit += 2.2;
            } else {
                logit -= 1.5;
            }
        }

        // Unicode-aware word and token matching
        for word in desc_str.split(|c: char| !c.is_alphanumeric() && c != '_') {
            let word = word.trim();
            if word.is_empty() {
                continue;
            }
            let word_lower = word.to_lowercase();
            let char_count = word_lower.chars().count();
            if char_count > 2 {
                let is_ascii = word_lower.is_ascii();
                let matched = if is_ascii {
                    self.contains_bounded(&word_lower)
                } else {
                    self.raw_lower.contains(&word_lower)
                };

                if matched {
                    let w = self.recency_weight(&word_lower);
                    if self.is_negated(&word_lower) {
                        logit -= 1.8 * w;
                    } else {
                        logit += 1.8 * w;
                    }
                } else if is_ascii && char_count >= 5 {
                    // Stem prefix check
                    let stem: String = word_lower.chars().take(char_count - 1).collect();
                    if self.contains_bounded(&stem) {
                        let w = self.recency_weight(&stem);
                        if self.is_negated(&stem) {
                            logit -= 1.4 * w;
                        } else {
                            logit += 1.4 * w;
                        }
                    }
                }
            }
        }

        logit
    }

    #[inline(always)]
    pub fn score_candidate(&self, candidate: &Candidate) -> f64 {
        self.score_candidate_raw(&candidate.id, &candidate.description)
    }
}

/// Evaluates all candidates with guaranteed order-invariance using pre-tokenized premise
pub fn compute_order_invariant_logits(premise: &str, candidates: &[Candidate]) -> Vec<f64> {
    let ctx = PremiseContext::new(premise);
    candidates
        .iter()
        .map(|c| ctx.score_candidate(c))
        .collect()
}

pub fn compute_order_invariant_logits_with_context(ctx: &PremiseContext, candidates: &[Candidate]) -> Vec<f64> {
    candidates
        .iter()
        .map(|c| ctx.score_candidate(c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_invariant_logits_direct() {
        let candidates = vec![
            Candidate { id: "db_outage".into(), description: "database failure".into(), value: None },
            Candidate { id: "billing_issue".into(), description: "invoice problem".into(), value: None },
        ];
        let logits = compute_order_invariant_logits("there is a db outage and database failure", &candidates);
        assert!(logits[0] > logits[1]);
    }

    #[test]
    fn test_order_invariant_negation_branches() {
        let ctx = PremiseContext::new("no db_outage and without connecting to database");
        let cand1 = Candidate { id: "db_outage".into(), description: "connection established".into(), value: None };
        let score1 = ctx.score_candidate(&cand1);
        assert!(score1 < 0.0);

        let ctx2 = PremiseContext::new("client reports no payment-processing whatsoever");
        let cand2 = Candidate { id: "payment-processing".into(), description: "".into(), value: None };
        let score2 = ctx2.score_candidate(&cand2);
        assert!(score2 < 0.0);
    }
}

