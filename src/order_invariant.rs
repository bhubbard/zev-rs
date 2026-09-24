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
            let check_window_start = actual_pos.saturating_sub(40);
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

        // Exact ID match & constituent token matching
        if !id.starts_with("__") {
            let id_lower = id.to_lowercase();
            if self.raw_lower.contains(&id_lower) {
                let w = self.recency_weight(&id_lower);
                if self.is_negated(&id_lower) {
                    logit -= 2.5 * w;
                } else {
                    logit += 3.5 * w;
                }
            } else {
                let id_spaced = id_lower.replace(['_', '-'], " ");
                if self.raw_lower.contains(&id_spaced) {
                    let w = self.recency_weight(&id_spaced);
                    if self.is_negated(&id_spaced) {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                } else {
                    for part in id_lower.split(['_', '-']) {
                        if part.len() > 3 && self.raw_lower.contains(part) {
                            let w = self.recency_weight(part);
                            if self.is_negated(part) {
                                logit -= 1.8 * w;
                            } else {
                                logit += 2.2 * w;
                            }
                        }
                    }
                }
            }
        }

        // Fast token/word matching using std SIMD substring search
        let desc = desc_str.as_bytes();
        let mut i = 0;
        let mut word_buf = [0u8; 32];
        while i < desc.len() {
            while i < desc.len() && !desc[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let word_start = i;
            while i < desc.len() && desc[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let word_len = i - word_start;
            if word_len > 3 {
                let word_slice = &desc[word_start..i];
                if word_len <= 32 {
                    word_buf[..word_len].copy_from_slice(word_slice);
                    word_buf[..word_len].make_ascii_lowercase();
                    if let Ok(word_str) = std::str::from_utf8(&word_buf[..word_len]) {
                        if self.raw_lower.contains(word_str) {
                            let w = self.recency_weight(word_str);
                            if self.is_negated(word_str) {
                                logit -= 1.8 * w; // negate resolved/negated feature
                            } else {
                                logit += 1.8 * w;
                            }
                        } else if word_len >= 5 {
                            // Stem prefix check
                            if let Ok(stem_str) = std::str::from_utf8(&word_buf[..word_len - 1]) {
                                if self.raw_lower.contains(stem_str) {
                                    let w = self.recency_weight(stem_str);
                                    if self.is_negated(stem_str) {
                                        logit -= 1.4 * w;
                                    } else {
                                        logit += 1.4 * w;
                                    }
                                }
                            }
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

