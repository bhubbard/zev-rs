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
            start = actual_pos + word.len();
        }
        false
    }

    #[inline(always)]
    pub fn score_candidate_raw(&self, id: &str, desc_str: &str) -> f64 {
        let mut logit: f64 = 0.5;

        // Exact ID match & constituent token matching
        if !id.starts_with("__") {
            let id_lower = id.to_lowercase();
            if self.raw_lower.contains(&id_lower) {
                if self.is_negated(&id_lower) {
                    logit -= 2.5;
                } else {
                    logit += 3.5;
                }
            } else {
                let id_spaced = id_lower.replace(['_', '-'], " ");
                if self.raw_lower.contains(&id_spaced) {
                    if self.is_negated(&id_spaced) {
                        logit -= 2.5;
                    } else {
                        logit += 3.5;
                    }
                } else {
                    for part in id_lower.split(['_', '-']) {
                        if part.len() > 3 && self.raw_lower.contains(part) {
                            if self.is_negated(part) {
                                logit -= 1.5;
                            } else {
                                logit += 1.8;
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
                            if self.is_negated(word_str) {
                                logit -= 1.8; // negate feature
                            } else {
                                logit += 1.8;
                            }
                        } else if word_len >= 5 {
                            // Stem prefix check
                            if let Ok(stem_str) = std::str::from_utf8(&word_buf[..word_len - 1]) {
                                if self.raw_lower.contains(stem_str) {
                                    if self.is_negated(stem_str) {
                                        logit -= 1.4;
                                    } else {
                                        logit += 1.4;
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
