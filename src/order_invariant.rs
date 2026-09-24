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

    #[inline(always)]
    pub fn score_candidate_raw(&self, id: &str, desc_str: &str) -> f64 {
        let mut logit: f64 = 0.5;

        // Exact ID match in premise
        if !id.starts_with("__") {
            let id_len = id.len();
            if id_len <= 32 {
                let mut buf = [0u8; 32];
                buf[..id_len].copy_from_slice(id.as_bytes());
                buf[..id_len].make_ascii_lowercase();
                if let Ok(id_lower) = std::str::from_utf8(&buf[..id_len]) {
                    if self.raw_lower.contains(id_lower) {
                        logit += 3.5;
                    }
                }
            } else if self.raw_lower.contains(&id.to_lowercase()) {
                logit += 3.5;
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
                            logit += 1.8;
                        } else if word_len >= 5 {
                            // Stem prefix check
                            if let Ok(stem_str) = std::str::from_utf8(&word_buf[..word_len - 1]) {
                                if self.raw_lower.contains(stem_str) {
                                    logit += 1.4;
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
