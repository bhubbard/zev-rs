use std::collections::HashMap;
use smallvec::SmallVec;
use crate::types::Candidate;

const PREFIX_NEGATORS: &[&str] = &[
    "not under any circumstances",
    "do not",
    "don't",
    "no",
    "not",
    "without",
    "never",
    "neither",
    "cannot",
    "unable",
    "unpaid",
    "disabled",
    "inactive",
    "absent",
];

const SCOPED_NEGATORS: &[&str] = &[
    "not asking for",
    "don't want",
    "do not want",
    "no longer",
    "not looking for",
    "no need for",
    "do not need",
    "rather than",
    "stop renewing",
    "refused",
    "declined",
    "resolved",
    "restored",
    "mitigated",
    "fixed",
    "reverted",
    "rolled back",
    "cannot",
    "unable",
    "absent",
    "absence",
];

#[inline]
fn starts_with_word(text: &str, word: &str) -> bool {
    if !text.starts_with(word) {
        return false;
    }
    if text.len() == word.len() {
        return true;
    }
    let next = text[word.len()..].chars().next().unwrap();
    !next.is_alphanumeric() && next != '_'
}

#[inline]
fn ends_with_word(text: &str, word: &str) -> bool {
    if !text.ends_with(word) {
        return false;
    }
    if text.len() == word.len() {
        return true;
    }
    let prev = text[..text.len() - word.len()].chars().last().unwrap();
    !prev.is_alphanumeric() && prev != '_'
}

#[inline]
fn contains_bounded_in(haystack: &str, pattern: &str) -> bool {
    if pattern.is_empty() || haystack.len() < pattern.len() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(pattern) {
        let actual_pos = start + pos;
        let end_pos = actual_pos + pattern.len();

        let left_ok = if actual_pos == 0 {
            true
        } else {
            let prev = haystack[..actual_pos].chars().last().unwrap();
            !prev.is_alphanumeric() && prev != '_'
        };

        let right_ok = if end_pos == haystack.len() {
            true
        } else {
            let next = haystack[end_pos..].chars().next().unwrap();
            !next.is_alphanumeric() && next != '_'
        };

        if left_ok && right_ok {
            return true;
        }
        start = actual_pos + pattern.len().max(1);
    }
    false
}

fn find_clause_start(text: &str) -> usize {
    let mut boundary_end = 0;

    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b';' || b == b'?' || b == b'!' {
            boundary_end = boundary_end.max(i + 1);
        } else if b == b'.' {
            let prev_is_digit = i > 0 && bytes[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
            if !(prev_is_digit && next_is_digit) {
                boundary_end = boundary_end.max(i + 1);
            }
        }
    }

    // Check for whole word "but"
    let mut search_start = 0;
    while let Some(pos) = text[search_start..].find("but") {
        let actual_pos = search_start + pos;
        let end_pos = actual_pos + 3;

        let left_ok = if actual_pos == 0 {
            true
        } else {
            let prev = text[..actual_pos].chars().last().unwrap();
            !prev.is_alphanumeric() && prev != '_'
        };

        let right_ok = if end_pos == text.len() {
            true
        } else {
            let next = text[end_pos..].chars().next().unwrap();
            !next.is_alphanumeric() && next != '_'
        };

        if left_ok && right_ok {
            boundary_end = boundary_end.max(end_pos);
        }
        search_start = actual_pos + 3;
    }

    boundary_end
}

pub struct PremiseContext {
    pub raw_lower: String,
    pub token_positions: HashMap<String, SmallVec<[u32; 4]>>,
}

impl PremiseContext {
    #[inline]
    pub fn new(premise: &str) -> Self {
        let raw_lower = premise.to_lowercase();
        let mut token_positions: HashMap<String, SmallVec<[u32; 4]>> = HashMap::new();
        let mut in_token = false;
        let mut token_start = 0;

        for (byte_idx, ch) in raw_lower.char_indices() {
            if ch.is_alphanumeric() || ch == '_' {
                if !in_token {
                    in_token = true;
                    token_start = byte_idx;
                }
            } else if in_token {
                in_token = false;
                let token = &raw_lower[token_start..byte_idx];
                token_positions
                    .entry(token.to_string())
                    .or_default()
                    .push(token_start as u32);
            }
        }
        if in_token {
            let token = &raw_lower[token_start..];
            token_positions
                .entry(token.to_string())
                .or_default()
                .push(token_start as u32);
        }

        Self {
            raw_lower,
            token_positions,
        }
    }

    #[inline]
    pub fn is_negated(&self, word: &str) -> bool {
        if word.is_empty() {
            return false;
        }

        // Morphological negation normalization
        match word {
            "paid" if self.contains_bounded("unpaid") => return true,
            "enabled" if self.contains_bounded("disabled") => return true,
            "active" if self.contains_bounded("inactive") => return true,
            "present" if self.contains_bounded("absent") => return true,
            "able" if self.contains_bounded("unable") || self.contains_bounded("cannot") => return true,
            _ => {}
        }

        let word_lower;
        let w = if word.chars().any(|c| c.is_uppercase()) {
            word_lower = word.to_lowercase();
            &word_lower
        } else {
            word
        };

        if let Some(positions) = self.token_positions.get(w) {
            for &pos in positions {
                let actual_pos = pos as usize;
                let clause_start = find_clause_start(&self.raw_lower[..actual_pos]);

                let prefix = self.raw_lower[clause_start..actual_pos].trim_end();
                for &neg in PREFIX_NEGATORS {
                    if ends_with_word(prefix, neg) {
                        return true;
                    }
                }

                // Window check for scoped negations and disclaimers (Winnow-12B protocol)
                let mut check_window_start = actual_pos.saturating_sub(45).max(clause_start);
                while !self.raw_lower.is_char_boundary(check_window_start) {
                    check_window_start += 1;
                }
                let window = &self.raw_lower[check_window_start..actual_pos];
                for &neg in SCOPED_NEGATORS {
                    if contains_bounded_in(window, neg) {
                        return true;
                    }
                }
            }
            return false;
        }

        // If w is a single word token and not in token_positions, it cannot appear as a bounded token in raw_lower
        if w.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return false;
        }

        // Fallback for multi-word or non-standard patterns
        let mut start = 0;
        while let Some(pos) = self.raw_lower[start..].find(w) {
            let actual_pos = start + pos;
            let end_pos = actual_pos + w.len();

            // Word-boundary isolation: check that this occurrence is bounded
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

            if !left_ok || !right_ok {
                start = actual_pos + w.len().max(1);
                continue;
            }

            let clause_start = find_clause_start(&self.raw_lower[..actual_pos]);

            let prefix = self.raw_lower[clause_start..actual_pos].trim_end();
            for &neg in PREFIX_NEGATORS {
                if ends_with_word(prefix, neg) {
                    return true;
                }
            }

            // Window check for scoped negations and disclaimers (Winnow-12B protocol)
            let mut check_window_start = actual_pos.saturating_sub(45).max(clause_start);
            while !self.raw_lower.is_char_boundary(check_window_start) {
                check_window_start += 1;
            }
            let window = &self.raw_lower[check_window_start..actual_pos];
            for &neg in SCOPED_NEGATORS {
                if contains_bounded_in(window, neg) {
                    return true;
                }
            }

            start = actual_pos + w.len().max(1);
        }
        false
    }

    #[inline]
    pub fn contains_bounded(&self, pattern: &str) -> bool {
        if pattern.is_empty() {
            return false;
        }
        if pattern.chars().all(|c| c.is_alphanumeric() || c == '_') {
            if self.token_positions.contains_key(pattern) {
                return true;
            }
            if pattern.chars().any(|c| c.is_uppercase()) {
                let pat_lower = pattern.to_lowercase();
                return self.token_positions.contains_key(&pat_lower);
            }
            return false;
        }
        if pattern.chars().any(|c| c.is_uppercase()) {
            let pat_lower = pattern.to_lowercase();
            contains_bounded_in(&self.raw_lower, &pat_lower)
        } else {
            contains_bounded_in(&self.raw_lower, pattern)
        }
    }

    #[inline]
    fn recency_weight(&self, word: &str) -> f64 {
        if word.is_empty() {
            return 1.0;
        }
        let last_pos = if let Some(positions) = self.token_positions.get(word) {
            positions.last().copied().map(|p| p as usize)
        } else if word.chars().any(|c| c.is_uppercase()) {
            let word_lower = word.to_lowercase();
            if let Some(positions) = self.token_positions.get(&word_lower) {
                positions.last().copied().map(|p| p as usize)
            } else if word_lower.chars().all(|c| c.is_alphanumeric() || c == '_') {
                None
            } else {
                self.find_last_bounded_pos(&word_lower)
            }
        } else if word.chars().all(|c| c.is_alphanumeric() || c == '_') {
            None
        } else {
            self.find_last_bounded_pos(word)
        };

        if let Some(pos) = last_pos {
            1.0 + (pos as f64 / self.raw_lower.len().max(1) as f64) * 0.6
        } else {
            1.0
        }
    }

    fn find_last_bounded_pos(&self, word: &str) -> Option<usize> {
        let mut last_pos = None;
        let mut start = 0;
        while let Some(pos) = self.raw_lower[start..].find(word) {
            let actual_pos = start + pos;
            let end_pos = actual_pos + word.len();
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
                last_pos = Some(actual_pos);
            }
            start = actual_pos + word.len().max(1);
        }
        last_pos
    }

    #[inline(always)]
    pub fn score_candidate_raw(&self, id: &str, desc_str: &str) -> f64 {
        let mut logit: f64 = 0.5;

        // Exact ID match & constituent token matching
        if !id.starts_with("__") && !id.is_empty() {
            let id_lower = id.to_lowercase();
            let id_len = id_lower.chars().count();

            // Morphological Negation Normalization on candidate ID
            let has_morph_neg = match id_lower.as_str() {
                "paid" => self.contains_bounded("unpaid"),
                "enabled" => self.contains_bounded("disabled"),
                "active" => self.contains_bounded("inactive"),
                "present" => self.contains_bounded("absent"),
                "able" => self.contains_bounded("unable") || self.contains_bounded("cannot"),
                _ => false,
            };

            if has_morph_neg {
                logit -= 2.5;
            } else if id_len <= 2 {
                // Word-boundary isolation for short/single-character options (length <= 2, e.g. 'M', 'L', 'S', '0', '1', '2', '3')
                // Require strict word boundaries \b so they do not collide with letters inside words (e.g. 'L' in 'blue' or 'please')
                if self.contains_bounded(&id_lower) {
                    let w = self.recency_weight(&id_lower);
                    if self.is_negated(&id_lower) {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                }
            } else {
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
        }

        // Semantic polarity alignment for binary / boolean options
        let desc_trimmed = desc_str.trim().to_lowercase();
        let is_bool_opt = id == "true"
            || id == "false"
            || id == "yes"
            || id == "no"
            || desc_trimmed == "yes"
            || desc_trimmed == "no"
            || desc_trimmed == "true"
            || desc_trimmed == "false";

        let is_affirmative = is_bool_opt
            && (id == "true"
                || id == "yes"
                || (id != "false"
                    && id != "no"
                    && (starts_with_word(&desc_trimmed, "yes")
                        || desc_trimmed == "true"
                        || desc_trimmed.starts_with("every required"))));

        let is_negative = is_bool_opt
            && (id == "false"
                || id == "no"
                || (id != "true"
                    && id != "yes"
                    && (starts_with_word(&desc_trimmed, "no")
                        || desc_trimmed == "false"
                        || desc_trimmed.starts_with("a condition is missing"))));

        if is_affirmative || is_negative {
            // Morphological negation normalization: include unpaid, disabled, inactive, absent, unable, cannot
            let has_neg = self.contains_bounded("not")
                || self.contains_bounded("no")
                || self.contains_bounded("never")
                || self.contains_bounded("denied")
                || self.contains_bounded("cannot")
                || self.contains_bounded("unable")
                || self.contains_bounded("unpaid")
                || self.contains_bounded("disabled")
                || self.contains_bounded("inactive")
                || self.contains_bounded("absent")
                || self.contains_bounded("prohibited");

            let has_perm = self.contains_bounded("may")
                || self.contains_bounded("allowed")
                || self.contains_bounded("permitted")
                || self.contains_bounded("can")
                || self.contains_bounded("eligible")
                || self.contains_bounded("paid in full")
                || self.contains_bounded("is enabled");

            if is_affirmative {
                if !has_neg || has_perm {
                    logit += 2.4;
                } else {
                    logit -= 1.5;
                }
            } else if is_negative {
                if has_neg && !has_perm {
                    logit += 2.2;
                } else {
                    logit -= 1.5;
                }
            }
        }

        // Unicode-aware word and token matching
        for word in desc_str.split(|c: char| !c.is_alphanumeric() && c != '_') {
            let word = word.trim();
            if word.is_empty() {
                continue;
            }
            let word_lower;
            let word_str: &str = if word.chars().any(|c| c.is_uppercase()) {
                word_lower = word.to_lowercase();
                &word_lower
            } else {
                word
            };
            let char_count = word_str.chars().count();

            // Morphological negation normalization on description words
            let has_morph_neg = match word_str {
                "paid" => self.contains_bounded("unpaid"),
                "enabled" => self.contains_bounded("disabled"),
                "active" => self.contains_bounded("inactive"),
                "present" => self.contains_bounded("absent"),
                "able" => self.contains_bounded("unable") || self.contains_bounded("cannot"),
                _ => false,
            };
            if has_morph_neg {
                let w = self.recency_weight(word_str);
                logit -= 1.8 * w;
                continue;
            }

            if char_count <= 2 {
                // Word-boundary isolation for short/single-character options (length <= 2)
                const COMMON_STOPWORDS: [&str; 18] = [
                    "a", "an", "in", "on", "at", "to", "is", "it", "or", "of", "by", "as", "if", "be", "do", "we", "he", "so"
                ];
                if !COMMON_STOPWORDS.contains(&word_str) && self.contains_bounded(word_str) {
                    let w = self.recency_weight(word_str);
                    if self.is_negated(word_str) {
                        logit -= 1.8 * w;
                    } else {
                        logit += 1.8 * w;
                    }
                }
            } else {
                let is_ascii = word_str.is_ascii();
                let matched = if is_ascii {
                    self.contains_bounded(word_str)
                } else {
                    self.raw_lower.contains(word_str)
                };

                if matched {
                    let w = self.recency_weight(word_str);
                    if self.is_negated(word_str) {
                        logit -= 1.8 * w;
                    } else {
                        logit += 1.8 * w;
                    }
                } else if is_ascii && char_count >= 5 {
                    // Stem prefix check
                    let stem = &word_str[..word_str.len() - 1];
                    if self.contains_bounded(stem) {
                        let w = self.recency_weight(stem);
                        if self.is_negated(stem) {
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

    #[test]
    fn test_word_boundary_isolation_short_options() {
        // "Could I get the blue shirt in size M, please?"
        // Option 'M' should match "size M,"
        // Option 'L' should NOT match inside "blue" or "please"
        let ctx = PremiseContext::new("Could I get the blue shirt in size M, please?");
        let cand_m = Candidate { id: "M".into(), description: "Medium".into(), value: None };
        let cand_l = Candidate { id: "L".into(), description: "Large".into(), value: None };
        let cand_s = Candidate { id: "S".into(), description: "Small".into(), value: None };

        let score_m = ctx.score_candidate(&cand_m);
        let score_l = ctx.score_candidate(&cand_l);
        let score_s = ctx.score_candidate(&cand_s);

        assert!(score_m > score_l, "M ({score_m}) must score higher than L ({score_l})");
        assert!(score_m > score_s, "M ({score_m}) must score higher than S ({score_s})");
        assert!(!ctx.contains_bounded("l"), "L must not match inside blue or please");
        assert!(ctx.contains_bounded("m"), "M must match isolated word boundary");
    }

    #[test]
    fn test_morphological_negation_normalization() {
        let ctx_unpaid = PremiseContext::new("Invoice 2026-045. Payment status: unpaid.");
        assert!(ctx_unpaid.is_negated("paid"));
        let cand_paid = Candidate { id: "paid".into(), description: "Invoice paid".into(), value: None };
        let score_paid = ctx_unpaid.score_candidate(&cand_paid);
        assert!(score_paid < 0.0, "paid score should be penalized under unpaid: {score_paid}");

        let ctx_disabled = PremiseContext::new("Account settings: two-factor authentication is disabled.");
        assert!(ctx_disabled.is_negated("enabled"));

        let ctx_absent = PremiseContext::new("Proof of purchase is absent.");
        assert!(ctx_absent.is_negated("present"));

        let ctx_inactive = PremiseContext::new("Account is inactive.");
        assert!(ctx_inactive.is_negated("active"));

        let ctx_unable = PremiseContext::new("User is unable to login; cannot access dashboard.");
        assert!(ctx_unable.is_negated("able"));
    }

    #[test]
    fn test_r1_p03_negation_word_tokens_and_boundaries() {
        // P03 probe cases:
        // 1. "ticket unresolved: database outage" -> "database" must NOT be negated
        let ctx1 = PremiseContext::new("ticket unresolved: database outage");
        assert!(!ctx1.is_negated("database"), "'unresolved' must not negate 'database'");

        // 2. "refund for the casino deposit" -> "deposit" must NOT be negated
        let ctx2 = PremiseContext::new("refund for the casino deposit");
        assert!(!ctx2.is_negated("deposit"), "'casino' must not negate 'deposit'");

        // 3. "the prefixed invoice number" -> "invoice" must NOT be negated
        let ctx3 = PremiseContext::new("the prefixed invoice number");
        assert!(!ctx3.is_negated("invoice"), "'prefixed' must not negate 'invoice'");

        // Clause boundaries: '.', ';', 'but'
        let ctx_dot = PremiseContext::new("ticket resolved. database outage ongoing");
        assert!(!ctx_dot.is_negated("database"), "sentence boundary '.' must stop negation");

        let ctx_semi = PremiseContext::new("ticket resolved; database outage ongoing");
        assert!(!ctx_semi.is_negated("database"), "clause boundary ';' must stop negation");

        let ctx_but = PremiseContext::new("ticket was resolved, but database outage still ongoing");
        assert!(!ctx_but.is_negated("database"), "clause boundary 'but' must stop negation");

        // Legitimate negation within same clause
        let ctx_neg = PremiseContext::new("ticket was not resolved; no database access");
        assert!(ctx_neg.is_negated("database"), "'no database' must be negated");

        // End-to-end P03: "ticket unresolved: database outage still ongoing"
        // database_outage must score higher than billing
        let candidates = vec![
            Candidate { id: "database_outage".into(), description: "database outage".into(), value: None },
            Candidate { id: "billing".into(), description: "billing issue".into(), value: None },
        ];
        let logits = compute_order_invariant_logits("ticket unresolved: database outage still ongoing", &candidates);
        assert!(
            logits[0] > logits[1],
            "database_outage ({}) must win over billing ({}) when issue is unresolved",
            logits[0],
            logits[1]
        );
    }

    #[test]
    fn test_r2_p04_description_polarity_keys() {
        let ctx = PremiseContext::new("please escalate this ticket");

        // P04: "Normal priority" should NOT be penalized as negative
        let cand_normal = Candidate { id: "normal".into(), description: "Normal priority".into(), value: None };
        let cand_standard = Candidate { id: "standard".into(), description: "Standard priority".into(), value: None };
        let score_normal = ctx.score_candidate(&cand_normal);
        let score_standard = ctx.score_candidate(&cand_standard);
        assert_eq!(
            score_normal, score_standard,
            "Normal priority ({score_normal}) must equal Standard priority ({score_standard})"
        );
        assert_eq!(score_normal, 0.5);

        // P04: "Yesterday's orders" should NOT be boosted as affirmative
        let cand_yesterday = Candidate { id: "yesterday".into(), description: "Yesterday's orders".into(), value: None };
        let cand_recent = Candidate { id: "recent".into(), description: "Recent orders".into(), value: None };
        let score_yesterday = ctx.score_candidate(&cand_yesterday);
        let score_recent = ctx.score_candidate(&cand_recent);
        assert_eq!(
            score_yesterday, score_recent,
            "Yesterday's orders ({score_yesterday}) must equal Recent orders ({score_recent})"
        );
        assert_eq!(score_yesterday, 0.5);

        // Actual boolean candidates still get polarity scoring
        let cand_true = Candidate { id: "true".into(), description: "Yes".into(), value: None };
        let cand_false = Candidate { id: "false".into(), description: "No".into(), value: None };
        let score_true = ctx.score_candidate(&cand_true);
        let score_false = ctx.score_candidate(&cand_false);
        assert!(
            score_true > score_false,
            "Affirmative boolean ({score_true}) must score higher than negative boolean ({score_false}) on neutral premise"
        );
    }

    #[test]
    fn test_r11_p09_empty_pattern_no_hang() {
        let ctx = PremiseContext::new("Some premise text for testing empty pattern handling");

        // contains_bounded("") must return false immediately
        assert!(!ctx.contains_bounded(""));

        // is_negated("") must return false immediately
        assert!(!ctx.is_negated(""));

        let empty_ctx = PremiseContext::new("");
        assert!(!empty_ctx.contains_bounded(""));
        assert!(!empty_ctx.contains_bounded("test"));
        assert!(!empty_ctx.is_negated(""));
        assert!(!empty_ctx.is_negated("test"));
    }
}


