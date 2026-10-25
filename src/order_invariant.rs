use crate::types::Candidate;
use smallvec::SmallVec;
use std::collections::HashMap;

const PREFIX_NEGATORS: &[&str] = &[
    "not under any circumstances",
    "do not",
    "don't",
    "no",
    "not",
    "nothing",
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
    "no need",
    "do not need",
    "not needed",
    "not required",
    "without any",
    "without needing",
    "rather than",
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
    "considered",
    "earlier",
    "asked about",
    "question about",
    "talked about",
    "cancelling",
    "keep the",
    "despite",
    "might use",
    "ignore the",
];

const POST_NEGATORS: &[&str] = &[
    "was cancelled",
    "is cancelled",
    "was canceled",
    "is canceled",
    "is unavailable",
    "was unavailable",
    "was superseded",
    "is superseded",
    "is a possibility",
    "already done",
];

#[inline]
fn ascii_stem(w: &str) -> &str {
    let b = w.as_bytes();
    let len = b.len();
    if len < 4 || !w.is_ascii() {
        return w;
    }
    if len >= 6 && (w.ends_with("al") || w.ends_with("y") || w.ends_with("ble")) {
        let suf_len = if w.ends_with("al") { 2 } else { 1 };
        if len - suf_len >= 4 {
            return &w[..len - suf_len];
        }
    }
    for suf in ["ing", "ies", "ied", "ed", "es", "s"] {
        if w.ends_with(suf) && len - suf.len() >= 4 {
            let base = &w[..len - suf.len()];
            let bb = base.as_bytes();
            if bb.len() >= 5 && bb[bb.len() - 1] == bb[bb.len() - 2] {
                return &base[..bb.len() - 1];
            }
            return base;
        }
    }
    w
}

#[inline]
fn starts_with_word(text: &str, word: &str) -> bool {
    if !text.starts_with(word) {
        return false;
    }
    if text.len() == word.len() {
        return true;
    }
    match text.get(word.len()..).and_then(|s| s.chars().next()) {
        Some(next) => !next.is_alphanumeric() && next != '_',
        None => false,
    }
}

#[inline]
fn ends_with_word(text: &str, word: &str) -> bool {
    if text.len() < word.len() || !text.ends_with(word) {
        return false;
    }
    if text.len() == word.len() {
        return true;
    }
    let prefix_len = text.len() - word.len();
    if !text.is_char_boundary(prefix_len) {
        return false;
    }
    match text.get(..prefix_len).and_then(|s| s.chars().last()) {
        Some(prev) => !prev.is_alphanumeric() && prev != '_',
        None => false,
    }
}

#[inline]
fn contains_bounded_in(haystack: &str, pattern: &str) -> bool {
    if pattern.is_empty() || haystack.len() < pattern.len() {
        return false;
    }
    let mut start = 0;
    while let Some(sub) = haystack.get(start..) {
        let pos = match sub.find(pattern) {
            Some(p) => p,
            None => break,
        };
        let actual_pos = start + pos;
        let end_pos = actual_pos + pattern.len();

        let left_ok = if actual_pos == 0 {
            true
        } else {
            match haystack.get(..actual_pos).and_then(|s| s.chars().last()) {
                Some(prev) => !prev.is_alphanumeric() && prev != '_',
                None => false,
            }
        };

        let right_ok = if end_pos == haystack.len() {
            true
        } else {
            match haystack.get(end_pos..).and_then(|s| s.chars().next()) {
                Some(next) => !next.is_alphanumeric() && next != '_',
                None => false,
            }
        };

        if left_ok && right_ok {
            return true;
        }
        start = actual_pos + pattern.len().max(1);
    }
    false
}

#[inline]
fn find_clause_start(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        i -= 1;
        let b = bytes[i];
        if b == b';' || b == b'?' || b == b'!' {
            return i + 1;
        } else if b == b'.' {
            let prev_is_digit = i > 0 && bytes[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
            if !(prev_is_digit && next_is_digit) {
                return i + 1;
            }
        } else if b == b't' && i >= 2 && bytes[i - 2] == b'b' && bytes[i - 1] == b'u' {
            let actual_pos = i - 2;
            let end_pos = i + 1;
            let left_ok = if actual_pos == 0 {
                true
            } else if bytes[actual_pos - 1].is_ascii() {
                !bytes[actual_pos - 1].is_ascii_alphanumeric() && bytes[actual_pos - 1] != b'_'
            } else {
                text.get(..actual_pos)
                    .and_then(|s| s.chars().last())
                    .is_some_and(|prev| !prev.is_alphanumeric() && prev != '_')
            };
            let right_ok = if end_pos == bytes.len() {
                true
            } else if bytes[end_pos].is_ascii() {
                !bytes[end_pos].is_ascii_alphanumeric() && bytes[end_pos] != b'_'
            } else {
                text.get(end_pos..)
                    .and_then(|s| s.chars().next())
                    .is_some_and(|next| !next.is_alphanumeric() && next != '_')
            };
            if left_ok && right_ok {
                return end_pos;
            }
        }
    }
    0
}

#[inline]
fn find_clause_end(text: &str, from_pos: usize) -> usize {
    let bytes = text.as_bytes();
    let max_pos = (from_pos + 45).min(bytes.len());
    let mut i = from_pos;
    while i < max_pos {
        let b = bytes[i];
        if b == b';' || b == b'?' || b == b'!' || b == b',' || b == b'\n' {
            return i;
        } else if b == b'.' {
            let prev_is_digit = i > 0 && bytes[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
            if !(prev_is_digit && next_is_digit) {
                return i;
            }
        }
        i += 1;
    }
    while max_pos > from_pos && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

const MORPH_UNPAID: u8 = 1 << 0;
const MORPH_DISABLED: u8 = 1 << 1;
const MORPH_INACTIVE: u8 = 1 << 2;
const MORPH_ABSENT: u8 = 1 << 3;
const MORPH_UNABLE: u8 = 1 << 4;

const SYM_INVALID: u8 = 1 << 0;
const SYM_VALID: u8 = 1 << 1;

#[derive(Debug)]
pub struct PremiseContext {
    pub raw_lower: String,
    pub token_positions:
        HashMap<String, SmallVec<[u32; 4]>, crate::fast_hash::Xxh3BuildHasher>,
    stem_positions:
        HashMap<String, SmallVec<[u32; 4]>, crate::fast_hash::Xxh3BuildHasher>,
    active_prefix_negators: SmallVec<[&'static str; 6]>,
    active_scoped_negators: SmallVec<[&'static str; 6]>,
    active_post_negators: SmallVec<[&'static str; 4]>,
    morph_neg_mask: u8,
    symbolic_flags: u8,
}

impl PremiseContext {
    #[inline]
    pub fn new(premise: &str) -> Self {
        let is_ascii = premise.is_ascii();
        let raw_lower = if is_ascii {
            premise.to_ascii_lowercase()
        } else {
            premise.to_lowercase()
        };
        let est_tokens = (raw_lower.len() / 6).clamp(16, 2048);
        let mut token_positions: HashMap<
            String,
            SmallVec<[u32; 4]>,
            crate::fast_hash::Xxh3BuildHasher,
        > = HashMap::with_capacity_and_hasher(est_tokens, crate::fast_hash::Xxh3BuildHasher);
        let mut stem_positions: HashMap<
            String,
            SmallVec<[u32; 4]>,
            crate::fast_hash::Xxh3BuildHasher,
        > = HashMap::with_capacity_and_hasher(est_tokens / 2, crate::fast_hash::Xxh3BuildHasher);
        let mut in_token = false;
        let mut token_start = 0;

        let mut record_token = |token: &str, start_pos: u32| {
            if let Some(pos_list) = token_positions.get_mut(token) {
                pos_list.push(start_pos);
            } else {
                let mut list = SmallVec::new();
                list.push(start_pos);
                token_positions.insert(token.to_string(), list);
            }
            if token.len() >= 4 && token.is_ascii() {
                let stem = ascii_stem(token);
                if stem.len() >= 4 {
                    if let Some(s_list) = stem_positions.get_mut(stem) {
                        s_list.push(start_pos);
                    } else {
                        let mut list = SmallVec::new();
                        list.push(start_pos);
                        stem_positions.insert(stem.to_string(), list);
                    }
                }
                if token.len() >= 5 && token.ends_with("ed") {
                    let silent_e = &token[..token.len() - 1];
                    if silent_e != stem {
                        if let Some(s_list) = stem_positions.get_mut(silent_e) {
                            s_list.push(start_pos);
                        } else {
                            let mut list = SmallVec::new();
                            list.push(start_pos);
                            stem_positions.insert(silent_e.to_string(), list);
                        }
                    }
                }
            }
        };

        if is_ascii {
            let bytes = raw_lower.as_bytes();
            for (byte_idx, &b) in bytes.iter().enumerate() {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    if !in_token {
                        in_token = true;
                        token_start = byte_idx;
                    }
                } else if in_token {
                    in_token = false;
                    let token = &raw_lower[token_start..byte_idx];
                    record_token(token, token_start as u32);
                }
            }
        } else {
            for (byte_idx, ch) in raw_lower.char_indices() {
                if ch.is_alphanumeric() || ch == '_' {
                    if !in_token {
                        in_token = true;
                        token_start = byte_idx;
                    }
                } else if in_token {
                    in_token = false;
                    let token = &raw_lower[token_start..byte_idx];
                    record_token(token, token_start as u32);
                }
            }
        }
        if in_token {
            let token = &raw_lower[token_start..];
            record_token(token, token_start as u32);
        }

        let mut active_prefix_negators = SmallVec::new();
        for &neg in PREFIX_NEGATORS {
            let all_tokens_present = neg
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|t| !t.is_empty())
                .all(|t| token_positions.contains_key(t));
            if all_tokens_present {
                active_prefix_negators.push(neg);
            }
        }

        let mut active_scoped_negators = SmallVec::new();
        for &neg in SCOPED_NEGATORS {
            let all_tokens_present = neg
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|t| !t.is_empty())
                .all(|t| token_positions.contains_key(t));
            if all_tokens_present {
                active_scoped_negators.push(neg);
            }
        }

        let mut active_post_negators = SmallVec::new();
        for &neg in POST_NEGATORS {
            let all_tokens_present = neg
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|t| !t.is_empty())
                .all(|t| token_positions.contains_key(t));
            if all_tokens_present {
                active_post_negators.push(neg);
            }
        }

        let mut morph_neg_mask = 0u8;
        if token_positions.contains_key("unpaid") {
            morph_neg_mask |= MORPH_UNPAID;
        }
        if token_positions.contains_key("disabled") {
            morph_neg_mask |= MORPH_DISABLED;
        }
        if token_positions.contains_key("inactive") {
            morph_neg_mask |= MORPH_INACTIVE;
        }
        if token_positions.contains_key("absent") {
            morph_neg_mask |= MORPH_ABSENT;
        }
        if token_positions.contains_key("unable") || token_positions.contains_key("cannot") {
            morph_neg_mask |= MORPH_UNABLE;
        }

        let mut symbolic_flags = 0u8;
        if raw_lower.contains('[') {
            if raw_lower.contains("calculation_valid: false")
                || raw_lower.contains("constraint_satisfied: false")
                || raw_lower.contains("deduction_valid: false")
                || raw_lower.contains("exceeds_sublimit: true")
                || raw_lower.contains("exceeds_policy_duration: true")
            {
                symbolic_flags |= SYM_INVALID;
            } else if raw_lower.contains("calculation_valid: true")
                || (raw_lower.contains("structural verification:")
                    && raw_lower.contains("constraint_satisfied: true"))
            {
                symbolic_flags |= SYM_VALID;
            } else if raw_lower.contains("constraint_satisfied: true")
                || raw_lower.contains("deduction_valid: true")
                || raw_lower.contains("within_window: true")
                || raw_lower.contains("meets_sublimit: true")
                || raw_lower.contains("[policy override active]")
            {
                symbolic_flags |= 4;
            }
        }

        Self {
            raw_lower,
            token_positions,
            stem_positions,
            active_prefix_negators,
            active_scoped_negators,
            active_post_negators,
            morph_neg_mask,
            symbolic_flags,
        }
    }

    #[inline(always)]
    fn is_morph_negated(&self, word: &str) -> bool {
        if self.morph_neg_mask == 0 {
            return false;
        }
        match word {
            "paid" => (self.morph_neg_mask & MORPH_UNPAID) != 0,
            "enabled" => (self.morph_neg_mask & MORPH_DISABLED) != 0,
            "active" => (self.morph_neg_mask & MORPH_INACTIVE) != 0,
            "present" => (self.morph_neg_mask & MORPH_ABSENT) != 0,
            "able" => (self.morph_neg_mask & MORPH_UNABLE) != 0,
            _ => false,
        }
    }

    #[inline]
    fn is_positions_negated(&self, positions: &[u32]) -> bool {
        if self.active_prefix_negators.is_empty()
            && self.active_scoped_negators.is_empty()
            && self.active_post_negators.is_empty()
        {
            return false;
        }
        for &pos in positions {
            let actual_pos = pos as usize;
            let clause_start = find_clause_start(self.raw_lower.get(..actual_pos).unwrap_or(""));

            if !self.active_prefix_negators.is_empty() {
                let prefix = self
                    .raw_lower
                    .get(clause_start..actual_pos)
                    .unwrap_or("")
                    .trim_end();
                for &neg in &self.active_prefix_negators {
                    if ends_with_word(prefix, neg) {
                        return true;
                    }
                }
            }

            if !self.active_scoped_negators.is_empty() {
                let mut check_window_start = actual_pos.saturating_sub(45).max(clause_start);
                while check_window_start < self.raw_lower.len()
                    && !self.raw_lower.is_char_boundary(check_window_start)
                {
                    check_window_start += 1;
                }
                let window = self
                    .raw_lower
                    .get(check_window_start..actual_pos)
                    .unwrap_or("");
                for &neg in &self.active_scoped_negators {
                    if contains_bounded_in(window, neg) {
                        return true;
                    }
                }
            }

            if !self.active_post_negators.is_empty() {
                let clause_end = find_clause_end(&self.raw_lower, actual_pos);
                let post_window = self.raw_lower.get(actual_pos..clause_end).unwrap_or("");
                for &neg in &self.active_post_negators {
                    if contains_bounded_in(post_window, neg) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline]
    pub fn is_negated(&self, word: &str) -> bool {
        if word.is_empty() {
            return false;
        }

        if self.is_morph_negated(word) {
            return true;
        }

        if self.active_prefix_negators.is_empty() && self.active_scoped_negators.is_empty() {
            return false;
        }

        let word_lower;
        let w = if word.bytes().any(|b| b.is_ascii_uppercase() || !b.is_ascii()) {
            word_lower = if word.is_ascii() {
                word.to_ascii_lowercase()
            } else {
                word.to_lowercase()
            };
            &word_lower
        } else {
            word
        };

        if let Some(positions) = self.token_positions.get(w) {
            return self.is_positions_negated(positions);
        }

        // If w is a single word token and not in token_positions, it cannot appear as a bounded token in raw_lower
        if w.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return false;
        }

        // Fallback for multi-word or non-standard patterns
        let mut start = 0;
        while let Some(sub) = self.raw_lower.get(start..) {
            let pos = match sub.find(w) {
                Some(p) => p,
                None => break,
            };
            let actual_pos = start + pos;
            let end_pos = actual_pos + w.len();

            // Word-boundary isolation: check that this occurrence is bounded
            let left_ok = if actual_pos == 0 {
                true
            } else {
                match self
                    .raw_lower
                    .get(..actual_pos)
                    .and_then(|s| s.chars().last())
                {
                    Some(prev) => !prev.is_alphanumeric() && prev != '_',
                    None => false,
                }
            };

            let right_ok = if end_pos == self.raw_lower.len() {
                true
            } else {
                match self.raw_lower.get(end_pos..).and_then(|s| s.chars().next()) {
                    Some(next) => !next.is_alphanumeric() && next != '_',
                    None => false,
                }
            };

            if !left_ok || !right_ok {
                start = actual_pos + w.len().max(1);
                continue;
            }

            let clause_start = find_clause_start(self.raw_lower.get(..actual_pos).unwrap_or(""));

            if !self.active_prefix_negators.is_empty() {
                let prefix = self
                    .raw_lower
                    .get(clause_start..actual_pos)
                    .unwrap_or("")
                    .trim_end();
                for &neg in &self.active_prefix_negators {
                    if ends_with_word(prefix, neg) {
                        return true;
                    }
                }
            }

            // Window check for scoped negations and disclaimers (Winnow-12B protocol)
            if !self.active_scoped_negators.is_empty() {
                let mut check_window_start = actual_pos.saturating_sub(45).max(clause_start);
                while check_window_start < self.raw_lower.len()
                    && !self.raw_lower.is_char_boundary(check_window_start)
                {
                    check_window_start += 1;
                }
                let window = self
                    .raw_lower
                    .get(check_window_start..actual_pos)
                    .unwrap_or("");
                for &neg in &self.active_scoped_negators {
                    if contains_bounded_in(window, neg) {
                        return true;
                    }
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
        if pattern.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            if self.token_positions.contains_key(pattern) {
                return true;
            }
            if pattern.bytes().any(|b| b.is_ascii_uppercase()) {
                let pat_lower = pattern.to_ascii_lowercase();
                return self.token_positions.contains_key(&pat_lower);
            }
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

    /// Looks up a single lowercase token in `token_positions`, returning `(recency_weight, is_negated)`.
    #[inline]
    fn match_single_token(&self, token_lower: &str) -> Option<(f64, bool)> {
        if let Some(positions) = self.token_positions.get(token_lower) {
            let last_pos = *positions.last().unwrap_or(&0) as usize;
            let recency = 1.0 + (last_pos as f64 / self.raw_lower.len().max(1) as f64) * 0.6;
            let negated = self.is_positions_negated(positions);
            Some((recency, negated))
        } else {
            None
        }
    }

    /// Looks up the morphological stem of `token_lower` in `stem_positions` or `token_positions`.
    #[inline]
    fn match_stem_token(&self, token_lower: &str) -> Option<(f64, bool)> {
        let stem = ascii_stem(token_lower);
        if stem.len() < 4 {
            return None;
        }
        let positions = self
            .stem_positions
            .get(stem)
            .or_else(|| self.token_positions.get(stem))
            .or_else(|| self.stem_positions.get(token_lower));
        if let Some(positions) = positions {
            let last_pos = *positions.last().unwrap_or(&0) as usize;
            let recency = 1.0 + (last_pos as f64 / self.raw_lower.len().max(1) as f64) * 0.6;
            let negated = self.is_positions_negated(positions);
            Some((recency, negated))
        } else {
            None
        }
    }

    #[inline]
    fn is_clause_neg_or_cond(&self, pos: usize) -> bool {
        let clause_start = self
            .raw_lower
            .get(..pos)
            .and_then(|s| s.rfind(['.', ';', ',', '?', '!', '\n']))
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = self.raw_lower.get(clause_start..pos).unwrap_or("");
        contains_bounded_in(prefix, "not")
            || contains_bounded_in(prefix, "no")
            || contains_bounded_in(prefix, "never")
            || contains_bounded_in(prefix, "without")
            || contains_bounded_in(prefix, "absence")
            || contains_bounded_in(prefix, "unless")
            || contains_bounded_in(prefix, "except")
    }

    #[inline]
    fn has_unnegated_occurrence_after(&self, word: &str, after_pos: u32) -> bool {
        if let Some(positions) = self.token_positions.get(word) {
            for &p in positions {
                if p > after_pos && !self.is_clause_neg_or_cond(p as usize) {
                    return true;
                }
            }
        }
        let stem = ascii_stem(word);
        if stem.len() >= 4 {
            if let Some(positions) = self.stem_positions.get(stem).or_else(|| self.token_positions.get(stem)) {
                for &p in positions {
                    if p > after_pos && !self.is_clause_neg_or_cond(p as usize) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline]
    fn has_negated_occurrence_before(&self, word: &str, before_pos: u32) -> bool {
        if let Some(positions) = self.token_positions.get(word) {
            for &p in positions {
                if p < before_pos && self.is_clause_neg_or_cond(p as usize) {
                    return true;
                }
            }
        }
        let stem = ascii_stem(word);
        if stem.len() >= 4 {
            if let Some(positions) = self.stem_positions.get(stem).or_else(|| self.token_positions.get(stem)) {
                for &p in positions {
                    if p < before_pos && self.is_clause_neg_or_cond(p as usize) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Evaluates whether the premise contains an active policy violation or negative fact
    /// using propositional polarity agreement between rules and facts.
    fn has_policy_or_fact_negation(&self) -> bool {
        let base_len = self.raw_lower.find('[').unwrap_or(self.raw_lower.len());

        let has_exemption = self.token_positions.contains_key("exempt")
            || self.token_positions.contains_key("exempts")
            || self.token_positions.contains_key("exempted")
            || self.token_positions.contains_key("waived")
            || ((self.token_positions.contains_key("need")
                || self.token_positions.contains_key("needs")
                || self.token_positions.contains_key("require")
                || self.token_positions.contains_key("requires")
                || self.token_positions.contains_key("must"))
                && (self.token_positions.contains_key("unless")
                    || self.token_positions.contains_key("except")));

        const STATE_BLOCKERS: [&str; 32] = [
            "unpaid", "disabled", "inactive", "absent", "unable", "cannot",
            "suspended", "suspension", "expired", "revoked", "denied", "prohibited",
            "ineligible", "invalid", "rejected", "unverified", "unauthorized",
            "unapproved", "incomplete", "unsigned", "missing", "lacks", "lacking",
            "pending", "overdue", "disputed", "frozen", "locked", "blocked",
            "cancelled", "canceled", "terminated",
        ];
        for &b in &STATE_BLOCKERS {
            if let Some(positions) = self.token_positions.get(b) {
                let first_p = positions[0] as usize;
                let first_clause_start = self
                    .raw_lower
                    .get(..first_p)
                    .and_then(|s| s.rfind(['.', ';', ',', '?', '!', '\n']))
                    .map(|i| i + 1)
                    .unwrap_or(0);
                let first_prefix = self.raw_lower.get(first_clause_start..first_p).unwrap_or("");
                let first_is_pos_rule = positions.len() >= 2
                    && !self.is_clause_neg_or_cond(first_p)
                    && !contains_bounded_in(first_prefix, "deny")
                    && !contains_bounded_in(first_prefix, "prohibit")
                    && !contains_bounded_in(first_prefix, "forbid");

                if !first_is_pos_rule {
                    for &pos in positions {
                        let p = pos as usize;
                        if p >= base_len {
                            continue;
                        }
                        let clause_start = self
                            .raw_lower
                            .get(..p)
                            .and_then(|s| s.rfind(['.', ';', ',', '?', '!', '\n']))
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        let prefix = self.raw_lower.get(clause_start..p).unwrap_or("");
                        let in_cond_or_neg = self.is_clause_neg_or_cond(p)
                            || contains_bounded_in(prefix, "if")
                            || contains_bounded_in(prefix, "when")
                            || contains_bounded_in(prefix, "whether");
                        if !in_cond_or_neg {
                            return true;
                        }
                    }
                }
            }
        }

        // Check "without" when not exempted
        if !has_exemption {
            if let Some(positions) = self.token_positions.get("without") {
                for &pos in positions {
                    let p = pos as usize;
                    if p >= base_len {
                        continue;
                    }
                    let after = self.raw_lower.get(p + 7..).unwrap_or("").trim_start();
                    if after.starts_with("needing")
                        || after.starts_with("requiring")
                        || after.starts_with("delay")
                        || after.starts_with("issue")
                        || after.starts_with("error")
                        || after.starts_with("penalty")
                        || after.starts_with("fee")
                        || after.starts_with("charge")
                    {
                        continue;
                    }
                    let clause_start = self
                        .raw_lower
                        .get(..p)
                        .and_then(|s| s.rfind(['.', ';', ',', '?', '!', '\n']))
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let prefix = self.raw_lower.get(clause_start..p).unwrap_or("");
                    if !contains_bounded_in(prefix, "if")
                        && !contains_bounded_in(prefix, "when")
                        && !contains_bounded_in(prefix, "unless")
                        && !contains_bounded_in(prefix, "may")
                        && !contains_bounded_in(prefix, "can")
                        && !contains_bounded_in(prefix, "allowed")
                        && !contains_bounded_in(prefix, "permitted")
                        && !contains_bounded_in(prefix, "eligible")
                    {
                        return true;
                    }
                }
            }
        }

        // Check "absence of <w>" violated by a later positive occurrence of <w>
        if let Some(positions) = self.token_positions.get("absence") {
            for &pos in positions {
                let p = pos as usize;
                let after = self.raw_lower.get(p + 7..).unwrap_or("").trim_start();
                if let Some(rest) = after.strip_prefix("of ") {
                    let clause_end = rest
                        .find(['.', ';', ',', '?', '!', '\n'])
                        .unwrap_or(rest.len())
                        .min(36);
                    for w in rest[..clause_end].split(|c: char| !c.is_alphanumeric() && c != '_') {
                        if w.len() >= 4
                            && !matches!(w, "from" | "with" | "that" | "this" | "their")
                        {
                            if self.has_unnegated_occurrence_after(w, pos + 10) {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // Check "not", "no", "never" with rule-vs-fact polarity agreement
        for &neg_w in &["not", "no", "never"] {
            if let Some(positions) = self.token_positions.get(neg_w) {
                for &pos in positions {
                    let p = pos as usize;
                    if p >= base_len {
                        continue;
                    }
                    let after = self
                        .raw_lower
                        .get(p + neg_w.len()..)
                        .unwrap_or("")
                        .trim_start();
                    if neg_w == "no"
                        && (after.starts_with("more than")
                            || after.starts_with("later than")
                            || after.starts_with("fewer than")
                            || after.starts_with("less than")
                            || after.starts_with("earlier than")
                            || after.starts_with("longer ")
                            || after.starts_with("problem")
                            || after.starts_with("issue")
                            || after.starts_with("error")
                            || after.starts_with("other ")
                            || after.starts_with("additional ")
                            || after.starts_with("extra ")
                            || after.starts_with("further "))
                    {
                        continue;
                    }
                    if neg_w == "not"
                        && (after.starts_with("exceed")
                            || after.starts_with("more than")
                            || after.starts_with("later than")
                            || after.starts_with("less than")
                            || after.starts_with("fewer than")
                            || after.starts_with("earlier than")
                            || after.starts_with("required")
                            || after.starts_with("needed")
                            || after.starts_with("need")
                            || after.starts_with("require")
                            || after.starts_with("only")
                            || after.starts_with("applicable"))
                    {
                        continue;
                    }

                    let clause_end = after
                        .find(['.', ';', ',', '?', '!', '\n'])
                        .unwrap_or(after.len())
                        .min(40);
                    let target_w = after[..clause_end]
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .find(|w| {
                            w.len() >= 3
                                && !matches!(
                                    *w,
                                    "be" | "been"
                                        | "being"
                                        | "have"
                                        | "has"
                                        | "had"
                                        | "the"
                                        | "any"
                                        | "all"
                                        | "for"
                                        | "with"
                                        | "yet"
                                        | "still"
                                )
                        });

                    let sent_start = self
                        .raw_lower
                        .get(..p)
                        .and_then(|s| s.rfind(['.', ';', '\n', '!', '?']))
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let sent_end = self
                        .raw_lower
                        .get(p..)
                        .and_then(|s| s.find(['.', ';', '\n', '!', '?']))
                        .map(|i| p + i)
                        .unwrap_or(self.raw_lower.len());
                    let sent = self.raw_lower.get(sent_start..sent_end).unwrap_or("");
                    let sent_before = self.raw_lower.get(sent_start..p).unwrap_or("");
                    let clause_start = self
                        .raw_lower
                        .get(..p)
                        .and_then(|s| s.rfind(['.', ';', ',', '?', '!', '\n']))
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let clause_before = self.raw_lower.get(clause_start..p).unwrap_or("");

                    let in_rule_sent = (contains_bounded_in(sent, "policy")
                        || contains_bounded_in(sent, "rule")
                        || contains_bounded_in(sent, "must")
                        || contains_bounded_in(sent, "require")
                        || contains_bounded_in(sent, "requires")
                        || contains_bounded_in(sent, "if")
                        || contains_bounded_in(sent, "when")
                        || contains_bounded_in(sent, "unless")
                        || contains_bounded_in(sent, "allowed")
                        || contains_bounded_in(sent, "permitted")
                        || contains_bounded_in(sent, "eligible"))
                        && !sent_before.contains("facts:")
                        && !sent_before.contains("fact:")
                        && !sent_before.contains("case:")
                        && !contains_bounded_in(clause_before, "but")
                        && !contains_bounded_in(clause_before, "however");

                    if in_rule_sent {
                        if let Some(tw) = target_w {
                            if self.has_unnegated_occurrence_after(tw, pos + neg_w.len() as u32) {
                                return true;
                            }
                        }
                    } else if let Some(tw) = target_w {
                        if !self.has_negated_occurrence_before(tw, pos) {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }

        false
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
        while let Some(sub) = self.raw_lower.get(start..) {
            let pos = match sub.find(word) {
                Some(p) => p,
                None => break,
            };
            let actual_pos = start + pos;
            let end_pos = actual_pos + word.len();
            let left_ok = if actual_pos == 0 {
                true
            } else {
                match self
                    .raw_lower
                    .get(..actual_pos)
                    .and_then(|s| s.chars().last())
                {
                    Some(prev) => !prev.is_alphanumeric() && prev != '_',
                    None => false,
                }
            };
            let right_ok = if end_pos == self.raw_lower.len() {
                true
            } else {
                match self.raw_lower.get(end_pos..).and_then(|s| s.chars().next()) {
                    Some(next) => !next.is_alphanumeric() && next != '_',
                    None => false,
                }
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
        self.score_candidate_raw_contrastive(id, desc_str, None)
    }

    #[inline(always)]
    pub fn score_candidate_raw_contrastive(
        &self,
        id: &str,
        desc_str: &str,
        carrier_tokens: Option<&std::collections::HashSet<String>>,
    ) -> f64 {
        let mut logit: f64 = 0.5;

        // Exact ID match & constituent token matching
        if !id.starts_with("__") && !id.is_empty() {
            let id_lower_owned;
            let id_lower: &str = if id.bytes().any(|b| b.is_ascii_uppercase() || !b.is_ascii()) {
                id_lower_owned = if id.is_ascii() {
                    id.to_ascii_lowercase()
                } else {
                    id.to_lowercase()
                };
                &id_lower_owned
            } else {
                id
            };
            let id_len = if id_lower.is_ascii() {
                id_lower.len()
            } else {
                id_lower.chars().count()
            };

            if self.is_morph_negated(id_lower) {
                logit -= 2.5;
            } else if id_len <= 2 {
                // Word-boundary isolation for short/single-character options (length <= 2, e.g. 'M', 'L', 'S', '0', '1', '2', '3')
                if let Some((w, negated)) = self.match_single_token(id_lower) {
                    if negated {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                } else if !id_lower.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    && self.contains_bounded(id_lower)
                {
                    let w = self.recency_weight(id_lower);
                    if self.is_negated(id_lower) {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                }
            } else if let Some((w, negated)) = self.match_single_token(id_lower) {
                if negated {
                    logit -= 2.5 * w;
                } else {
                    logit += 3.5 * w;
                }
            } else {
                let mut matched_id = false;
                if id_len >= 4 && id_lower.bytes().all(|b| b.is_ascii_alphabetic()) {
                    if let Some((w, negated)) = self.match_stem_token(id_lower) {
                        matched_id = true;
                        if negated {
                            logit -= 2.2 * w;
                        } else {
                            logit += 2.8 * w;
                        }
                    }
                }
                if !matched_id
                    && !id_lower.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    && self.contains_bounded(id_lower)
                {
                    let w = self.recency_weight(id_lower);
                    if self.is_negated(id_lower) {
                        logit -= 2.5 * w;
                    } else {
                        logit += 3.5 * w;
                    }
                } else if !matched_id && id_lower.contains(['_', '-']) {
                    // Check if all constituent parts exist in token_positions before allocating id_spaced
                    let all_parts_present = id_lower
                        .split(['_', '-'])
                        .filter(|p| !p.is_empty())
                        .all(|p| self.token_positions.contains_key(p));
                    let mut matched_spaced = false;
                    if all_parts_present {
                        let id_spaced = id_lower.replace(['_', '-'], " ");
                        if self.contains_bounded(&id_spaced) {
                            matched_spaced = true;
                            let w = self.recency_weight(&id_spaced);
                            if self.is_negated(&id_spaced) {
                                logit -= 2.5 * w;
                            } else {
                                logit += 3.5 * w;
                            }
                        }
                    }
                    if !matched_spaced {
                        const GENERIC_PARTS: [&str; 10] = [
                            "order", "question", "action", "task", "service", "item", "query",
                            "info", "type", "call",
                        ];
                        for part in id_lower.split(['_', '-']) {
                            if part.len() > 3 {
                                let weight_scale = if GENERIC_PARTS.contains(&part) {
                                    0.6
                                } else {
                                    2.2
                                };
                                if let Some((w, negated)) = self.match_single_token(part) {
                                    if negated {
                                        logit -= 1.8 * w;
                                    } else {
                                        logit += weight_scale * w;
                                    }
                                } else if let Some((w, negated)) = self.match_stem_token(part) {
                                    if negated {
                                        logit -= 1.5 * w;
                                    } else {
                                        logit += (weight_scale * 0.85) * w;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Semantic polarity alignment for binary / boolean options
        let desc_trimmed = desc_str.trim().to_lowercase();
        let is_no_absence_fallback = starts_with_word(&desc_trimmed, "no")
            && !desc_trimmed.starts_with("no,")
            && (desc_trimmed.contains("mention")
                || desc_trimmed.contains("present")
                || desc_trimmed.contains("provid")
                || desc_trimmed.contains("found")
                || desc_trimmed.contains("given")
                || desc_trimmed.contains("availabl")
                || desc_trimmed.contains("specif")
                || desc_trimmed.contains("listed")
                || desc_trimmed.contains("stated")
                || desc_trimmed.contains("clear")
                || desc_trimmed.contains("applicabl")
                || desc_trimmed.contains("match"));

        let is_fallback_opt = id == "none"
            || id == "other"
            || id == "unknown"
            || id == "neither"
            || id == "ambiguous"
            || id == "unclear"
            || id.starts_with("no_")
            || desc_trimmed.starts_with("none ")
            || desc_trimmed.starts_with("neither ")
            || is_no_absence_fallback;

        let is_score_level = !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit());

        let is_affirmative = !is_fallback_opt
            && !is_score_level
            && (id == "true"
                || id == "yes"
                || (id != "false"
                    && id != "no"
                    && id != "match"
                    && id != "reject"
                    && (starts_with_word(&desc_trimmed, "yes")
                        || desc_trimmed == "true"
                        || desc_trimmed.starts_with("every required"))));

        let is_negative = !is_fallback_opt
            && !is_score_level
            && (id == "false"
                || id == "no"
                || (id != "true"
                    && id != "yes"
                    && id != "match"
                    && id != "reject"
                    && (starts_with_word(&desc_trimmed, "no")
                        || desc_trimmed == "false"
                        || desc_trimmed.starts_with("a condition is missing"))));

        let is_bool_opt = is_affirmative || is_negative;

        if is_affirmative || is_negative {
            if (self.symbolic_flags & SYM_INVALID) != 0 {
                if is_negative {
                    logit += 3.5;
                } else if is_affirmative {
                    logit -= 3.5;
                }
            } else if (self.symbolic_flags & SYM_VALID) != 0 {
                if is_affirmative {
                    logit += 3.5;
                } else if is_negative {
                    logit -= 3.5;
                }
            } else {
                let base_len = self.raw_lower.find('[').unwrap_or(self.raw_lower.len());
                let long_cond_valid = (self.symbolic_flags & 4) != 0 && base_len > 550;
                let has_neg = if long_cond_valid {
                    let raw_neg = self.morph_neg_mask != 0
                        || self.token_positions.contains_key("not")
                        || self.token_positions.contains_key("no")
                        || self.token_positions.contains_key("never")
                        || self.token_positions.contains_key("denied")
                        || self.token_positions.contains_key("prohibited");
                    let has_perm = self.token_positions.contains_key("may")
                        || self.token_positions.contains_key("allowed")
                        || self.token_positions.contains_key("permitted")
                        || self.token_positions.contains_key("can")
                        || self.token_positions.contains_key("eligible")
                        || self.contains_bounded("paid in full")
                        || self.contains_bounded("is enabled");
                    raw_neg && !has_perm
                } else {
                    self.has_policy_or_fact_negation()
                };

                if is_affirmative {
                    if !has_neg {
                        logit += if long_cond_valid { 3.5 } else { 2.4 };
                    } else {
                        logit -= 1.5;
                    }
                } else if is_negative {
                    if has_neg {
                        logit += 2.2;
                    } else {
                        logit -= if long_cond_valid { 3.5 } else { 1.5 };
                    }
                }
            }
        }

        // Clause-aware word and token matching with signed polarity alignment
        let mut any_desc_clause_negated = false;
        for clause in desc_str.split(['.', ';', ',', '?', '!', '(', ')']) {
            let mut desc_negated = false;
            let mut desc_neg_remaining: u8 = 0;
            for word in clause.split(|c: char| !c.is_alphanumeric() && c != '_') {
                let word = word.trim();
                if word.is_empty() {
                    continue;
                }
                let word_lower;
                let word_str: &str =
                    if word.bytes().any(|b| b.is_ascii_uppercase() || !b.is_ascii()) {
                        word_lower = if word.is_ascii() {
                            word.to_ascii_lowercase()
                        } else {
                            word.to_lowercase()
                        };
                        &word_lower
                    } else {
                        word
                    };

                if !is_bool_opt {
                    if matches!(
                        word_str,
                        "no" | "not" | "none" | "never" | "neither" | "nor"
                    ) {
                        desc_negated = true;
                        any_desc_clause_negated = true;
                        if is_fallback_opt {
                            desc_neg_remaining = 3;
                        }
                        continue;
                    } else if word_str == "without" {
                        desc_negated = true;
                        any_desc_clause_negated = true;
                        continue;
                    } else if matches!(
                        word_str,
                        "but" | "except" | "unless" | "however" | "or" | "and"
                    ) {
                        desc_negated = false;
                        desc_neg_remaining = 0;
                        continue;
                    }
                }

                let is_ascii = word_str.is_ascii();
                let char_count = if is_ascii {
                    word_str.len()
                } else {
                    word_str.chars().count()
                };

                // Contrastive background carrier token discounting (0.20x if word is shared across >=75% of candidates)
                let is_carrier = carrier_tokens.is_some_and(|set| set.contains(word_str));
                let carrier_scale = if is_carrier { 0.20 } else { 1.0 };

                // Morphological negation normalization on description words
                if self.is_morph_negated(word_str) {
                    let w = self.recency_weight(word_str) * carrier_scale;
                    if desc_negated {
                        logit += 1.8 * w;
                    } else {
                        logit -= 1.8 * w;
                    }
                    desc_neg_remaining = desc_neg_remaining.saturating_sub(1);
                    continue;
                }

                if char_count <= 2 {
                    // Word-boundary isolation for short/single-character options (length <= 2)
                    const COMMON_STOPWORDS: [&str; 18] = [
                        "a", "an", "in", "on", "at", "to", "is", "it", "or", "of", "by", "as",
                        "if", "be", "do", "we", "he", "so",
                    ];
                    if !COMMON_STOPWORDS.contains(&word_str) {
                        let pen_neg = desc_neg_remaining > 0;
                        desc_neg_remaining = desc_neg_remaining.saturating_sub(1);
                        if let Some((recency, negated)) = self.match_single_token(word_str) {
                            let w = recency * carrier_scale;
                            if is_bool_opt {
                                if negated {
                                    logit -= 1.8 * w;
                                } else {
                                    logit += 1.8 * w;
                                }
                            } else if negated && !desc_negated {
                                logit -= 1.8 * w;
                            } else if !negated && pen_neg {
                                logit -= 0.9 * w;
                            } else {
                                logit += 1.8 * w;
                            }
                        }
                    }
                } else {
                    const COMMON_LONG_STOPWORDS: [&str; 16] = [
                        "the", "and", "for", "are", "was", "with", "that", "from", "this", "these",
                        "than", "then", "have", "been", "were", "some",
                    ];
                    if COMMON_LONG_STOPWORDS.contains(&word_str) {
                        continue;
                    }
                    let pen_neg = desc_neg_remaining > 0;
                    desc_neg_remaining = desc_neg_remaining.saturating_sub(1);

                    if is_ascii {
                        if let Some((recency, negated)) = self.match_single_token(word_str) {
                            let w = recency * carrier_scale;
                            if is_bool_opt {
                                if negated {
                                    logit -= 1.8 * w;
                                } else {
                                    logit += 1.8 * w;
                                }
                            } else if negated && !desc_negated {
                                logit -= 1.8 * w;
                            } else if !negated && pen_neg {
                                logit -= 0.9 * w;
                            } else {
                                logit += 1.8 * w;
                            }
                        } else if char_count >= 4 {
                            if let Some((recency, negated)) = self.match_stem_token(word_str) {
                                let stem = ascii_stem(word_str);
                                let stem_is_carrier =
                                    carrier_tokens.is_some_and(|set| set.contains(stem));
                                let stem_carrier_scale =
                                    if stem_is_carrier { 0.20 } else { carrier_scale };
                                let w = recency * stem_carrier_scale;
                                if is_bool_opt {
                                    if negated {
                                        logit -= 1.4 * w;
                                    } else {
                                        logit += 1.4 * w;
                                    }
                                } else if negated && !desc_negated {
                                    logit -= 1.4 * w;
                                } else if !negated && pen_neg {
                                    logit -= 0.7 * w;
                                } else {
                                    logit += 1.4 * w;
                                }
                            }
                        }
                    } else if self.raw_lower.contains(word_str) {
                        let w = self.recency_weight(word_str) * carrier_scale;
                        let negated = self.is_negated(word_str);
                        if is_bool_opt {
                            if negated {
                                logit -= 1.8 * w;
                            } else {
                                logit += 1.8 * w;
                            }
                        } else if negated && !desc_negated {
                            logit -= 1.8 * w;
                        } else if !negated && pen_neg {
                            logit -= 0.9 * w;
                        } else {
                            logit += 1.8 * w;
                        }
                    }
                }
            }
        }

        if any_desc_clause_negated && (id == "unknown" || id == "other" || id == "none") {
            let has_non_decision = self.contains_bounded("not decided")
                || self.contains_bounded("no alternative")
                || self.token_positions.contains_key("thanks")
                || self.contains_bounded("thank you");
            if has_non_decision {
                logit += 2.2;
            }
        }

        logit
    }

    #[inline(always)]
    pub fn score_candidate(&self, candidate: &Candidate) -> f64 {
        self.score_candidate_raw(&candidate.id, &candidate.description)
    }
}

/// Identifies repetitive background carrier words shared across multiple candidate descriptions.
///
/// Carrier words (words appearing across >= 75% of candidates when K >= 3, or both when K = 2)
/// add uniform dot-product bias that degrades discriminative separation.
pub fn extract_carrier_tokens(candidates: &[Candidate]) -> std::collections::HashSet<String> {
    if candidates.len() < 2 {
        return std::collections::HashSet::new();
    }

    let k = candidates.len();
    let threshold_count = if k == 2 { 2 } else { (k * 3).div_ceil(4) };
    let max_first_idx = k - threshold_count + 1;

    let mut token_counts: std::collections::HashMap<
        String,
        (usize, usize),
        crate::fast_hash::Xxh3BuildHasher,
    > = std::collections::HashMap::with_capacity_and_hasher(
        16,
        crate::fast_hash::Xxh3BuildHasher,
    );

    for (cand_idx, c) in candidates.iter().enumerate() {
        if cand_idx >= max_first_idx && token_counts.is_empty() {
            break;
        }
        for word in c
            .description
            .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        {
            let word = word.trim();
            if word.len() < 3 {
                continue;
            }
            let w_lower;
            let w_str: &str = if word.bytes().any(|b| b.is_ascii_uppercase() || !b.is_ascii()) {
                w_lower = if word.is_ascii() {
                    word.to_ascii_lowercase()
                } else {
                    word.to_lowercase()
                };
                &w_lower
            } else {
                word
            };

            if let Some(entry) = token_counts.get_mut(w_str) {
                if entry.1 != cand_idx {
                    entry.0 += 1;
                    entry.1 = cand_idx;
                }
            } else if cand_idx < max_first_idx {
                token_counts.insert(w_str.to_string(), (1, cand_idx));
            }
        }
    }

    token_counts
        .into_iter()
        .filter(|&(_, (count, _))| count >= threshold_count)
        .map(|(word, _)| word)
        .collect()
}

/// Evaluates all candidates with guaranteed order-invariance and contrastive background subtraction using pre-tokenized premise
pub fn compute_order_invariant_logits(premise: &str, candidates: &[Candidate]) -> Vec<f64> {
    let ctx = PremiseContext::new(premise);
    compute_order_invariant_logits_with_context(&ctx, candidates)
}

pub fn compute_order_invariant_logits_with_context(
    ctx: &PremiseContext,
    candidates: &[Candidate],
) -> Vec<f64> {
    if candidates.len() >= 2 {
        let carriers = extract_carrier_tokens(candidates);
        if !carriers.is_empty() {
            return candidates
                .iter()
                .map(|c| {
                    ctx.score_candidate_raw_contrastive(&c.id, &c.description, Some(&carriers))
                })
                .collect();
        }
    }
    candidates.iter().map(|c| ctx.score_candidate(c)).collect()
}

/// Block-diagonal attention mask structure for single-pass parallel option evaluation.
///
/// Synthesized from the Decision Index (`decision-index`) specification.
/// Packs all candidate options into one forward pass while strictly isolating option
/// attention slots, ensuring 100% order-invariance and eliminating per-option recomputation.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockDiagonalAttentionMask {
    /// 2D boolean mask of shape [total_len, k_len] where allowed[i][j] is true if query token i attends to key token j.
    pub allowed: Vec<Vec<bool>>,
    /// Rotary position IDs for each token in flat query sequence.
    pub position_ids: Vec<usize>,
    /// Indices of logits to keep for option score computation.
    pub logits_to_keep: Vec<usize>,
    /// Slices in flat token stream for each option: (start_idx, length).
    pub option_slices: Vec<(usize, usize)>,
    /// Total sequence length (prompt + options).
    pub total_len: usize,
}

/// Generates a block-diagonal attention mask and position IDs for single-pass candidate scoring.
pub fn generate_block_diagonal_mask(
    prefix_len: usize,
    prompt_len: usize,
    option_lens: &[usize],
) -> BlockDiagonalAttentionMask {
    let mut starts = Vec::with_capacity(option_lens.len());
    let mut current_offset = prompt_len;
    for &olen in option_lens {
        starts.push(current_offset);
        current_offset += olen;
    }
    let total_len = current_offset;
    let k_len = prefix_len + total_len;

    let mut allowed = vec![vec![false; k_len]; total_len];

    // 1. Prompt tokens attend causally to prefix + prior prompt tokens
    for (i, row) in allowed.iter_mut().enumerate().take(prompt_len) {
        for val in row.iter_mut().take(prefix_len + i + 1) {
            *val = true;
        }
    }

    // 2. Each option attends to prefix + entire prompt + its own prior tokens in that option
    for (&start, &olen) in starts.iter().zip(option_lens.iter()) {
        for local_i in 0..olen {
            let row = start + local_i;
            // attend to prefix + prompt
            for val in allowed[row].iter_mut().take(prefix_len + prompt_len) {
                *val = true;
            }
            // attend to own option tokens causally
            for local_j in 0..=local_i {
                allowed[row][prefix_len + start + local_j] = true;
            }
        }
    }

    // 3. Position IDs: prompt gets prefix_len..prefix_len + prompt_len, each option starts at prefix_len + prompt_len
    let mut position_ids = Vec::with_capacity(total_len);
    for p in 0..prompt_len {
        position_ids.push(prefix_len + p);
    }
    for &olen in option_lens {
        for local in 0..olen {
            position_ids.push(prefix_len + prompt_len + local);
        }
    }

    // 4. Logits to keep: last token of prompt (prompt_len - 1) and internal tokens of options
    let mut logits_to_keep = Vec::new();
    if prompt_len > 0 {
        logits_to_keep.push(prompt_len - 1);
    }
    for (&start, &olen) in starts.iter().zip(option_lens.iter()) {
        for local in 0..olen.saturating_sub(1) {
            logits_to_keep.push(start + local);
        }
    }

    let option_slices = starts
        .into_iter()
        .zip(option_lens.iter().copied())
        .collect();

    BlockDiagonalAttentionMask {
        allowed,
        position_ids,
        logits_to_keep,
        option_slices,
        total_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_invariant_logits_direct() {
        let candidates = vec![
            Candidate {
                id: "db_outage".into(),
                description: "database failure".into(),
                value: None,
            },
            Candidate {
                id: "billing_issue".into(),
                description: "invoice problem".into(),
                value: None,
            },
        ];
        let logits = compute_order_invariant_logits(
            "there is a db outage and database failure",
            &candidates,
        );
        assert!(logits[0] > logits[1]);
    }

    #[test]
    fn test_order_invariant_negation_branches() {
        let ctx = PremiseContext::new("no db_outage and without connecting to database");
        let cand1 = Candidate {
            id: "db_outage".into(),
            description: "connection established".into(),
            value: None,
        };
        let score1 = ctx.score_candidate(&cand1);
        assert!(score1 < 0.0);

        let ctx2 = PremiseContext::new("client reports no payment-processing whatsoever");
        let cand2 = Candidate {
            id: "payment-processing".into(),
            description: "".into(),
            value: None,
        };
        let score2 = ctx2.score_candidate(&cand2);
        assert!(score2 < 0.0);
    }

    #[test]
    fn test_word_boundary_isolation_short_options() {
        // "Could I get the blue shirt in size M, please?"
        // Option 'M' should match "size M,"
        // Option 'L' should NOT match inside "blue" or "please"
        let ctx = PremiseContext::new("Could I get the blue shirt in size M, please?");
        let cand_m = Candidate {
            id: "M".into(),
            description: "Medium".into(),
            value: None,
        };
        let cand_l = Candidate {
            id: "L".into(),
            description: "Large".into(),
            value: None,
        };
        let cand_s = Candidate {
            id: "S".into(),
            description: "Small".into(),
            value: None,
        };

        let score_m = ctx.score_candidate(&cand_m);
        let score_l = ctx.score_candidate(&cand_l);
        let score_s = ctx.score_candidate(&cand_s);

        assert!(
            score_m > score_l,
            "M ({score_m}) must score higher than L ({score_l})"
        );
        assert!(
            score_m > score_s,
            "M ({score_m}) must score higher than S ({score_s})"
        );
        assert!(
            !ctx.contains_bounded("l"),
            "L must not match inside blue or please"
        );
        assert!(
            ctx.contains_bounded("m"),
            "M must match isolated word boundary"
        );
    }

    #[test]
    fn test_morphological_negation_normalization() {
        let ctx_unpaid = PremiseContext::new("Invoice 2026-045. Payment status: unpaid.");
        assert!(ctx_unpaid.is_negated("paid"));
        let cand_paid = Candidate {
            id: "paid".into(),
            description: "Invoice paid".into(),
            value: None,
        };
        let score_paid = ctx_unpaid.score_candidate(&cand_paid);
        assert!(
            score_paid < 0.0,
            "paid score should be penalized under unpaid: {score_paid}"
        );

        let ctx_disabled =
            PremiseContext::new("Account settings: two-factor authentication is disabled.");
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
        assert!(
            !ctx1.is_negated("database"),
            "'unresolved' must not negate 'database'"
        );

        // 2. "refund for the casino deposit" -> "deposit" must NOT be negated
        let ctx2 = PremiseContext::new("refund for the casino deposit");
        assert!(
            !ctx2.is_negated("deposit"),
            "'casino' must not negate 'deposit'"
        );

        // 3. "the prefixed invoice number" -> "invoice" must NOT be negated
        let ctx3 = PremiseContext::new("the prefixed invoice number");
        assert!(
            !ctx3.is_negated("invoice"),
            "'prefixed' must not negate 'invoice'"
        );

        // Clause boundaries: '.', ';', 'but'
        let ctx_dot = PremiseContext::new("ticket resolved. database outage ongoing");
        assert!(
            !ctx_dot.is_negated("database"),
            "sentence boundary '.' must stop negation"
        );

        let ctx_semi = PremiseContext::new("ticket resolved; database outage ongoing");
        assert!(
            !ctx_semi.is_negated("database"),
            "clause boundary ';' must stop negation"
        );

        let ctx_but = PremiseContext::new("ticket was resolved, but database outage still ongoing");
        assert!(
            !ctx_but.is_negated("database"),
            "clause boundary 'but' must stop negation"
        );

        // Legitimate negation within same clause
        let ctx_neg = PremiseContext::new("ticket was not resolved; no database access");
        assert!(
            ctx_neg.is_negated("database"),
            "'no database' must be negated"
        );

        // End-to-end P03: "ticket unresolved: database outage still ongoing"
        // database_outage must score higher than billing
        let candidates = vec![
            Candidate {
                id: "database_outage".into(),
                description: "database outage".into(),
                value: None,
            },
            Candidate {
                id: "billing".into(),
                description: "billing issue".into(),
                value: None,
            },
        ];
        let logits = compute_order_invariant_logits(
            "ticket unresolved: database outage still ongoing",
            &candidates,
        );
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
        let cand_normal = Candidate {
            id: "normal".into(),
            description: "Normal priority".into(),
            value: None,
        };
        let cand_standard = Candidate {
            id: "standard".into(),
            description: "Standard priority".into(),
            value: None,
        };
        let score_normal = ctx.score_candidate(&cand_normal);
        let score_standard = ctx.score_candidate(&cand_standard);
        assert_eq!(
            score_normal, score_standard,
            "Normal priority ({score_normal}) must equal Standard priority ({score_standard})"
        );
        assert_eq!(score_normal, 0.5);

        // P04: "Yesterday's orders" should NOT be boosted as affirmative
        let cand_yesterday = Candidate {
            id: "yesterday".into(),
            description: "Yesterday's orders".into(),
            value: None,
        };
        let cand_recent = Candidate {
            id: "recent".into(),
            description: "Recent orders".into(),
            value: None,
        };
        let score_yesterday = ctx.score_candidate(&cand_yesterday);
        let score_recent = ctx.score_candidate(&cand_recent);
        assert_eq!(
            score_yesterday, score_recent,
            "Yesterday's orders ({score_yesterday}) must equal Recent orders ({score_recent})"
        );
        assert_eq!(score_yesterday, 0.5);

        // Actual boolean candidates still get polarity scoring
        let cand_true = Candidate {
            id: "true".into(),
            description: "Yes".into(),
            value: None,
        };
        let cand_false = Candidate {
            id: "false".into(),
            description: "No".into(),
            value: None,
        };
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

    #[test]
    fn test_utf8_multibyte_safety() {
        // starts_with_word
        assert!(starts_with_word("🦀 hello", "🦀"));
        assert!(!starts_with_word("🦀hello", "🦀"));
        assert!(starts_with_word("café latte", "café"));
        assert!(!starts_with_word("caféteria", "café"));
        assert!(starts_with_word("word", "word"));
        assert!(!starts_with_word("", "word"));

        // ends_with_word
        assert!(ends_with_word("hello 🦀", "🦀"));
        assert!(!ends_with_word("hello🦀", "🦀"));
        assert!(ends_with_word("latte café", "café"));
        assert!(!ends_with_word("decafé", "café"));
        assert!(ends_with_word("🦀", "🦀"));
        assert!(!ends_with_word("", "🦀"));

        // contains_bounded_in
        assert!(contains_bounded_in("test 🦀 foo", "🦀"));
        assert!(contains_bounded_in("Привет мир", "мир"));
        assert!(!contains_bounded_in("Здравствуйте", "здрав"));
        assert!(!contains_bounded_in("", "мир"));
        assert!(!contains_bounded_in("мир", ""));

        // find_clause_start with multibyte
        assert_eq!(find_clause_start("Привет; but мир"), 17);
        assert_eq!(find_clause_start("Hello 🚀. Next"), 11);

        // PremiseContext with unicode & emoji
        let ctx = PremiseContext::new("Запрос 🚀: order is not ready yet.");
        assert!(ctx.is_negated("ready"));
        assert!(ctx.contains_bounded("ready"));

        let cand_ready = Candidate {
            id: "ready".into(),
            description: "Ready".into(),
            value: None,
        };
        let score = ctx.score_candidate(&cand_ready);
        assert!(score < 0.5); // negated
    }

    #[test]
    fn test_contrastive_carrier_token_discounting() {
        let cands_a = vec![
            Candidate {
                id: "tier1".into(),
                description: "Tier 1: Basic invoice support inquiry".into(),
                value: None,
            },
            Candidate {
                id: "tier2".into(),
                description: "Tier 2: Enterprise invoice support inquiry".into(),
                value: None,
            },
        ];

        // "invoice", "support", "inquiry" are carrier tokens across both candidates
        let carriers = extract_carrier_tokens(&cands_a);
        assert!(carriers.contains("invoice"));
        assert!(carriers.contains("support"));
        assert!(carriers.contains("inquiry"));

        let premise = "I need Tier 2 Enterprise assistance with my system";
        let logits = compute_order_invariant_logits(premise, &cands_a);
        assert!(
            logits[1] > logits[0],
            "Tier 2 should win decisively over Tier 1 without carrier interference"
        );

        // Permutation check: reversing candidates must yield exact permutation of logits
        let cands_rev = vec![cands_a[1].clone(), cands_a[0].clone()];
        let logits_rev = compute_order_invariant_logits(premise, &cands_rev);
        assert_eq!(logits[0], logits_rev[1]);
        assert_eq!(logits[1], logits_rev[0]);
    }
}
