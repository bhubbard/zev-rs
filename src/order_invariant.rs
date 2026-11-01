use crate::types::Candidate;
use smallvec::SmallVec;
use std::collections::HashMap;

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
    "no need",
    "do not need",
    "not needed",
    "not required",
    "without any",
    "without needing",
    "rather than",
    "instead of",
    "in place of",
    "replacing",
    "might use",
    "could use",
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

const POST_NEGATORS: &[&str] = &[
    "is not",
    "are not",
    "was not",
    "were not",
    "is disabled",
    "is inactive",
    "is absent",
    "is unpaid",
];

#[cfg(test)]
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

fn find_clause_start(text: &str) -> usize {
    let mut boundary_end = 0;

    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b';'
            || b == b'?'
            || b == b'!'
            || b == b'\n'
            || b == b'|'
            || b == b'{'
            || b == b'}'
            || b == b'['
            || b == b']'
        {
            boundary_end = boundary_end.max(i + 1);
        } else if b == b'.' {
            let prev_is_digit = i > 0 && bytes[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
            if !(prev_is_digit && next_is_digit) {
                boundary_end = boundary_end.max(i + 1);
            }
        }
    }

    // Check for contrastive clause-reset conjunctions ("but", "however")
    for conj in ["but", "however"] {
        let clen = conj.len();
        let mut search_start = 0;
        while let Some(sub) = text.get(search_start..) {
            let pos = match sub.find(conj) {
                Some(p) => p,
                None => break,
            };
            let actual_pos = search_start + pos;
            let end_pos = actual_pos + clen;

            let left_ok = if actual_pos == 0 {
                true
            } else {
                match text.get(..actual_pos).and_then(|s| s.chars().last()) {
                    Some(prev) => !prev.is_alphanumeric() && prev != '_',
                    None => false,
                }
            };

            let right_ok = if end_pos == text.len() {
                true
            } else {
                match text.get(end_pos..).and_then(|s| s.chars().next()) {
                    Some(next) => !next.is_alphanumeric() && next != '_',
                    None => false,
                }
            };

            if left_ok && right_ok {
                boundary_end = boundary_end.max(end_pos);
            }
            search_start = actual_pos + clen;
        }
    }

    boundary_end
}

/// Deterministic morphological stemmer for English content words.
/// Normalizes inflectional and derivational suffixes to a shared canonical root.
#[inline]
pub fn stem_english_word(word: &str) -> String {
    let w = word.trim().to_lowercase();
    if w.len() <= 3 || !w.is_ascii() {
        return w;
    }
    match w.as_str() {
        "lost" | "loss" | "losses" | "lose" | "loses" | "losing" => return "los".to_string(),
        "paid" | "payment" | "payments" | "paying" => return "pay".to_string(),
        "bought" | "buying" | "purchase" | "purchased" | "purchases" => return "purchas".to_string(),
        "sold" | "sale" | "sales" | "selling" => return "sal".to_string(),
        "broke" | "broken" | "breaking" => return "break".to_string(),
        "spoke" | "spoken" | "speech" | "speaking" => return "speak".to_string(),
        "wrote" | "written" | "writing" => return "writ".to_string(),
        _ => {}
    }
    let mut s = w.as_str();
    if s.ends_with("ies") && s.len() > 4 {
        return format!("{}y", &s[..s.len() - 3]);
    }
    if s.ends_with("ied") && s.len() > 4 {
        return format!("{}y", &s[..s.len() - 3]);
    }
    if s.ends_with("es") && s.len() > 4 {
        if s.ends_with("sses")
            || s.ends_with("ches")
            || s.ends_with("shes")
            || s.ends_with("xes")
        {
            s = &s[..s.len() - 2];
        } else {
            s = &s[..s.len() - 1];
        }
    } else if s.ends_with('s')
        && !s.ends_with("ss")
        && !s.ends_with("us")
        && !s.ends_with("is")
        && s.len() > 3
    {
        s = &s[..s.len() - 1];
    }

    for &(suffix, replacement) in &[
        ("ibility", "ibl"),
        ("ability", "abl"),
        ("bility", "bl"),
        ("llation", "l"),
        ("ation", ""),
        ("ition", "it"),
        ("ibly", "ibl"),
        ("ably", "abl"),
        ("ible", "ibl"),
        ("able", "abl"),
        ("ment", ""),
        ("ness", ""),
        ("ingly", ""),
        ("edly", ""),
        ("fully", ""),
        ("less", ""),
        ("ing", ""),
        ("ed", ""),
        ("al", ""),
        ("ly", ""),
        ("er", ""),
        ("y", ""),
    ] {
        if s.len() >= suffix.len() + 3 && s.ends_with(suffix) {
            let base = &s[..s.len() - suffix.len()];
            let mut out = format!("{}{}", base, replacement);
            if out.ends_with('e') && out.len() > 4 {
                out.pop();
            }
            if out.len() >= 4 {
                let bytes = out.as_bytes();
                let n = bytes.len();
                if bytes[n - 1] == bytes[n - 2]
                    && matches!(
                        bytes[n - 1],
                        b'l' | b'p' | b't' | b'n' | b'r' | b'm' | b'g' | b'd' | b'b'
                    )
                {
                    out.pop();
                }
            }
            return out;
        }
    }

    let mut out = s.to_string();
    if out.ends_with('e') && out.len() > 4 {
        out.pop();
    }
    out
}

#[derive(Debug)]
pub struct PremiseContext {
    pub raw_lower: String,
    pub token_positions: HashMap<String, SmallVec<[u32; 4]>, crate::fast_hash::Xxh3BuildHasher>,
    pub stem_positions: HashMap<String, SmallVec<[u32; 4]>, crate::fast_hash::Xxh3BuildHasher>,
    pub synset_positions: HashMap<u16, SmallVec<[u32; 4]>, crate::fast_hash::Xxh3BuildHasher>,
}

impl PremiseContext {
    #[inline]
    pub fn new(premise: &str) -> Self {
        let raw_lower = premise.to_lowercase();
        let mut token_positions: HashMap<
            String,
            SmallVec<[u32; 4]>,
            crate::fast_hash::Xxh3BuildHasher,
        > = HashMap::with_hasher(crate::fast_hash::Xxh3BuildHasher);
        let mut stem_positions: HashMap<
            String,
            SmallVec<[u32; 4]>,
            crate::fast_hash::Xxh3BuildHasher,
        > = HashMap::with_hasher(crate::fast_hash::Xxh3BuildHasher);
        let mut synset_positions: HashMap<
            u16,
            SmallVec<[u32; 4]>,
            crate::fast_hash::Xxh3BuildHasher,
        > = HashMap::with_hasher(crate::fast_hash::Xxh3BuildHasher);
        let mut in_token = false;
        let mut token_start = 0;

        let mut index_subtoken = |sub: &str, pos_u32: u32| {
            if sub.len() >= 3 {
                if let Some(entry) = crate::lexicon::lookup_word(sub) {
                    if entry.synset_id != 0 {
                        synset_positions
                            .entry(entry.synset_id)
                            .or_default()
                            .push(pos_u32);
                    }
                }
            }
            if sub.len() >= 4 {
                let stem = stem_english_word(sub);
                if stem.len() >= 3 {
                    if let Some(entry) = crate::lexicon::lookup_word(&stem) {
                        if entry.synset_id != 0 {
                            synset_positions
                                .entry(entry.synset_id)
                                .or_default()
                                .push(pos_u32);
                        }
                    }
                    stem_positions.entry(stem).or_default().push(pos_u32);
                }
            }
        };

        for (byte_idx, ch) in raw_lower.char_indices() {
            if ch.is_alphanumeric() || ch == '_' {
                if !in_token {
                    in_token = true;
                    token_start = byte_idx;
                }
            } else if in_token {
                in_token = false;
                let token = &raw_lower[token_start..byte_idx];
                let pos_u32 = token_start as u32;
                token_positions
                    .entry(token.to_string())
                    .or_default()
                    .push(pos_u32);
                for sub in token.split('_') {
                    index_subtoken(sub, pos_u32);
                }
            }
        }
        if in_token {
            let token = &raw_lower[token_start..];
            let pos_u32 = token_start as u32;
            token_positions
                .entry(token.to_string())
                .or_default()
                .push(pos_u32);
            for sub in token.split('_') {
                index_subtoken(sub, pos_u32);
            }
        }

        Self {
            raw_lower,
            token_positions,
            stem_positions,
            synset_positions,
        }
    }

    #[inline]
    fn is_pos_negated(&self, actual_pos: usize) -> bool {
        let clause_start = find_clause_start(self.raw_lower.get(..actual_pos).unwrap_or(""));

        let prefix = self
            .raw_lower
            .get(clause_start..actual_pos)
            .unwrap_or("")
            .trim_end();
        for &neg in PREFIX_NEGATORS {
            if ends_with_word(prefix, neg) {
                return true;
            }
        }

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
        // Sub-clause syntactic negators ("no", "not", "never", "without", "neither", "cannot")
        // when not separated by a coordinating conjunction (" and ", " or ", ",")
        let sub_clause = match window
            .rfind(" and ")
            .map(|p| p + 5)
            .into_iter()
            .chain(window.rfind(" or ").map(|p| p + 4))
            .chain(window.rfind(',').map(|p| p + 1))
            .max()
        {
            Some(sub_start) => &window[sub_start..],
            None => window,
        };
        for &neg in ["no", "not", "never", "without", "neither", "cannot"].iter() {
            if contains_bounded_in(sub_clause, neg) {
                return true;
            }
        }
        for &neg in SCOPED_NEGATORS {
            if contains_bounded_in(window, neg) {
                return true;
            }
        }

        // Check immediate post-modifiers within the same clause (e.g. "X is not", "X policy is clearer")
        let tail = self.raw_lower.get(actual_pos..).unwrap_or("");
        let rel_clause_end = tail
            .find(['.', ';', '?', '!', '\n', '|', '{', '}', '[', ']'])
            .unwrap_or(tail.len())
            .min(40);
        let mut post_end = actual_pos + rel_clause_end;
        while post_end > actual_pos && !self.raw_lower.is_char_boundary(post_end) {
            post_end -= 1;
        }
        let post_window = self.raw_lower.get(actual_pos..post_end).unwrap_or("");
        for &post_neg in POST_NEGATORS {
            if contains_bounded_in(post_window, post_neg) {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn contains_stem(&self, word: &str) -> bool {
        if word.len() < 4 {
            return false;
        }
        let stem = stem_english_word(word);
        stem.len() >= 3 && self.stem_positions.contains_key(&stem)
    }

    #[inline]
    pub fn is_stem_negated(&self, word: &str) -> bool {
        if word.len() < 4 {
            return false;
        }
        let stem = stem_english_word(word);
        if let Some(positions) = self.stem_positions.get(&stem) {
            positions.iter().any(|&pos| self.is_pos_negated(pos as usize))
        } else {
            false
        }
    }

    #[inline]
    pub fn contains_synset(&self, synset_id: u16) -> bool {
        synset_id != 0 && self.synset_positions.contains_key(&synset_id)
    }

    #[inline]
    pub fn is_synset_negated(&self, synset_id: u16) -> bool {
        if let Some(positions) = self.synset_positions.get(&synset_id) {
            positions.iter().any(|&pos| self.is_pos_negated(pos as usize))
        } else {
            false
        }
    }

    #[inline]
    pub fn has_affirmative_synset(&self, synset_id: u16) -> bool {
        if let Some(positions) = self.synset_positions.get(&synset_id) {
            positions.iter().any(|&pos| !self.is_pos_negated(pos as usize))
        } else {
            false
        }
    }

    #[inline]
    pub fn synset_recency_weight(&self, synset_id: u16) -> f64 {
        if let Some(positions) = self.synset_positions.get(&synset_id) {
            if let Some(&pos) = positions.last() {
                return 1.0 + (pos as f64 / self.raw_lower.len().max(1) as f64) * 0.6;
            }
        }
        1.0
    }

    /// Checks whether the premise contains a morphological antonym (`un-`, `in-`, `im-`, `ir-`, `dis-`)
    /// or strict lexical antonym of `word`.
    #[inline]
    pub fn has_morphological_antonym(&self, word: &str) -> bool {
        let w = word.trim().to_lowercase();
        match w.as_str() {
            "paid" if self.contains_bounded("unpaid") => return true,
            "unpaid" if self.contains_bounded("paid") && !self.is_negated("paid") => return true,
            "enabled" if self.contains_bounded("disabled") => return true,
            "disabled" if self.contains_bounded("enabled") && !self.is_negated("enabled") => {
                return true
            }
            "active" if self.contains_bounded("inactive") => return true,
            "inactive" if self.contains_bounded("active") && !self.is_negated("active") => {
                return true
            }
            "present" if self.contains_bounded("absent") => return true,
            "absent" if self.contains_bounded("present") && !self.is_negated("present") => {
                return true
            }
            "able" if self.contains_bounded("unable") || self.contains_bounded("cannot") => {
                return true
            }
            _ => {}
        }

        if w.len() < 4 || !w.is_ascii() {
            return false;
        }
        let stem = stem_english_word(&w);

        if w.len() >= 5 && stem.len() >= 4 {
            // If `w` itself is negated by a prefix ("un...", "in...", "im...", "ir...", "dis..."),
            // check if the base stem is affirmatively present while the prefixed stem is NOT present.
            for prefix in ["un", "in", "im", "ir", "dis"] {
                if let Some(base_stem) = stem.strip_prefix(prefix) {
                    if base_stem.len() >= 4
                        && !self.stem_positions.contains_key(&stem)
                        && self.stem_positions.contains_key(base_stem)
                    {
                        return true;
                    }
                } else {
                    let prefixed_stem = format!("{}{}", prefix, stem);
                    if self.stem_positions.contains_key(&prefixed_stem) {
                        return true;
                    }
                }
            }
        }

        if let Some(entry) = crate::lexicon::lookup_word_or_stem(&w, &stem) {
            if entry.antonym_synset_id != 0
                && !self.contains_bounded(&w)
                && !self.stem_positions.contains_key(&stem)
                && !self.has_affirmative_synset(entry.synset_id)
                && self.has_affirmative_synset(entry.antonym_synset_id)
            {
                return true;
            }
        }

        false
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
            "able" if self.contains_bounded("unable") || self.contains_bounded("cannot") => {
                return true
            }
            _ => {}
        }

        let word_lower;
        let w = if word.chars().any(|c| c.is_uppercase()) {
            word_lower = word.to_lowercase();
            &word_lower
        } else {
            word
        };

        // The lowercase token "a" in English prose is the indefinite article ("not a problem",
        // "without a doubt"), where negation scopes over the head noun rather than the symbol "A".
        if w == "a" {
            return false;
        }

        if let Some(positions) = self.token_positions.get(w) {
            return positions.iter().any(|&pos| self.is_pos_negated(pos as usize));
        }

        // If w is a single word token and not in token_positions, it cannot appear as a bounded token in raw_lower
        if w.chars().all(|c| c.is_alphanumeric() || c == '_') {
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

            let prefix = self
                .raw_lower
                .get(clause_start..actual_pos)
                .unwrap_or("")
                .trim_end();
            for &neg in PREFIX_NEGATORS {
                if ends_with_word(prefix, neg) {
                    return true;
                }
            }

            // Window check for scoped negations and disclaimers (Winnow-12B protocol)
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
        const ID_PART_STOPWORDS: &[&str] = &[
            "the", "and", "for", "are", "was", "with", "that", "from", "this", "these", "those",
            "than", "then", "have", "been", "were", "some", "yes", "all", "what", "when", "where",
            "which", "who", "whom", "whose", "why", "how", "you", "your", "yours", "they", "them",
            "their", "theirs", "our", "ours", "him", "his", "her", "hers", "she", "does", "did",
            "has", "had", "into", "onto", "upon", "about", "there", "here", "just", "very", "also",
            "will", "would", "could", "should", "can", "may", "might", "must", "shall",
        ];

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
                        const GENERIC_PARTS: [&str; 18] = [
                            "order", "question", "action", "task", "service", "item", "query",
                            "info", "type", "call", "option", "choice", "make", "give", "find",
                            "tell", "show", "list",
                        ];
                        for part in id_lower.split(['_', '-']) {
                            if part.len() >= 4
                                && !part.chars().all(|c| c.is_ascii_digit())
                                && !ID_PART_STOPWORDS.contains(&part)
                            {
                                let weight_scale = if GENERIC_PARTS.contains(&part) {
                                    0.6
                                } else {
                                    2.2
                                };
                                if self.contains_bounded(part) {
                                    let w = self.recency_weight(part);
                                    if self.is_negated(part) {
                                        logit -= 1.8 * w;
                                    } else {
                                        logit += weight_scale * w;
                                    }
                                } else if self.contains_stem(part) {
                                    let stem = stem_english_word(part);
                                    let w = self.recency_weight(&stem);
                                    if self.is_stem_negated(part) {
                                        logit -= 1.4 * w;
                                    } else {
                                        logit += (weight_scale * 0.8) * w;
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
        let is_multi_class = carrier_tokens.is_some();
        let has_yes_interjection = (!is_multi_class && desc_trimmed == "yes")
            || desc_trimmed.starts_with("yes,")
            || desc_trimmed.starts_with("yes:")
            || desc_trimmed.starts_with("yes.")
            || desc_trimmed.starts_with("yes;")
            || desc_trimmed.starts_with("yes -");
        let has_no_interjection = (!is_multi_class && desc_trimmed == "no")
            || desc_trimmed.starts_with("no,")
            || desc_trimmed.starts_with("no:")
            || desc_trimmed.starts_with("no.")
            || desc_trimmed.starts_with("no;")
            || desc_trimmed.starts_with("no -");

        let is_bool_opt = id == "true"
            || id == "false"
            || (!is_multi_class && (id == "yes" || id == "no"))
            || has_yes_interjection
            || has_no_interjection
            || desc_trimmed.starts_with("every required")
            || desc_trimmed.starts_with("a condition is missing");

        let is_affirmative = is_bool_opt
            && (id == "true"
                || (!is_multi_class && id == "yes")
                || (id != "false"
                    && id != "no"
                    && (has_yes_interjection
                        || desc_trimmed.starts_with("every required"))));

        let is_negative = is_bool_opt
            && (id == "false"
                || (!is_multi_class && id == "no")
                || (id != "true"
                    && id != "yes"
                    && (has_no_interjection
                        || desc_trimmed.starts_with("a condition is missing"))));

        if is_affirmative || is_negative {
            // Check explicit symbolic / temporal / policy-DAG verifier findings first
            let has_verifier_fail = self.raw_lower.contains("calculation_valid: false")
                || self.raw_lower.contains("constraint_satisfied: false")
                || self.raw_lower.contains("deduction_valid: false")
                || self.raw_lower.contains("within_window: false")
                || self.raw_lower.contains("exceeds_sublimit: true")
                || self.raw_lower.contains("exception_active: true");

            let has_verifier_pass = !has_verifier_fail
                && (self.raw_lower.contains("calculation_valid: true")
                    || self.raw_lower.contains("constraint_satisfied: true")
                    || self.raw_lower.contains("deduction_valid: true")
                    || self.raw_lower.contains("within_window: true"));

            if has_verifier_fail {
                if is_negative {
                    logit += 3.5;
                } else {
                    logit -= 2.5;
                }
            } else if has_verifier_pass {
                if is_affirmative {
                    logit += 3.5;
                } else {
                    logit -= 2.5;
                }
            } else {
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

                let has_strong_pos = self.contains_bounded("paid in full")
                    || (self.contains_bounded("is enabled") && !self.is_negated("enabled"));

                if is_affirmative {
                    if !has_neg || has_strong_pos {
                        logit += 2.4;
                    } else {
                        logit -= 1.5;
                    }
                } else if is_negative {
                    if has_neg && !has_strong_pos {
                        logit += 2.2;
                    } else {
                        logit -= 1.5;
                    }
                }
            }
        }

        let is_abstain_opt = crate::abstain::is_abstain_candidate(id, desc_str);
        let is_catalog_option = id.starts_with("option_");
        let mut desc_logit: f64 = 0.0;
        let mut desc_content_words: usize = 0;

        // Clause-scoped 2x2 polarity and morphological stem matching across description
        for clause in desc_str.split(['.', ';', ',', ':', '!', '?', '\n']) {
            let mut cand_neg_remaining: usize = 0;

            for word in clause.split(|c: char| !c.is_alphanumeric() && c != '_') {
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

                // Candidate-side negation operators ("no", "not", "without", "never", "neither", "none")
                // act as polarity inverters on subsequent content words in the same clause rather than content tokens.
                if matches!(
                    word_str,
                    "no" | "not" | "without" | "never" | "neither" | "none"
                ) {
                    if !is_abstain_opt {
                        cand_neg_remaining = 3;
                    }
                    continue;
                }

                // Contrastive background carrier token discounting (0.20x if word is shared across >=75% of candidates)
                let is_carrier = carrier_tokens.is_some_and(|set| set.contains(word_str));
                let carrier_scale = if is_carrier { 0.20 } else { 1.0 };

                // Carrier tokens represent shared domain topics (e.g. "function" in "No function impaired");
                // candidate-side negation applies to differentiating predicates rather than affirmative carrier nouns.
                let cand_word_negated = (cand_neg_remaining > 0) && !is_carrier;

                // Bilateral morphological antonym check (un-, in-, im-, ir-, dis-)
                if self.has_morphological_antonym(word_str) {
                    desc_content_words += 1;
                    let w = self.recency_weight(word_str) * carrier_scale;
                    if cand_word_negated {
                        desc_logit += 1.8 * w;
                    } else {
                        desc_logit -= 1.8 * w;
                    }
                    cand_neg_remaining = cand_neg_remaining.saturating_sub(1);
                    continue;
                }

                if char_count <= 2 {
                    // Word-boundary isolation for short/single-character options (length <= 2)
                    const COMMON_STOPWORDS: [&str; 18] = [
                        "a", "an", "in", "on", "at", "to", "is", "it", "or", "of", "by", "as",
                        "if", "be", "do", "we", "he", "so",
                    ];
                    const CATALOG_SHORT_STOPWORDS: [&str; 9] =
                        ["i", "my", "me", "am", "us", "up", "go", "no", "ok"];
                    if (!COMMON_STOPWORDS.contains(&word_str)
                        && !(is_catalog_option && CATALOG_SHORT_STOPWORDS.contains(&word_str)))
                        || desc_trimmed == word_str
                    {
                        desc_content_words += 1;
                        if self.contains_bounded(word_str) {
                            let w = self.recency_weight(word_str) * carrier_scale;
                            let premise_negated = self.is_negated(word_str);
                            if premise_negated == cand_word_negated {
                                desc_logit += 1.8 * w;
                            } else {
                                desc_logit -= 1.8 * w;
                            }
                            cand_neg_remaining = cand_neg_remaining.saturating_sub(1);
                        }
                    }
                } else {
                    const COMMON_LONG_STOPWORDS: &[&str] = &[
                        "the", "and", "for", "are", "was", "with", "that", "from", "this", "these",
                        "those", "than", "then", "have", "been", "were", "some", "yes", "all",
                        "what", "why", "how", "you", "your", "yours", "they", "them", "their",
                        "theirs", "our", "ours", "him", "his", "her", "hers", "she", "does", "did",
                        "has", "had", "into", "onto", "upon", "about", "there", "here", "just",
                        "very", "also",
                    ];
                    if COMMON_LONG_STOPWORDS.contains(&word_str)
                        || (is_catalog_option && ID_PART_STOPWORDS.contains(&word_str))
                    {
                        continue;
                    }
                    if !is_carrier {
                        desc_content_words += 1;
                    }

                    let is_ascii = word_str.is_ascii();
                    let matched = if is_ascii {
                        self.contains_bounded(word_str)
                    } else {
                        self.raw_lower.contains(word_str)
                    };

                    const GENERIC_DESC_PARTS: [&str; 10] = [
                        "order", "question", "action", "task", "service", "item", "query", "info",
                        "type", "call",
                    ];
                    let generic_scale = if GENERIC_DESC_PARTS.contains(&word_str) {
                        0.25
                    } else {
                        1.0
                    };

                    if matched {
                        let w = self.recency_weight(word_str) * carrier_scale * generic_scale;
                        let premise_negated = self.is_negated(word_str);
                        if premise_negated == cand_word_negated {
                            desc_logit += 1.8 * w;
                        } else {
                            desc_logit -= 1.8 * w;
                        }
                    } else if is_ascii && char_count >= 4 {
                        // Morphological stem match or prefix stem check
                        if self.contains_stem(word_str) {
                            let stem = stem_english_word(word_str);
                            let stem_is_carrier =
                                carrier_tokens.is_some_and(|set| set.contains(&stem));
                            let stem_carrier_scale =
                                if stem_is_carrier { 0.20 } else { carrier_scale };
                            let w = self.recency_weight(&stem) * stem_carrier_scale;
                            let premise_negated = self.is_stem_negated(word_str);
                            if premise_negated == cand_word_negated {
                                desc_logit += 1.5 * w;
                            } else {
                                desc_logit -= 1.5 * w;
                            }
                        } else {
                            let mut prefix_matched = false;
                            if char_count >= 5 {
                                let stem = word_str.get(..word_str.len() - 1).unwrap_or("");
                                if self.contains_bounded(stem) {
                                    prefix_matched = true;
                                    let stem_is_carrier =
                                        carrier_tokens.is_some_and(|set| set.contains(stem));
                                    let stem_carrier_scale =
                                        if stem_is_carrier { 0.20 } else { carrier_scale };
                                    let w = self.recency_weight(stem) * stem_carrier_scale;
                                    let premise_negated = self.is_negated(stem);
                                    if premise_negated == cand_word_negated {
                                        desc_logit += 1.4 * w;
                                    } else {
                                        desc_logit -= 1.4 * w;
                                    }
                                }
                            }
                            if !prefix_matched {
                                let stem = stem_english_word(word_str);
                                if let Some(entry) =
                                    crate::lexicon::lookup_word_or_stem(word_str, &stem)
                                {
                                    if self.contains_synset(entry.synset_id) {
                                        let syn_key = format!("__syn_{}", entry.synset_id);
                                        let syn_is_carrier = carrier_tokens
                                            .is_some_and(|set| set.contains(&syn_key));
                                        let syn_carrier_scale =
                                            if syn_is_carrier { 0.20 } else { carrier_scale };
                                        let w = self.synset_recency_weight(entry.synset_id)
                                            * syn_carrier_scale
                                            * generic_scale;
                                        let premise_negated =
                                            self.is_synset_negated(entry.synset_id);
                                        if premise_negated == cand_word_negated {
                                            desc_logit += 1.05 * w;
                                        } else {
                                            desc_logit -= 1.05 * w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    cand_neg_remaining = cand_neg_remaining.saturating_sub(1);
                }
            }
        }

        // BM25-style candidate length normalization on lettered multiple-choice options ('A'..'E')
        // to prevent verbose distractors that echo multiple background nouns from overwhelming concise options.
        let is_mc_option =
            id.len() == 1 && matches!(id, "A" | "B" | "C" | "D" | "E") && desc_str.trim().len() > 2;
        if is_mc_option && desc_logit > 0.0 && desc_content_words > 2 {
            let norm = 1.0 / (1.0 + 0.25 * (desc_content_words.saturating_sub(2) as f64));
            logit += desc_logit * norm;
        } else {
            logit += desc_logit;
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

    let mut token_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for c in candidates {
        let mut seen = std::collections::HashSet::new();
        for word in c
            .description
            .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        {
            let w = word.trim().to_lowercase();
            if w.len() >= 3 && seen.insert(w.clone()) {
                *token_counts.entry(w).or_insert(0) += 1;
            }
        }
    }

    token_counts
        .into_iter()
        .filter(|&(_, count)| count >= threshold_count)
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
    let mut logits: Vec<f64> = if candidates.len() >= 2 {
        let carriers = extract_carrier_tokens(candidates);
        if !carriers.is_empty() {
            candidates
                .iter()
                .map(|c| {
                    ctx.score_candidate_raw_contrastive(&c.id, &c.description, Some(&carriers))
                })
                .collect()
        } else {
            candidates.iter().map(|c| ctx.score_candidate(c)).collect()
        }
    } else {
        candidates.iter().map(|c| ctx.score_candidate(c)).collect()
    };

    // Abstention / Catch-all Fallback:
    // If the candidate set includes an explicit abstention/catch-all option ("unknown", "other", "none", "oos"),
    // no concrete candidate has positive evidence above 0.5, and at least one concrete candidate was
    // explicitly negated/rejected (min_concrete < 0.5), elevate the abstention option.
    if candidates.len() >= 2 {
        let mut max_concrete = f64::NEG_INFINITY;
        let mut min_concrete = f64::INFINITY;
        let mut has_abstain = false;
        let mut has_concrete = false;
        for (c, &l) in candidates.iter().zip(logits.iter()) {
            if crate::abstain::is_abstain_candidate(&c.id, &c.description) {
                has_abstain = true;
            } else {
                has_concrete = true;
                if l > max_concrete {
                    max_concrete = l;
                }
                if l < min_concrete {
                    min_concrete = l;
                }
            }
        }

        if has_abstain
            && has_concrete
            && max_concrete <= 0.5 + 1e-6
            && min_concrete < 0.5 - 1e-6
        {
            for (c, l) in candidates.iter().zip(logits.iter_mut()) {
                if crate::abstain::is_abstain_candidate(&c.id, &c.description) {
                    *l = 2.0;
                }
            }
        }
    }

    logits
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
