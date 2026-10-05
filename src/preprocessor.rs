#[cfg(not(target_arch = "wasm32"))]
use chrono::Utc;
use chrono::{Duration, NaiveDate};
use regex::Regex;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::LazyLock;

// -------------------------------------------------------------------------------------------------
// 1. Pairwise Date Arithmetic (Synthesized from kev-rs)
// -------------------------------------------------------------------------------------------------

static RE_ISO_DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").expect("valid iso date regex"));

static RE_TEXT_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\b",
    )
    .expect("valid text date regex")
});

/// Computes pairwise date arithmetic facts between all dates mentioned in the text.
///
/// Ported from `kev-rs`. Eliminates temporal subtraction errors on warranty, return,
/// SLA, and flight schedule queries by making relative date math explicitly available.
pub fn compute_pairwise_date_facts(text: &str) -> String {
    let mut found = Vec::new();

    // Match ISO dates: YYYY-MM-DD
    for cap in RE_ISO_DATE.captures_iter(text) {
        let raw = cap.get(0).unwrap().as_str();
        if let Ok(d) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
            if !found.iter().any(|(r, _)| *r == raw) {
                found.push((raw.to_string(), d));
            }
        }
    }

    // Match Month D, YYYY dates
    for cap in RE_TEXT_DATE.captures_iter(text) {
        let raw = cap.get(0).unwrap().as_str();
        let month_str = &cap[1];
        let day_str = &cap[2];
        let year_str = &cap[3];
        let date_str = format!("{month_str} {day_str}, {year_str}");
        if let Ok(d) = NaiveDate::parse_from_str(&date_str, "%B %d, %Y") {
            if !found.iter().any(|(r, _)| *r == raw) {
                found.push((raw.to_string(), d));
            }
        }
    }

    if found.len() < 2 {
        return String::new();
    }

    let mut facts = Vec::new();
    for i in 0..found.len() {
        for j in (i + 1)..found.len() {
            let (raw_i, date_i) = &found[i];
            let (raw_j, date_j) = &found[j];
            let days = (*date_j - *date_i).num_days();

            if days == 0 {
                facts.push(format!("{raw_j} is the same day as {raw_i}."));
            } else if days > 0 {
                let s = if days == 1 { "" } else { "s" };
                facts.push(format!("{raw_j} is {days} day{s} after {raw_i}."));
            } else {
                let abs_d = days.abs();
                let s = if abs_d == 1 { "" } else { "s" };
                facts.push(format!("{raw_j} is {abs_d} day{s} before {raw_i}."));
            }
        }
    }

    facts.join(" ")
}

// -------------------------------------------------------------------------------------------------
// 2. Email, Header, & Noise Boilerplate Cleaning (Synthesized from laya-rs)
// -------------------------------------------------------------------------------------------------

static QUOTE_HEADERS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)^\s*On .{0,300}wrote:\s*$",
        r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}",
        r"^\s*_{8,}\s*$",
        r"(?i)^\s*From:\s.+$",
    ]
    .iter()
    .map(|pattern| Regex::new(pattern).expect("valid quote header pattern"))
    .collect()
});

static SIGNATURE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"^\s*--\s*$").unwrap(),
        Regex::new(r"(?i)^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[ \w,!.]*$").unwrap(),
        Regex::new(r"(?i)^\s*sent from my (iphone|android|mobile|ipad)").unwrap(),
    ]
});

static DISCLAIMER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|if you (have )?received this (e-?mail|message) in error|confidentiality notice:)",
    )
    .expect("valid disclaimer regex")
});

/// Remove quoted email history, signatures, and disclaimers to keep input focused.
/// Synthesized from `laya-rs`.
pub fn clean_email_body(body: &str) -> String {
    let text = body
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace("\\n", "\n");

    let mut lines: Vec<String> = Vec::new();
    for line in text.split('\n') {
        // Strip quote history headers
        if !lines.is_empty() && QUOTE_HEADERS.iter().any(|p| p.is_match(line)) {
            break;
        }
        // Strip quoted lines
        if line.trim_start().starts_with('>') {
            continue;
        }
        lines.push(line.trim_end().to_string());
    }

    // Inspect tail 40% for signature markers
    let count = lines.len();
    let tail_start = if count > 2 {
        (count as f64 * 0.6) as usize
    } else {
        count
    };
    let mut sig_idx = None;

    for (i, line) in lines.iter().enumerate().skip(tail_start) {
        if SIGNATURE_PATTERNS.iter().any(|p| p.is_match(line)) {
            sig_idx = Some(i);
            break;
        }
    }

    if let Some(idx) = sig_idx {
        lines.truncate(idx);
    }

    // Strip disclaimer lines
    let mut cleaned_lines = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let trimmed = line.trim();
        if DISCLAIMER_REGEX.is_match(&line)
            || (i > 0 && (trimmed.starts_with("---") || trimmed.starts_with("___")))
        {
            break;
        }
        cleaned_lines.push(line);
    }

    cleaned_lines.join("\n").trim().to_string()
}

/// Cleans email signatures, disclaimers, and boilerplate quotes.
/// Zero-allocation fast-path when no disclaimers are present.
pub fn clean_text<'a>(input: &'a str) -> Cow<'a, str> {
    let has_disclaimer = input.contains("---")
        || input.contains("___")
        || input.contains("On ")
        || input
            .as_bytes()
            .windows(15)
            .any(|w| w.eq_ignore_ascii_case(b"confidentiality"))
        || input
            .as_bytes()
            .windows(18)
            .any(|w| w.eq_ignore_ascii_case(b"this email and any"));

    if !has_disclaimer {
        return Cow::Borrowed(input.trim());
    }

    Cow::Owned(clean_email_body(input))
}

// -------------------------------------------------------------------------------------------------
// 3. Unicode Script Detection (Synthesized from laya-rs)
// -------------------------------------------------------------------------------------------------

pub const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScriptDetection {
    pub primary_script: String,
    pub is_english: bool,
    pub non_latin_fraction: f64,
}

/// Fast dependency-free Unicode script profiling.
/// Used to gate English token filtering and avoid mangling non-Latin inputs.
pub fn detect_script(text: &str) -> ScriptDetection {
    let mut total_alpha = 0usize;
    let mut non_latin = 0usize;
    let mut counts: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();

    for ch in text.chars() {
        if ch.is_alphabetic() {
            total_alpha += 1;
            let cp = ch as u32;

            // Latin range: ASCII letters + Latin-1 Supplement + Latin Extended
            let is_latin = (0x0041..=0x005A).contains(&cp)
                || (0x0061..=0x007A).contains(&cp)
                || (0x00C0..=0x024F).contains(&cp);

            if !is_latin {
                non_latin += 1;
                for &(name, ranges) in SCRIPT_RANGES {
                    if ranges.iter().any(|&(lo, hi)| cp >= lo && cp <= hi) {
                        *counts.entry(name).or_insert(0) += 1;
                        break;
                    }
                }
            }
        }
    }

    if total_alpha == 0 {
        return ScriptDetection {
            primary_script: "latin".to_string(),
            is_english: true,
            non_latin_fraction: 0.0,
        };
    }

    let non_latin_fraction = (non_latin as f64) / (total_alpha as f64);
    let is_english = non_latin_fraction < 0.15;

    let primary_script = if non_latin_fraction > 0.5 {
        counts
            .into_iter()
            .max_by_key(|&(_, c)| c)
            .map(|(s, _)| s.to_string())
            .unwrap_or_else(|| "non_latin".to_string())
    } else {
        "latin".to_string()
    };

    ScriptDetection {
        primary_script,
        is_english,
        non_latin_fraction,
    }
}

// -------------------------------------------------------------------------------------------------
// 4. Combined Preprocessor & Temporal Grounding
// -------------------------------------------------------------------------------------------------

/// Injects dynamic temporal reference facts and pairwise dates into text context.
pub fn inject_temporal_facts<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let has_relative_time = bytes
        .windows(5)
        .any(|w| w.eq_ignore_ascii_case(b"today") || w.eq_ignore_ascii_case(b"hours"))
        || bytes.windows(7).any(|w| {
            w.eq_ignore_ascii_case(b"days ag")
                || w.eq_ignore_ascii_case(b"yesterd")
                || w.eq_ignore_ascii_case(b"tomorro")
                || w.eq_ignore_ascii_case(b"last we")
                || w.eq_ignore_ascii_case(b"last mo")
        });

    let pairwise_facts = compute_pairwise_date_facts(text);

    if !has_relative_time && pairwise_facts.is_empty() {
        return Cow::Borrowed(text);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let now = Utc::now().date_naive();
    #[cfg(target_arch = "wasm32")]
    let now = chrono::NaiveDate::from_ymd_opt(2026, 10, 1).unwrap();
    let yesterday = now - Duration::days(1);
    let seven_days_ago = now - Duration::days(7);

    let mut result = text.trim().to_string();

    if !pairwise_facts.is_empty() {
        result.push_str(&format!("\n\n[Pairwise Date Facts: {pairwise_facts}]"));
    }

    if has_relative_time {
        result.push_str(&format!(
            "\n\n[Temporal Facts: reference_date={}, yesterday={}, 7_days_ago={}]",
            now, yesterday, seven_days_ago
        ));
    }

    Cow::Owned(result)
}

/// Formats structured JSON state (e.g. conversations, invoices, tickets) into clean natural text.
pub fn format_structured_state<'a>(input: &'a str) -> Cow<'a, str> {
    let trimmed = input.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Cow::Borrowed(input);
    }
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let mut parts = Vec::new();
        if let Some(serde_json::Value::Array(conv)) = map.get("conversation") {
            for turn in conv {
                let speaker = turn
                    .get("speaker")
                    .and_then(|v| v.as_str())
                    .unwrap_or("user");
                let text = turn.get("text").and_then(|v| v.as_str()).unwrap_or("");
                parts.push(format!("{}: {}", speaker, text));
            }
        }
        for (k, v) in &map {
            if k == "conversation" || k == "archive" {
                continue;
            }
            if let serde_json::Value::Object(sub) = v {
                let sub_strs: Vec<String> = sub
                    .iter()
                    .map(|(sk, sv)| {
                        match sv {
                            serde_json::Value::String(s) => format!("{}: {}", sk, s),
                            serde_json::Value::Object(nested) => {
                                let nest_strs: Vec<String> = nested
                                    .iter()
                                    .map(|(nk, nv)| format!("{}: {}", nk, nv))
                                    .collect();
                                format!("{}: {{{}}}", sk, nest_strs.join(", "))
                            }
                            _ => format!("{}: {}", sk, sv),
                        }
                    })
                    .collect();
                parts.push(format!("{}: {}", k, sub_strs.join(", ")));
            } else if !v.is_null() {
                parts.push(format!("{}: {}", k, v));
            }
        }
        if !parts.is_empty() {
            return Cow::Owned(parts.join("\n\n"));
        }
    }
    Cow::Borrowed(input)
}

static RE_UNTRUSTED_PAYLOAD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)((?:uploaded\s+file\s+preview\s+contains|preview\s+contains|text\s+layer\s+embedded\s+in\s+[^,]+\s+reads|contains\s+white-on-white\s+text|attachment\s+contains)[:,\s]+)[“"'\x60]([^“”"'\x60]+)[”"'\x60]"#).expect("valid untrusted regex")
});

/// Masks untrusted attachments, OCR text layers, and embedded prompt injection payloads.
pub fn mask_untrusted_payload<'a>(text: &'a str) -> Cow<'a, str> {
    if RE_UNTRUSTED_PAYLOAD.is_match(text) {
        let replaced = RE_UNTRUSTED_PAYLOAD.replace_all(text, "$1[UNTRUSTED_CONTENT_FILTERED]");
        Cow::Owned(replaced.into_owned())
    } else {
        Cow::Borrowed(text)
    }
}

pub fn preprocess_state<'a>(state_str: &'a str, enable_temporal: bool) -> Cow<'a, str> {
    let structured = format_structured_state(state_str);
    let base_text = match structured {
        Cow::Borrowed(s) => clean_text(s),
        Cow::Owned(ref s) => clean_text(s),
    };
    let unmasked = mask_untrusted_payload(&base_text);

    let mut enriched = if enable_temporal {
        inject_temporal_facts(&unmasked).into_owned()
    } else {
        unmasked.into_owned()
    };

    // 1. Generic markdown table resolution
    let table_relations = crate::table_graph::resolve_tabular_relations(&enriched);
    if !table_relations.is_empty() {
        enriched.push_str("\n\n");
        enriched.push_str(&table_relations.join("\n"));
    }

    // 2. Generic policy hierarchy resolution
    let policy_overrides = crate::table_graph::resolve_policy_hierarchy(&enriched);
    if !policy_overrides.is_empty() {
        enriched.push_str("\n\n");
        enriched.push_str(&policy_overrides.join("\n"));
    }

    // 3. Generic monetary sublimit and claim constraint checks
    let monetary_constraints = crate::table_graph::resolve_monetary_constraints(&enriched);
    if !monetary_constraints.is_empty() {
        enriched.push_str("\n\n");
        enriched.push_str(&monetary_constraints.join("\n"));
    }

    // 4. Generic symbolic arithmetic equation verification
    let arithmetic_findings = crate::symbolic::verify_arithmetic_equations(&enriched);
    if !arithmetic_findings.is_empty() {
        enriched.push_str("\n\n");
        enriched.push_str(&arithmetic_findings.join("\n"));
    }

    // 5. Generic structural constraint verification for request/response pairs
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(state_str.trim()) {
        if let (Some(req_val), Some(resp_val)) = (map.get("request"), map.get("response")) {
            if let (Some(req_str), Some(resp_str)) = (req_val.as_str(), resp_val.as_str()) {
                let structural_findings = crate::symbolic::verify_structural_constraints(req_str, resp_str);
                if !structural_findings.is_empty() {
                    enriched.push_str("\n\n");
                    enriched.push_str(&structural_findings.join("\n"));
                }
            }
        }
    }

    Cow::Owned(enriched)
}

// -------------------------------------------------------------------------------------------------
// 4. Entity & Protected-Range Masking (Ported from translate-rs/src/masker.rs)
// -------------------------------------------------------------------------------------------------

static RE_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(https?://[^\s<>"']+|www\.[^\s<>"']+)"#).expect("valid url regex")
});

static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"#).expect("valid email regex")
});

static RE_TEMPLATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\{\{[^}]+\}\}|\$\{[^}]+\})"#).expect("valid template placeholder regex")
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedSegment {
    pub text: String,
    pub is_protected: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EntityMaskResult {
    pub masked_text: String,
    pub placeholders: Vec<(String, String)>,
}

pub struct EntityMasker;

impl EntityMasker {
    pub fn protected_ranges(text: &str) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        ranges.extend(Self::backtick_ranges(text));

        for mat in RE_URL.find_iter(text) {
            ranges.push(mat.range());
        }

        for mat in RE_EMAIL.find_iter(text) {
            ranges.push(mat.range());
        }

        for mat in RE_TEMPLATE.find_iter(text) {
            ranges.push(mat.range());
        }

        ranges.sort_by(|a, b| {
            if a.start == b.start {
                a.end.cmp(&b.end)
            } else {
                a.start.cmp(&b.start)
            }
        });

        let mut merged: Vec<Range<usize>> = Vec::new();
        for range in ranges {
            if let Some(last) = merged.last_mut() {
                if range.start <= last.end {
                    last.end = last.end.max(range.end);
                    continue;
                }
            }
            merged.push(range);
        }
        merged
    }

    fn backtick_ranges(text: &str) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let bytes = text.as_bytes();
        let mut index = 0;
        let len = bytes.len();

        while index < len {
            if bytes[index] != b'`' {
                index += 1;
                continue;
            }

            let start = index;
            if index + 2 < len && bytes[index + 1] == b'`' && bytes[index + 2] == b'`' {
                let body_start = index + 3;
                if let Some(close_offset) = text[body_start..].find("```") {
                    let end = body_start + close_offset + 3;
                    ranges.push(start..end);
                    index = end;
                } else {
                    ranges.push(start..len);
                    break;
                }
            } else {
                let body_start = index + 1;
                if let Some(close_offset) = text[body_start..].find('`') {
                    let end = body_start + close_offset + 1;
                    ranges.push(start..end);
                    index = end;
                } else {
                    ranges.push(start..len);
                    break;
                }
            }
        }
        ranges
    }

    pub fn segments(text: &str) -> Vec<ProtectedSegment> {
        if text.is_empty() {
            return Vec::new();
        }
        let ranges = Self::protected_ranges(text);
        let mut segments = Vec::new();
        let mut cursor = 0;

        for r in ranges {
            if cursor < r.start {
                segments.push(ProtectedSegment {
                    text: text[cursor..r.start].to_string(),
                    is_protected: false,
                });
            }
            segments.push(ProtectedSegment {
                text: text[r.clone()].to_string(),
                is_protected: true,
            });
            cursor = r.end;
        }

        if cursor < text.len() {
            segments.push(ProtectedSegment {
                text: text[cursor..].to_string(),
                is_protected: false,
            });
        }
        segments
    }

    /// Masks all protected entities with deterministic tokens `__ZEV_MASK_{i}__`.
    pub fn mask(text: &str) -> EntityMaskResult {
        let segments = Self::segments(text);
        let mut masked = String::with_capacity(text.len());
        let mut placeholders = Vec::new();
        let mut mask_idx = 0;

        for seg in segments {
            if seg.is_protected {
                let tag = format!("__ZEV_MASK_{mask_idx}__");
                masked.push_str(&tag);
                placeholders.push((tag, seg.text));
                mask_idx += 1;
            } else {
                masked.push_str(&seg.text);
            }
        }

        EntityMaskResult {
            masked_text: masked,
            placeholders,
        }
    }

    /// Restores previously masked tokens to their original values.
    pub fn unmask(masked_text: &str, placeholders: &[(String, String)]) -> String {
        let mut result = masked_text.to_string();
        for (tag, orig) in placeholders {
            result = result.replace(tag, orig);
        }
        result
    }
}

// -------------------------------------------------------------------------------------------------
// 5. Structural Symbol Extraction (Ported from purgecss-rs/src/extractor.rs)
// -------------------------------------------------------------------------------------------------

static RE_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z0-9_\-/:@]+").expect("valid regex"));

static RE_CLASS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:class|className)\s*=\s*["'`]([^"'`]+)["'`]"#).expect("valid regex")
});

static RE_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"id\s*=\s*["'`]([^"'`]+)["'`]"#).expect("valid regex"));

static RE_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"</?([a-zA-Z][a-zA-Z0-9\-]*)(?:\s+[^>]*)?/?>"#).expect("valid regex")
});

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractedSymbols {
    pub classes: HashSet<String>,
    pub ids: HashSet<String>,
    pub tags: HashSet<String>,
    pub words: HashSet<String>,
}

impl ExtractedSymbols {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extract_from_content(&mut self, content: &str) {
        for mat in RE_WORD.find_iter(content) {
            self.words.insert(mat.as_str().to_string());
        }

        for cap in RE_CLASS.captures_iter(content) {
            for cls in cap[1].split_whitespace() {
                self.classes.insert(cls.to_string());
                self.words.insert(cls.to_string());
            }
        }

        for cap in RE_ID.captures_iter(content) {
            let id = cap[1].trim();
            if !id.is_empty() {
                self.ids.insert(id.to_string());
                self.words.insert(id.to_string());
            }
        }

        for cap in RE_TAG.captures_iter(content) {
            let tag = cap[1].to_lowercase();
            self.tags.insert(tag.clone());
            self.words.insert(tag);
        }
    }

    pub fn matches_word(&self, word: &str) -> bool {
        self.words.contains(word) || self.classes.contains(word) || self.ids.contains(word)
    }

    pub fn total_count(&self) -> usize {
        self.classes.len() + self.ids.len() + self.tags.len() + self.words.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_date_facts() {
        let text =
            "Order placed on 2026-09-01. Return requested on 2026-09-25. Must be within 30 days.";
        let facts = compute_pairwise_date_facts(text);
        assert!(facts.contains("2026-09-25 is 24 days after 2026-09-01"));
    }

    #[test]
    fn test_written_date_facts() {
        let text = "Incident opened September 10, 2026 and resolved September 14, 2026.";
        let facts = compute_pairwise_date_facts(text);
        assert!(facts.contains("September 14, 2026 is 4 days after September 10, 2026"));
    }

    #[test]
    fn test_email_body_cleaning() {
        let text = "The database cluster was restarted.\n\nOn Sep 24, 2026, John wrote:\n> Was it restarted?\n\nRegards,\nDevOps\nSent from my iPhone";
        let cleaned = clean_email_body(text);
        assert_eq!(cleaned, "The database cluster was restarted.");
    }

    #[test]
    fn test_script_detection() {
        let en = "High latency detected on checkout server";
        let det_en = detect_script(en);
        assert!(det_en.is_english);
        assert_eq!(det_en.primary_script, "latin");

        let ru = "Ошибка подключения к базе данных на сервере";
        let det_ru = detect_script(ru);
        assert!(!det_ru.is_english);
        assert_eq!(det_ru.primary_script, "cyrillic");
    }

    #[test]
    fn test_entity_masker() {
        let raw = "Check api at https://example.com/api/v1 and email support@zev.ai with `tok_123`";
        let masked = EntityMasker::mask(raw);
        assert!(masked.masked_text.contains("__ZEV_MASK_0__"));
        assert!(masked.masked_text.contains("__ZEV_MASK_1__"));
        assert!(masked.masked_text.contains("__ZEV_MASK_2__"));

        let restored = EntityMasker::unmask(&masked.masked_text, &masked.placeholders);
        assert_eq!(restored, raw);
    }

    #[test]
    fn test_extracted_symbols() {
        let markup = r#"<div id="status-card" class="card active-state"><button class="btn btn-primary">Retry</button></div>"#;
        let mut symbols = ExtractedSymbols::new();
        symbols.extract_from_content(markup);

        assert!(symbols.ids.contains("status-card"));
        assert!(symbols.classes.contains("card"));
        assert!(symbols.classes.contains("active-state"));
        assert!(symbols.classes.contains("btn"));
        assert!(symbols.classes.contains("btn-primary"));
        assert!(symbols.tags.contains("div"));
        assert!(symbols.tags.contains("button"));
        assert!(symbols.matches_word("status-card"));
        assert!(symbols.matches_word("btn-primary"));
    }
}
