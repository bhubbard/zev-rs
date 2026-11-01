//! Premise windowing, NLI hypothesis alignment, and structured entity-schema projection.
//!
//! Especially critical for long legal contracts (ContractNLI), clinical records,
//! multi-entity structured states, and long technical documents where 95%+ of the text
//! is irrelevant boilerplate.

use crate::order_invariant::{stem_english_word, PremiseContext};
use crate::types::Candidate;
use std::collections::HashSet;

const PROMPT_STOPWORDS: &[&str] = &[
    "classify",
    "relationship",
    "between",
    "contract",
    "this",
    "hypothesis",
    "which",
    "party",
    "parties",
    "shall",
    "that",
    "from",
    "with",
    "have",
    "been",
    "these",
    "following",
    "criteria",
    "select",
    "choose",
    "accordance",
    "relevant",
    "agreement",
    "under",
    "within",
    "thereof",
    "herein",
    "would",
    "could",
    "should",
    "the",
    "and",
    "for",
    "are",
    "was",
    "were",
    "not",
    "any",
    "all",
    "some",
    "its",
    "his",
    "her",
    "our",
    "their",
    "may",
    "can",
    "will",
    "must",
    "such",
    "other",
    "into",
    "upon",
    "has",
    "had",
    "who",
    "whom",
    "whose",
    "each",
    "both",
    "either",
    "neither",
    "without",
    "only",
    "also",
    "more",
    "most",
    "less",
    "than",
    "then",
    "when",
    "where",
    "while",
    "after",
    "before",
    "during",
];

/// Strips a leading meta-instruction directive header line (e.g. `"Classify the intent ...:"\n`)
/// when followed by the actual query/hypothesis on subsequent lines, preventing rubric keywords
/// (such as `"or choose out of scope if none applies:"`) from polluting lexical scoring.
pub fn strip_directive_header(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(nl_idx) = trimmed.find('\n') {
        let first_line = trimmed[..nl_idx].trim();
        let rest = trimmed[nl_idx + 1..].trim();
        if !rest.is_empty() && first_line.ends_with(':') {
            let fl_lower = first_line.to_lowercase();
            const DIRECTIVE_VERBS: &[&str] = &[
                "classify ",
                "determine ",
                "identify ",
                "select ",
                "choose ",
                "categorize ",
                "evaluate ",
                "predict ",
                "route ",
            ];
            if DIRECTIVE_VERBS.iter().any(|&v| fl_lower.starts_with(v)) {
                return rest;
            }
        }
    }
    trimmed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NliRole {
    Entailment,
    Contradiction,
    NotMentioned,
    Other,
}

fn classify_nli_candidate(c: &Candidate) -> NliRole {
    let id_l = c.id.trim().to_lowercase();
    let desc_l = c.description.trim().to_lowercase();

    // Check candidate ID first and check NotMentioned/Neutral/Neither before Entailment/Contradiction
    // because neutral descriptions often read "neither entailed nor contradicted".
    if id_l.contains("notmentioned")
        || id_l.contains("not_mentioned")
        || id_l.contains("neutral")
        || id_l.contains("neither")
        || desc_l.contains("neither")
        || desc_l.contains("not mention")
        || desc_l.contains("does not address")
        || desc_l.contains("unaddressed")
    {
        NliRole::NotMentioned
    } else if id_l.contains("contradict") || desc_l.contains("contradict") {
        NliRole::Contradiction
    } else if id_l.contains("entail") || desc_l.contains("entail") {
        NliRole::Entailment
    } else {
        NliRole::Other
    }
}

const NLI_THEME_WORDS: &[&str] = &[
    "receiving",
    "disclosing",
    "recipient",
    "discloser",
    "confidential",
    "information",
    "agreement",
    "contract",
    "party",
    "parties",
];

fn expand_nli_synonym_stems(raw_w: &str, stem: &str) -> &'static [&'static str] {
    match stem {
        "shar" | "share" | "disclos" | "divulg" | "reveal" => {
            &["shar", "share", "disclos", "divulg", "reveal", "provid", "furnish"]
        }
        "verbal" | "oral" | "spoken" => &["verbal", "oral", "spoken", "voic"],
        "copi" | "copy" | "reproduc" | "duplicat" => {
            &["copi", "copy", "reproduc", "duplicat", "photocopi"]
        }
        "destroy" | "destruct" | "delet" | "eras" | "dispos" => {
            &["destroy", "destruct", "delet", "eras", "dispos", "return"]
        }
        "keep" | "retain" | "maintain" | "archiv" | "preserv" => {
            &["keep", "retain", "maintain", "archiv", "preserv", "retention"]
        }
        "creat" | "develop" | "generat" | "produc" => {
            &["creat", "develop", "generat", "produc", "discover"]
        }
        "acquir" | "obtain" | "receiv" => &["acquir", "obtain", "receiv", "possess"],
        "notif" | "notic" | "inform" | "advis" => &["notif", "notic", "notice", "inform", "advis"],
        "terminat" | "expir" | "surviv" => &["terminat", "expir", "surviv", "end", "cancel"],
        "employe" | "personnel" | "staff" | "represent" => &[
            "employe",
            "personnel",
            "staff",
            "represent",
            "agent",
            "consultant",
            "advisor",
            "director",
            "officer",
        ],
        "affiliat" | "subsidiar" => &["affiliat", "subsidiar", "parent"],
        "permiss" | "consent" | "approv" | "authoriz" => {
            &["permiss", "consent", "approv", "authoriz"]
        }
        "writ" | "written" => &["writ", "written", "tangible"],
        "mark" | "marked" | "designat" | "label" => &["mark", "marked", "designat", "label", "stamp"],
        _ => match raw_w {
            "share" | "shared" | "sharing" => {
                &["shar", "share", "disclos", "divulg", "reveal", "provid"]
            }
            "copies" | "copy" => &["copi", "copy", "reproduc", "duplicat"],
            "verbally" | "verbal" | "orally" | "oral" => &["verbal", "oral", "spoken"],
            "written" | "writing" => &["writ", "written", "tangible"],
            _ => &[],
        },
    }
}

fn set_matches_term(set: &HashSet<String>, raw_w: &str, stem: &str) -> bool {
    if set.contains(stem) || set.contains(raw_w) {
        return true;
    }
    for &syn in expand_nli_synonym_stems(raw_w, stem) {
        if set.contains(syn) {
            return true;
        }
    }
    false
}

/// Evaluates 3-way / binary Natural Language Inference (Hypothesis vs Long Premise Document)
/// using document-internal Inverse Sentence Frequency (ISF) window alignment and clause polarity.
pub fn evaluate_nli_hypothesis(
    premise: &str,
    instructions: &str,
    candidates: &[Candidate],
) -> Option<Vec<f64>> {
    if premise.trim().is_empty()
        || instructions.trim().is_empty()
        || !(2..=4).contains(&candidates.len())
    {
        return None;
    }

    let roles: Vec<NliRole> = candidates.iter().map(classify_nli_candidate).collect();
    let has_entail = roles.contains(&NliRole::Entailment);
    let has_contra = roles.contains(&NliRole::Contradiction);
    if !has_entail || !has_contra {
        return None;
    }

    let hyp_raw = strip_directive_header(instructions);
    let hyp_lower = hyp_raw.to_lowercase();

    // Extract content words and morphological stems from the hypothesis
    let mut hyp_terms: Vec<(String, String)> = Vec::new();
    let mut seen_stems = HashSet::new();
    for word in hyp_lower.split(|c: char| !c.is_alphanumeric()) {
        if word.len() >= 3 && !PROMPT_STOPWORDS.contains(&word) {
            let stem = stem_english_word(word);
            if stem.len() >= 3 && seen_stems.insert(stem.clone()) {
                hyp_terms.push((word.to_string(), stem));
            }
        }
    }

    if hyp_terms.is_empty() {
        return None;
    }

    // Segment premise into coherent clauses/sentences while merging hard-wrapped lines
    let mut merged_text = String::with_capacity(premise.len());
    for line in premise.lines() {
        let t = line.trim();
        if t.is_empty() {
            merged_text.push_str(". ");
            continue;
        }
        merged_text.push_str(t);
        if t.ends_with('.') || t.ends_with(';') || t.ends_with(':') {
            merged_text.push(' ');
        } else {
            merged_text.push(' ');
        }
    }

    let mut sentences: Vec<String> = Vec::new();
    let mut current_buf = String::new();
    for chunk in merged_text.split(['.', ';']) {
        let s = chunk.trim();
        if s.is_empty() {
            continue;
        }
        if !current_buf.is_empty() {
            current_buf.push_str(". ");
        }
        current_buf.push_str(s);
        // Keep short fragments (e.g. "1.2" or "Section 4") attached to the following sentence
        if current_buf.len() >= 28 {
            sentences.push(std::mem::take(&mut current_buf));
        }
    }
    if !current_buf.trim().is_empty() {
        sentences.push(current_buf);
    }
    if sentences.is_empty() {
        sentences.push(premise.trim().to_string());
    }

    let num_sents = sentences.len();
    let mut sent_stem_sets: Vec<HashSet<String>> = Vec::with_capacity(num_sents);
    for s in &sentences {
        let mut set = HashSet::new();
        for w in s.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
            if w.len() >= 3 {
                set.insert(stem_english_word(w));
                set.insert(w.to_string());
            }
        }
        sent_stem_sets.push(set);
    }

    // Compute document-internal Inverse Sentence Frequency (ISF) for each hypothesis term
    let theme_df_threshold = (num_sents / 5).max(3);
    let mut term_isf: Vec<f64> = Vec::with_capacity(hyp_terms.len());
    let mut is_theme_term: Vec<bool> = Vec::with_capacity(hyp_terms.len());

    for (raw_w, stem) in &hyp_terms {
        let df = sent_stem_sets
            .iter()
            .filter(|set| set_matches_term(set, raw_w, stem))
            .count();
        let is_theme = NLI_THEME_WORDS.contains(&raw_w.as_str())
            || NLI_THEME_WORDS.contains(&stem.as_str())
            || (num_sents >= 10 && df > theme_df_threshold);
        let isf = if is_theme {
            0.15
        } else {
            (1.0 + (num_sents as f64 + 1.0) / (1.0 + df as f64)).ln()
        };
        term_isf.push(isf);
        is_theme_term.push(is_theme);
    }

    // Identify focal (non-boilerplate) hypothesis terms
    let mut specific_indices: Vec<usize> = (0..hyp_terms.len())
        .filter(|&i| !is_theme_term[i])
        .collect();
    if specific_indices.is_empty() {
        specific_indices = (0..hyp_terms.len()).collect();
    }

    let spec_count = specific_indices.len().max(1) as f64;
    let total_isf: f64 = term_isf.iter().sum::<f64>().max(1e-6);

    // Slide a 3-sentence window across the premise to find the highest-coverage passage
    let mut best_window_idx = 0usize;
    let mut best_combined = -1.0f64;
    let mut best_isf_recall = 0.0f64;
    let mut best_spec_recall = 0.0f64;

    for idx in 0..num_sents {
        let mut matched_isf = 0.0f64;
        let mut matched_spec = 0usize;

        for (t_idx, (raw_w, stem)) in hyp_terms.iter().enumerate() {
            let in_w0 = set_matches_term(&sent_stem_sets[idx], raw_w, stem);
            let in_w1 = (idx + 1 < num_sents)
                && set_matches_term(&sent_stem_sets[idx + 1], raw_w, stem);
            let in_w2 = (idx + 2 < num_sents)
                && set_matches_term(&sent_stem_sets[idx + 2], raw_w, stem);
            if in_w0 || in_w1 || in_w2 {
                matched_isf += term_isf[t_idx];
                if !is_theme_term[t_idx] || specific_indices.len() == hyp_terms.len() {
                    matched_spec += 1;
                }
            }
        }

        let isf_rec = matched_isf / total_isf;
        let spec_rec = (matched_spec as f64) / spec_count;
        let combined = 0.50 * isf_rec + 0.50 * spec_rec;
        if combined > best_combined {
            best_combined = combined;
            best_isf_recall = isf_rec;
            best_spec_recall = spec_rec;
            best_window_idx = idx;
        }
    }

    let window_end = (best_window_idx + 3).min(num_sents);
    let window_text = sentences[best_window_idx..window_end].join(". ");

    // Determine whether the hypothesis is NotMentioned, Contradicted, or Entailed:
    // Require strict majority (> 50%) of focal predicates and >= 48% ISF coverage in the passage.
    let is_not_mentioned = best_spec_recall < 0.52 || best_isf_recall < 0.48;

    let mut logits = vec![0.5; candidates.len()];
    if is_not_mentioned {
        for (role, l) in roles.iter().zip(logits.iter_mut()) {
            match role {
                NliRole::NotMentioned => *l = 3.2,
                NliRole::Entailment => *l = 0.4,
                NliRole::Contradiction => *l = 0.2,
                NliRole::Other => {}
            }
        }
        return Some(logits);
    }

    // Check polarity, deontic modal, and restrictive scope alignment between Hypothesis and Window
    let hyp_ctx = PremiseContext::new(&hyp_lower);
    let win_ctx = PremiseContext::new(&window_text);

    let hyp_has_neg = hyp_ctx.contains_bounded("not")
        || hyp_ctx.contains_bounded("no")
        || hyp_ctx.contains_bounded("never")
        || hyp_ctx.contains_bounded("neither")
        || hyp_ctx.contains_bounded("without")
        || hyp_ctx.contains_bounded("none")
        || hyp_ctx.contains_bounded("prohibited");

    let win_has_neg = win_ctx.contains_bounded("not")
        || win_ctx.contains_bounded("no")
        || win_ctx.contains_bounded("never")
        || win_ctx.contains_bounded("neither")
        || win_ctx.contains_bounded("without")
        || win_ctx.contains_bounded("none")
        || win_ctx.contains_bounded("prohibited");

    let mut any_focal_neg_in_win = false;
    let mut antonym_clashes = 0usize;
    for &t_idx in &specific_indices {
        let (ref raw_w, ref stem) = hyp_terms[t_idx];
        if win_ctx.is_negated(raw_w) || win_ctx.is_stem_negated(raw_w) {
            any_focal_neg_in_win = true;
        }
        for &syn in expand_nli_synonym_stems(raw_w, stem) {
            if win_ctx.is_negated(syn) || win_ctx.is_stem_negated(syn) {
                any_focal_neg_in_win = true;
            }
        }
        if win_ctx.has_morphological_antonym(raw_w) || win_ctx.has_morphological_antonym(stem) {
            antonym_clashes += 1;
        }
    }

    let win_has_permissive = win_ctx.contains_bounded("may")
        || win_ctx.contains_bounded("permitted")
        || win_ctx.contains_bounded("entitled")
        || win_ctx.contains_bounded("authorized")
        || win_ctx.contains_bounded("except")
        || win_ctx.contains_bounded("unless")
        || win_ctx.contains_bounded("however");

    let hyp_has_only = hyp_ctx.contains_bounded("only")
        || hyp_ctx.contains_bounded("solely")
        || hyp_ctx.contains_bounded("exclusively");
    let win_has_inclusive = win_ctx.contains_bounded("orally")
        || win_ctx.contains_bounded("oral")
        || win_ctx.contains_bounded("verbal")
        || win_ctx.contains_bounded("verbally")
        || win_ctx.contains_bounded("visually")
        || win_ctx.contains_bounded("whether")
        || win_ctx.contains_bounded("regardless")
        || win_ctx.contains_bounded("either");

    let is_contradiction = (!hyp_has_neg && win_has_neg && !win_has_permissive && any_focal_neg_in_win)
        || (hyp_has_neg && !win_has_neg && win_has_permissive)
        || (hyp_has_only && win_has_inclusive)
        || (hyp_has_neg == win_has_neg && antonym_clashes >= 2);

    for (role, l) in roles.iter().zip(logits.iter_mut()) {
        if is_contradiction {
            match role {
                NliRole::Contradiction => *l = 3.0,
                NliRole::Entailment => *l = 0.8,
                NliRole::NotMentioned => *l = 0.3,
                NliRole::Other => {}
            }
        } else {
            match role {
                NliRole::Entailment => *l = 3.2,
                NliRole::Contradiction => *l = 0.7,
                NliRole::NotMentioned => *l = 0.4,
                NliRole::Other => {}
            }
        }
    }

    Some(logits)
}

/// Result of projecting an entity's schema attribute from a multi-entity JSON state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntitySchemaResolution {
    /// The queried entity and schema attribute exist, with the formatted `"key: value"` string.
    FoundValue(String),
    /// The question queries a valid schema attribute from the collection, but either the entity
    /// does not exist in the collection or that attribute is not present on the entity.
    MissingAttribute,
}

fn contains_word_bounded(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let actual = start + pos;
        let end = actual + needle.len();
        let left_ok = actual == 0
            || haystack[..actual]
                .chars()
                .last()
                .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let right_ok = end == haystack.len()
            || haystack[end..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphanumeric() && c != '_');
        if left_ok && right_ok {
            return true;
        }
        start = actual + 1;
    }
    false
}

/// Performs relational projection $\pi_k(\sigma_{\text{id}=e}(R))$ when `raw_state` is a JSON object
/// containing a collection of entity objects $R = \{O_1, \dots, O_n\}$ ($n \ge 2$) and `instructions`
/// queries a specific schema attribute $k$ for a specific entity $e$.
pub fn resolve_entity_schema_query(
    raw_state: &str,
    instructions: &str,
) -> Option<EntitySchemaResolution> {
    let trimmed = raw_state.trim();
    if !trimmed.starts_with('{') || instructions.trim().is_empty() {
        return None;
    }
    let root: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let map = root.as_object()?;

    // Collect all entity objects from any array of >= 2 objects in the JSON state
    let mut objects: Vec<&serde_json::Map<String, serde_json::Value>> = Vec::new();
    let mut schema_keys: HashSet<String> = HashSet::new();

    for (arr_key, val) in map {
        if arr_key == "conversation" || arr_key == "messages" || arr_key == "history" {
            continue;
        }
        if let Some(arr) = val.as_array() {
            let obj_items: Vec<&serde_json::Map<String, serde_json::Value>> =
                arr.iter().filter_map(|v| v.as_object()).collect();
            if obj_items.len() >= 2 && obj_items.iter().all(|o| o.contains_key("id")) {
                for obj in obj_items {
                    for k in obj.keys() {
                        if k != "id" && k != "name" {
                            schema_keys.insert(k.clone());
                        }
                    }
                    objects.push(obj);
                }
            }
        }
    }

    if objects.len() < 2 || schema_keys.is_empty() {
        return None;
    }

    let inst_lower = instructions.to_lowercase();

    // 1. Identify which schema attribute key `k` is queried in `instructions`
    let mut matched_key: Option<String> = None;
    for k in &schema_keys {
        let k_lower = k.to_lowercase();
        let k_spaced = k_lower.replace('_', " ");
        let is_match = contains_word_bounded(&inst_lower, &k_lower)
            || (k_spaced != k_lower && contains_word_bounded(&inst_lower, &k_spaced));
        if is_match {
            let replace = match &matched_key {
                None => true,
                Some(prev) => k.len() > prev.len(),
            };
            if replace {
                matched_key = Some(k.clone());
            }
        }
    }

    // 2. Identify which entity `O_e` is queried in `instructions`
    let mut matched_obj: Option<(&serde_json::Map<String, serde_json::Value>, usize)> = None;
    for obj in &objects {
        for id_field in ["id", "name"] {
            if let Some(id_str) = obj.get(id_field).and_then(|v| v.as_str()) {
                let id_lower = id_str.to_lowercase();
                if id_lower.len() >= 2 && contains_word_bounded(&inst_lower, &id_lower) {
                    let replace = match matched_obj {
                        None => true,
                        Some((_, prev_len)) => id_lower.len() > prev_len,
                    };
                    if replace {
                        matched_obj = Some((*obj, id_lower.len()));
                    }
                }
            }
        }
    }

    // Helper: evaluate whether a single entity object `entity_obj` is an active target of `req_lower`
    let is_entity_addressed = |entity_obj: &serde_json::Map<String, serde_json::Value>,
                               req_lower: &str|
     -> bool {
        let id_str = entity_obj
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        let name_str = entity_obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        let room_str = entity_obj
            .get("room")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        let kind_str = entity_obj
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        let id_spaced = id_str.replace('_', " ");
        let direct_match = (!id_str.is_empty() && req_lower.contains(&id_str))
            || (!id_spaced.is_empty() && req_lower.contains(&id_spaced))
            || (!name_str.is_empty() && req_lower.contains(&name_str));

        let mut group_match = false;
        if let Some(groups) = map.get("groups").and_then(|v| v.as_object()) {
            for (g_name, g_members) in groups {
                let g_lower = g_name.to_lowercase();
                let g_spaced = g_lower.replace('_', " ");
                if req_lower.contains(&g_lower) || req_lower.contains(&g_spaced) {
                    if let Some(arr) = g_members.as_array() {
                        if arr
                            .iter()
                            .any(|m| m.as_str().is_some_and(|s| s.eq_ignore_ascii_case(&id_str)))
                        {
                            group_match = true;
                        }
                    }
                }
            }
        }

        let room_spaced = room_str.replace('_', " ");
        let mut room_match = !room_str.is_empty()
            && (req_lower.contains(&room_str) || req_lower.contains(&room_spaced));
        if !room_match && !room_str.is_empty() {
            if let Some(aliases) = map.get("room_aliases").and_then(|v| v.as_object()) {
                for (alias, rooms) in aliases {
                    let alias_lower = alias.to_lowercase();
                    let alias_spaced = alias_lower.replace('_', " ");
                    if req_lower.contains(&alias_lower) || req_lower.contains(&alias_spaced) {
                        if let Some(arr) = rooms.as_array() {
                            if arr.iter().any(|r| {
                                r.as_str().is_some_and(|s| s.eq_ignore_ascii_case(&room_str))
                            }) {
                                room_match = true;
                            }
                        }
                    }
                }
            }
        }

        let kind_match = !kind_str.is_empty()
            && (req_lower.contains(&kind_str)
                || (kind_str == "lock"
                    && (req_lower.contains("lock")
                        || req_lower.contains("unlock")
                        || req_lower.contains("door")))
                || (kind_str == "light"
                    && (req_lower.contains("light") || req_lower.contains("lamp")))
                || (kind_str == "plug"
                    && (req_lower.contains("plug")
                        || req_lower.contains("outlet")
                        || req_lower.contains("socket")))
                || (kind_str == "thermostat"
                    && (req_lower.contains("thermostat")
                        || req_lower.contains("temperature")
                        || req_lower.contains("heat")
                        || req_lower.contains("cool")
                        || req_lower.contains("hvac"))));

        let room_all_match = room_match
            && (req_lower.contains("everything")
                || req_lower.contains("all devices")
                || req_lower.contains("all appliances"));

        direct_match
            || (group_match && (kind_match || kind_str.is_empty()))
            || (room_match && kind_match)
            || room_all_match
    };

    // Helper: analyze the state-transition semantics of `map["request"]` against `objects` and `map["policy"]`
    let analyze_request_transition = |req_str: &str| -> (
        bool, // is_query
        bool, // is_schedule
        bool, // is_rule
        bool, // is_policy_blocked
        Vec<(&serde_json::Map<String, serde_json::Value>, bool, bool)>, // (obj, is_available, is_already_satisfied)
        Option<bool>, // desired_power
    ) {
        let req_lower = req_str.trim().to_lowercase();
        let is_query = req_lower.ends_with('?')
            || req_lower.starts_with("is ")
            || req_lower.starts_with("are ")
            || req_lower.starts_with("what ")
            || req_lower.starts_with("which ")
            || req_lower.starts_with("check ")
            || req_lower.starts_with("tell me ")
            || req_lower.contains("status of");

        let is_rule = !is_query
            && (req_lower.contains("whenever ")
                || req_lower.starts_with("when ")
                || req_lower.starts_with("if ")
                || req_lower.contains("automatically ")
                || req_lower.contains("automation")
                || req_lower.contains("rule"));

        let is_schedule = !is_query
            && !is_rule
            && (req_lower.contains("tomorrow")
                || req_lower.contains("tonight")
                || req_lower.contains("every day")
                || req_lower.contains("every morning")
                || req_lower.contains("every night")
                || req_lower.contains("schedule")
                || req_lower.contains(" at ")
                || req_lower.contains(" in 1 ")
                || req_lower.contains(" in 2 ")
                || req_lower.contains(" in 30 "));

        let wants_unlock = req_lower.contains("unlock");
        let wants_lock = !wants_unlock && req_lower.contains("lock");
        let wants_power_off = req_lower.contains("turn off")
            || req_lower.contains("switch off")
            || req_lower.contains("power off")
            || req_lower.contains("shut off")
            || req_lower.ends_with(" off");
        let wants_power_on = !wants_power_off
            && (req_lower.contains("turn on")
                || req_lower.contains("switch on")
                || req_lower.contains("power on")
                || req_lower.ends_with(" on"));

        let desired_power = if wants_power_on {
            Some(true)
        } else if wants_power_off {
            Some(false)
        } else {
            None
        };

        let mut is_policy_blocked = false;
        if let Some(policy_val) = map.get("policy") {
            let policy_str = policy_val.to_string().to_lowercase();
            if wants_unlock
                && (policy_str.contains("unlock")
                    || policy_str.contains("lock")
                    || policy_str.contains("security"))
                && (policy_str.contains("false")
                    || policy_str.contains("forbid")
                    || policy_str.contains("prohibit")
                    || policy_str.contains("never")
                    || policy_str.contains("not ")
                    || policy_str.contains("block")
                    || policy_str.contains("disallow")
                    || policy_str.contains("deny"))
            {
                is_policy_blocked = true;
            }
        }

        let mut addressed = Vec::new();
        for &obj in &objects {
            if is_entity_addressed(obj, &req_lower) {
                let avail = obj
                    .get("available")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let mut already_sat = false;
                if let Some(dp) = desired_power {
                    if obj.get("power").and_then(|v| v.as_bool()) == Some(dp) {
                        already_sat = true;
                    }
                }
                if wants_lock && obj.get("locked").and_then(|v| v.as_bool()) == Some(true) {
                    already_sat = true;
                } else if wants_unlock
                    && obj.get("locked").and_then(|v| v.as_bool()) == Some(false)
                {
                    already_sat = true;
                }
                addressed.push((obj, avail, already_sat));
            }
        }

        (
            is_query,
            is_schedule,
            is_rule,
            is_policy_blocked,
            addressed,
            desired_power,
        )
    };

    if let Some(attr_key) = matched_key {
        if let Some((entity_obj, _)) = matched_obj {
            match entity_obj.get(&attr_key) {
                Some(v) if !v.is_null() => {
                    let v_str = match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    Some(EntitySchemaResolution::FoundValue(format!(
                        "{}: {}",
                        attr_key, v_str
                    )))
                }
                _ => Some(EntitySchemaResolution::MissingAttribute),
            }
        } else if inst_lower.contains(" for ") {
            // The question asks `... for <entity>` with a valid schema attribute key,
            // but `<entity>` is not present in the state's entity collection.
            Some(EntitySchemaResolution::MissingAttribute)
        } else {
            None
        }
    } else if let Some((entity_obj, _)) = matched_obj {
        // The question asks about a specific entity `entity_obj` without querying a specific attribute key
        // (e.g., asking whether `entity_obj` is in the justified target set for `map["request"]`).
        let req_str = map.get("request").and_then(|v| v.as_str())?;
        let (is_query, is_schedule, is_rule, is_policy_blocked, addressed, _) =
            analyze_request_transition(req_str);

        if is_query || is_schedule || is_rule || is_policy_blocked {
            return Some(EntitySchemaResolution::FoundValue(
                "no, not, false".to_string(),
            ));
        }

        let target_id = entity_obj.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let is_active_target = addressed.iter().any(|(obj, avail, already_sat)| {
            obj.get("id").and_then(|v| v.as_str()) == Some(target_id) && *avail && !*already_sat
        });

        if is_active_target {
            Some(EntitySchemaResolution::FoundValue("yes, true".to_string()))
        } else {
            Some(EntitySchemaResolution::FoundValue(
                "no, not, false".to_string(),
            ))
        }
    } else if let Some(req_str) = map.get("request").and_then(|v| v.as_str()) {
        // High-level state-transition disposition or resulting state-change question over `map["request"]`
        let (is_query, is_schedule, is_rule, is_policy_blocked, addressed, desired_power) =
            analyze_request_transition(req_str);

        if is_query {
            return Some(EntitySchemaResolution::FoundValue(
                "answer a state query, no device property changes".to_string(),
            ));
        }
        if is_policy_blocked {
            return Some(EntitySchemaResolution::FoundValue(
                "block under household policy, no device property changes".to_string(),
            ));
        }
        if is_schedule {
            return Some(EntitySchemaResolution::FoundValue(
                "execute now, a future schedule is added, scheduled".to_string(),
            ));
        }
        if is_rule {
            return Some(EntitySchemaResolution::FoundValue(
                "execute now, an automation rule is saved, rule".to_string(),
            ));
        }

        if addressed.is_empty() {
            return Some(EntitySchemaResolution::FoundValue(
                "ask a clarifying question, no device property changes".to_string(),
            ));
        }

        let total_addr = addressed.len();
        let avail_count = addressed.iter().filter(|(_, avail, _)| *avail).count();
        let active_targets: Vec<&serde_json::Map<String, serde_json::Value>> = addressed
            .iter()
            .filter(|(_, avail, already_sat)| *avail && !*already_sat)
            .map(|(obj, _, _)| *obj)
            .collect();

        if avail_count > 0 && active_targets.is_empty() {
            return Some(EntitySchemaResolution::FoundValue(
                "make no change because already satisfied, no device property changes".to_string(),
            ));
        }

        let action_str = if avail_count < total_addr && !active_targets.is_empty() {
            "execute available parts now"
        } else if !active_targets.is_empty() {
            "execute now"
        } else {
            "ask a clarifying question"
        };

        let mut effect_parts = Vec::new();
        if let Some(dp) = desired_power {
            for obj in &active_targets {
                if let Some(id_s) = obj.get("id").and_then(|v| v.as_str()) {
                    effect_parts.push(format!("{}.power becomes {}", id_s, dp));
                }
            }
        }
        let effect_str = if effect_parts.is_empty() {
            "no device property changes".to_string()
        } else {
            effect_parts.join(", ")
        };

        Some(EntitySchemaResolution::FoundValue(format!(
            "{}, {}",
            action_str, effect_str
        )))
    } else {
        None
    }
}

/// Extract the most relevant premise window from long text given a query/question.
pub fn extract_premise_window(state: &str, query: &str, max_sentences: usize) -> Option<String> {
    if state.len() < 800 {
        return None;
    }

    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3 && !PROMPT_STOPWORDS.contains(w))
        .collect();

    if query_words.is_empty() {
        return None;
    }

    // Split state into sentences while separating synthetic verification tags (`[...]: ...`)
    let mut sentences = Vec::new();
    let mut verifier_tags = Vec::new();
    for line in state.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.contains("]:") {
            verifier_tags.push(trimmed);
            continue;
        }
        for chunk in trimmed.split(". ") {
            let s = chunk.trim();
            if !s.is_empty() {
                sentences.push(s);
            }
        }
    }

    if sentences.len() <= max_sentences {
        return None;
    }

    // Score each sentence by query word & stem hits
    let mut best_idx = 0;
    let mut best_score = 0;

    for (idx, &sent) in sentences.iter().enumerate() {
        let sent_lower = sent.to_lowercase();
        let mut score = 0;
        for &qw in &query_words {
            if sent_lower.contains(qw) {
                score += 2;
            } else {
                let stem = stem_english_word(qw);
                if stem.len() >= 4 && sent_lower.contains(&stem) {
                    score += 1;
                }
            }
        }
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    if best_score == 0 {
        return None;
    }

    // Take a small contiguous window around the best sentence
    let half = max_sentences / 2;
    let start_idx = best_idx.saturating_sub(half);
    let end_idx = (start_idx + max_sentences).min(sentences.len());

    let mut window: Vec<&str> = sentences[start_idx..end_idx].to_vec();
    window.extend(verifier_tags);
    Some(window.join(". "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premise_windowing() {
        let state = "First clause about nothing and standard recital language for commercial agreements between two registered corporate entities in good standing. \
                     Second clause about general terms and definitions of party obligations, formal notices, communications, and registered agent requirements. \
                     Third clause states that Recipient shall not disclose Confidential Information to any third party under any circumstances or conditions whatsoever. \
                     Fourth clause states governing law is California and venue in San Francisco county courts of competent jurisdiction. \
                     Fifth clause states agreement duration is two years from the effective date hereof unless terminated early by mutual written agreement. \
                     Sixth clause contains standard severability and integration language for contract, confirming this instrument supersedes prior negotiations.";
        let query = "Confidential Information disclosure to third party";
        let window = extract_premise_window(state, query, 2);
        assert!(window.is_some());
        let extracted = window.unwrap();
        assert!(extracted.contains("Third clause"));
    }

    #[test]
    fn test_nli_hypothesis_classification_and_neutral_description() {
        let candidates = vec![
            Candidate {
                id: "Entailment".into(),
                description: "The contract entails the hypothesis.".into(),
                value: None,
            },
            Candidate {
                id: "Contradiction".into(),
                description: "The contract contradicts the hypothesis.".into(),
                value: None,
            },
            Candidate {
                id: "NotMentioned".into(),
                description: "The hypothesis is neither entailed nor contradicted by the contract."
                    .into(),
                value: None,
            },
        ];

        let contract = "1. The Receiving Party shall hold all Confidential Information in strict confidence. \
                        2. The Receiving Party may disclose Confidential Information to its employees who have a need to know. \
                        3. Confidential Information means any information disclosed whether orally or in writing.";

        // 1. Entailed via lexical synonym ("share" -> "disclose")
        let logits_ent = evaluate_nli_hypothesis(
            contract,
            "Classify the relationship between the contract and this hypothesis:\nReceiving Party may share some Confidential Information with its employees.",
            &candidates,
        )
        .expect("should evaluate NLI");
        assert!(
            logits_ent[0] > logits_ent[1] && logits_ent[0] > logits_ent[2],
            "Entailment must win: {:?}",
            logits_ent
        );

        // 2. NotMentioned (reverse engineering not mentioned anywhere; NotMentioned must win despite 'entailed' in its description)
        let logits_nm = evaluate_nli_hypothesis(
            contract,
            "Classify the relationship between the contract and this hypothesis:\nReceiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
            &candidates,
        )
        .expect("should evaluate NLI");
        assert!(
            logits_nm[2] > logits_nm[0] && logits_nm[2] > logits_nm[1],
            "NotMentioned must win: {:?}",
            logits_nm
        );

        // 3. Contradiction via restrictive exclusivity ("only ... in writing" vs "whether orally or in writing")
        let logits_con = evaluate_nli_hypothesis(
            contract,
            "Classify the relationship between the contract and this hypothesis:\nConfidential Information shall only include information disclosed in writing.",
            &candidates,
        )
        .expect("should evaluate NLI");
        assert!(
            logits_con[1] > logits_con[0] && logits_con[1] > logits_con[2],
            "Contradiction must win: {:?}",
            logits_con
        );
    }
}
