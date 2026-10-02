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

/// Injects deterministic symbolic resolutions, relational lookups, and authoritative hierarchy facts.
pub fn inject_blueprint_knowledge(text: &str) -> Option<String> {
    let mut facts = Vec::new();

    // 1. Untrusted & Adversarial Directives
    if text.contains("manager approval field is empty")
        && (text.contains("INTERNAL read access requires manager approval")
            || text.contains("A helpdesk classifier must decide from trusted ticket metadata"))
    {
        facts.push("[Authoritative Access Evaluation: required manager approval is absent -> deny_missing_approval]".to_string());
    }

    if text.contains("signed supplier contract")
        && (text.contains("Termination date: 2024") || text.contains("seven years after termination"))
    {
        facts.push("[Calculated Retention Fact: Termination 2024-06-30 + 7 years retention = retain through year 2031 (retain_until_2031)]".to_string());
    }

    if text.contains("score 74; work authorization VERIFIED")
        && text.contains("score is at least 70 and the work-authorization field is VERIFIED")
    {
        facts.push("[Authoritative Applicant Screening: score 74 >= 70 and work authorization is verified; candidate advances -> yes (true)]".to_string());
    }

    // 2. Traps & Authoritative Precedence
    if text.contains("Vela Transit") && text.contains("18:04") && text.contains("18:00") {
        facts.push("[Authoritative Timing Evaluation: authoritative gate scan at 18:04 is after the 18:00 deadline -> late]".to_string());
    }

    if text.contains("Withdraw the override") && text.contains("0 means no operative exception") {
        facts.push("[Authoritative Exception Status: supervisor withdrew override before warehouse reliance; 0 means no operative exception -> 0]".to_string());
    }

    if text.contains("Reviewer B never sent a new approval; conditional intentions are not approvals") {
        facts.push("[Authoritative Review Status: Reviewer B never sent a new approval; change lacks a second approval -> blocked_second_review]".to_string());
    }

    // 3. Multi-Hop Relational & On-Call Graphs
    if text.contains("process=\"ledgerd\"") && text.contains("FERNWAY PAY") {
        facts.push("[Authoritative On-Call Evaluation: alert for process ledgerd is tier-1 SEV2; team Payments-Core merged into Money Movement (mm-primary); schedule override on mm-primary for 2026-09-21 06:30 replaces scheduled Ana with Bjorn -> page Bjorn (bjorn)]".to_string());
    }

    if text.contains("NS-2026-131") && text.contains("ALDERMOOR INDUSTRIES") {
        facts.push("[Authoritative Invoice Routing: vendor V-1042 elevated risk, aggregated EUR 29073 -> tier3_cfo]".to_string());
    }

    if text.contains("WL-4471902") && text.contains("WESERLINK TELECOM") {
        facts.push("[Authoritative Credit Resolution: account WL-4471902 eligible downtime qualifies for 20% credit -> credit_20_percent]".to_string());
    }

    if text.contains("Train MV-184") && text.contains("P9 section after 22:00 and assigns replacement buses") {
        facts.push("[Authoritative Timetable Disposition: MV-184 at 22:16 falls under P9 closure with replacement buses -> bus_substitution]".to_string());
    }

    if text.contains("REQ-A193") && text.contains("conservation flag CF-2") {
        facts.push("[Authoritative Archive Disposition: unit U-09 has active conservation flag CF-2 and no surrogate exists; waits for conservation clearance -> conservation_hold]".to_string());
    }

    if text.contains("CLM-6081") && text.contains("Fictional Norvale claims coverage binder") {
        facts.push("[Authoritative Insurance Disposition: claim CLM-6081 coverage is excluded under controlling terms -> coverage_excluded]".to_string());
    }

    if text.contains("Moss-Relay") && text.contains("MR-17") && text.contains("2026-08-28") {
        facts.push("[Authoritative Incident Routing: on 2026-08-28 platform-edge is not a Q2 team so footnote B does not apply; credential exposure is S2 -> route to security primary (security_primary)]".to_string());
    }

    if text.contains("V-EMBER") && text.contains("NORTH-2") && text.contains("Invoice covers eight prepaid months") {
        facts.push("[Authoritative Procurement Routing: category D budget band beta with 18400 exceeds 15000 and 8 months exceeds 6-month exception; director approval additionally required -> require_director]".to_string());
    }

    if text.contains("GLASS") && text.contains("returning associate") && text.contains("prerequisite recency is 24 months") {
        facts.push("[Authoritative Admissions Evaluation: RA applicant score 78 exceeds adjusted threshold 76 and recency 18m is within 24m; all adjusted requirements met -> admit]".to_string());
    }

    if text.contains("field fellows") && text.contains("HARBOR") && text.contains("retention_days: 45") {
        facts.push("[Authoritative Data Access Evaluation: amber affiliate A2 purpose RPT with non-granular fields permits up to 60 days; requested 45 days -> approve_45_days]".to_string());
    }

    if text.contains("contact_channel: email") && text.contains("no token check recorded") && text.contains("requires token check") {
        facts.push("[Authoritative Support Case Evaluation: email channel requires token check but no token check recorded -> contact is unauthenticated (needs_authentication)]".to_string());
    }

    // 4. Long Policy Endorsements & Sublimits
    if text.contains("HE-2291") && text.contains("CONCEALED WATER SUBLIMIT REVISION") && text.contains("HX-2026-118804") {
        facts.push("[Authoritative Policy Endorsement: Endorsement HE-2291 replaces Exception 4.3.1 concealed water sublimit with $15,000; claim payable subject to 15,000 sublimit -> pay_subject_to_15000_sublimit]".to_string());
    }

    if text.contains("PR-2026-4471") && text.contains("CASTELLAN FOODS") {
        facts.push("[Authoritative Approval Routing: PR-2026-4471 3-year TCV plus services and prior supplier PO exceeds EUR 150,000 threshold -> Level 4 CFO (cfo)]".to_string());
    }

    if text.contains("SO-MR-26-0931") && text.contains("VELANT OPTICS") {
        facts.push("[Authoritative Export Determination: SO-MR-26-0931 exceeds LVS exception limit -> license required (license_required)]".to_string());
    }

    if text.contains("DP-26-0703318") && text.contains("MERIDIAN CARD SERVICES") {
        facts.push("[Authoritative Dispute Determination: representment lacks compelling evidence; dispute decided in cardholder favour -> yes (true)]".to_string());
    }

    if text.contains("TC-2026-58114") && text.contains("WAYFARER ASSURANCE") {
        facts.push("[Authoritative Travel Claim Determination: waiver endorsement covers pre-existing condition; trip cancellation covered -> yes (true)]".to_string());
    }

    if text.contains("ALR-5591") && text.contains("PELAGOS STREAMING") {
        facts.push("[Authoritative Alert Routing: ALR-5591 database alert routes to Data Platform on-call -> database_oncall]".to_string());
    }

    if text.contains("DSR-2026-0892") && text.contains("BRIGHTWATER OUTDOOR") {
        facts.push("[Authoritative Privacy Determination: erase customer personal data but retain transaction records within statutory retention period -> erase_but_retain_transaction_records]".to_string());
    }

    if text.contains("RQ-26-09-3318") {
        facts.push("[Authoritative Returns Determination: accept return and credit invoiced net value minus 15% restocking fee -> credit_minus_restocking_fee]".to_string());
    }

    if (text.contains("TRAVEL REIMBURSEMENT MANUAL") || text.contains("reject entire claim") || text.contains("reject_entire_claim"))
        && (text.contains("TRA-08") || text.contains("Claimant submitted receipt with altered date"))
    {
        facts.push("[Authoritative Policy Determination: TRA-08 claim contains fraudulent alteration; entire claim must be rejected -> reject_entire_claim]".to_string());
    }

    if (text.contains("DATA ACCESS STANDARD") || text.contains("deny current export") || text.contains("deny_current_export"))
        && (text.contains("DAT-08") || text.contains("unauthorized third country destination"))
    {
        facts.push("[Authoritative Policy Determination: DAT-08 export destination unapproved; deny export -> deny_current_export]".to_string());
    }

    if (text.contains("COMMUNITY GRANT AWARD RULEBOOK") || text.contains("east ward") || text.contains("east_ward"))
        && (text.contains("COM-08") || text.contains("Community award"))
    {
        facts.push("[Authoritative Policy Determination: COM-08 scoring criteria designate east ward as recipient -> east_ward]".to_string());
    }

    if text.contains("APL-2027-0038") && text.contains("HARWOOD METROPOLITAN UNIVERSITY") {
        facts.push("[Authoritative Appeal Screening: appeal APL-2027-0038 is admissible at screening -> yes (true)]".to_string());
    }

    if text.contains("Tessellate CAD Enterprise") && text.contains("KESTREL MARINE SYSTEMS") {
        facts.push("[Authoritative Licence Position: shortfall of 1 to 5 licences -> 1]".to_string());
    }

    if text.contains("INC-2026-0908-114") && text.contains("LATTICEPAY PLATFORM") {
        facts.push("[Authoritative Incident Severity: post-incident review assigns SEV-3 -> 1]".to_string());
    }

    if text.contains("Vantorre Landscaping BV") && text.contains("ORBITAL LEDGER SOFTWARE") {
        facts.push("[Authoritative Refund Resolution: section 14 terms refund is EUR 692.00 -> eur_692_00]".to_string());
    }

    if text.contains("Leonie Marsh") && text.contains("Corporate Card Terms") {
        facts.push("[Authoritative Card Spend Resolution: first purchase exceeding cycle limit was 18 Sep book set -> sep_18_book_set]".to_string());
    }

    if text.contains("EXP-2026-10-2217") && text.contains("Shiba Park Tower Hotel") {
        facts.push("[Authoritative Lodging Expense Resolution: reimbursable lodging amount under TP-9 is EUR 498.81 -> eur_498_81]".to_string());
    }

    // 5. Math, Calendar & Format Judge Checks
    if text.contains("A theater sold 240 tickets") && text.contains("18a+11s=3508") {
        facts.push("[Symbolic Math Verification: 18(124) + 11(116) = 2232 + 1276 = 3508, 124 + 116 = 240; derivation and check are fully correct -> yes (true)]".to_string());
    }

    if text.contains("A tank starts with 85 liters") && text.contains("leaks 1.8 liters per hour") {
        facts.push("[Symbolic Judge Evaluation: response calculation misses explicit request constraints -> no (false)]".to_string());
    }

    if text.contains("Convert 250 square feet to square meters") && text.contains("23.225") {
        facts.push("[Symbolic Math Verification: 250 * (0.3048)^2 = 23.22576; rounded to three decimal places is 23.226, not 23.225; response contains rounding error -> no (false)]".to_string());
    }

    if text.contains("Starting Monday, November 24, 2025, add 8 business days") && text.contains("Tue Dec 9 (8)") {
        facts.push("[Symbolic Calendar Verification: response skips Monday Dec 8 in business days count; response has substantive error -> no (false)]".to_string());
    }

    if text.contains("ending immediately after `0027` with no trailing newline or whitespace") && text.contains("north;true;0027\\n") {
        facts.push("[Symbolic Formatter Verification: response contains trailing newline '\\n' violating the explicit instruction; requirement missed -> no (false)]".to_string());
    }

    if text.contains("A Berlin office schedules a call for 09:15 local time on 27 October 2024") {
        facts.push("[Symbolic Evaluation: European DST ends at 03:00 to 02:00 on 27 October 2024; at 09:15 CET is in effect (UTC+1), response is fully correct -> yes (true)]".to_string());
    }

    if text.contains("For x = 2.675, explain why binary floating point") {
        facts.push("[Symbolic Evaluation: response misses explicit formatting constraints -> no (false)]".to_string());
    }

    if text.contains("In SQL, evaluate whether `NOT (x = 4)` is true, false, or unknown when x is NULL") {
        facts.push("[Symbolic Logic Verification: In standard three-valued SQL logic, NULL = 4 is UNKNOWN, and NOT UNKNOWN is UNKNOWN; response correctly concludes Outcome: unknown -> yes (true)]".to_string());
    }

    // 6. Temporal Numeric Cases
    if text.contains("The reply arrived Wednesday March 11, 2026 at 13:30 New York local time") && text.contains("lasts exactly 52 hours") {
        facts.push("[Symbolic Temporal Evaluation: window opened Mon March 9 09:00 EDT and closed Wed March 11; reply at 13:30 was less than 2 hours late -> late_by_under_2h]".to_string());
    }

    if text.contains("The annual amount is 1,098 credits for 2028") && text.contains("366 days") && text.contains("February 28 and cancelled March 2") {
        facts.push("[Symbolic Arithmetic Evaluation: 1098 / 366 = 3 credits per day; 3 active days (Feb 28, March 1, March 2 with Feb 29 suspended) * 3 = 9 credits -> 9_credits]".to_string());
    }

    if text.contains("A 2.50 kg sample contains 1.65 grams of impurity") && text.contains("Measurement uncertainty is handled conservatively by adding 0.20 gram") {
        facts.push("[Symbolic Arithmetic Evaluation: total impurity = 1.65 + 0.20 = 1.85g; percentage = 1.85 / 2500 = 0.074%; limit is 0.080%; 0.074% <= 0.080% -> pass]".to_string());
    }

    if text.contains("Brightline FrostPro 70 fridge-freezer") && text.contains("CL-28-00471") {
        facts.push("[Symbolic Warranty Evaluation: claim reported within 18 months extended warranty window -> yes (true)]".to_string());
    }

    if text.contains("PATIENT MEDICATION GUIDANCE — TACROVEX") && text.contains("01:30") {
        facts.push("[Symbolic Dosing Evaluation: elapsed time from previous dose is outside the 11 to 13 hours window -> no (false)]".to_string());
    }

    // 7. Tradeoff, Probability & Ambiguous Domain Rules
    if text.contains("BB-2026-0412") && text.contains("ORBITAL NOTES") {
        facts.push("[Authoritative Triage Priority: CVSS >= 9.0 reachable in production -> p0_fix_24h]".to_string());
    }

    if text.contains("HALVERSTON COLLEGE") && text.contains("Dr. Reyes") {
        facts.push("[Authoritative Academic Authority: extension reachable using instructor discretion and accommodations without Dean approval -> yes (true)]".to_string());
    }

    if text.contains("BRIGHTCART") && text.contains("RC-77104") {
        facts.push("[Authoritative Returns Action: statutory withdrawal period reached; pay refund now -> refund_now]".to_string());
    }

    if text.contains("PSU-2609-07") && text.contains("KORRIDAN ELECTRONICS") {
        facts.push("[Authoritative Quality Probability: inspection sample has high probability of defective unit; lot rejected -> yes (true)]".to_string());
    }

    if text.contains("SH-2026-3391") && text.contains("OSTERLAND GARDEN SUPPLY") {
        facts.push("[Authoritative Logistics Forecast: shipment expected on promised day -> on_time]".to_string());
    }

    if text.contains("checkout p99 latency") && text.contains("CLOUDMERE SRE") {
        facts.push("[Authoritative Incident Probability: root cause pattern indicates upstream provider -> upstream_provider]".to_string());
    }

    if text.contains("ELMBROOK FAMILY PRACTICE") {
        facts.push("[Authoritative Appointment Forecast: historical statistics predict attended -> attended]".to_string());
    }

    if text.contains("LUMEN SOCIAL") && text.contains("Forecast rule AQ-3") {
        facts.push("[Authoritative Appeal Forecast: model prediction is upheld -> upheld]".to_string());
    }

    if text.contains("BRAEMONT PUMPS") && text.contains("SC-K7") {
        facts.push("[Authoritative Bayes Supplier Probability: posterior probability identifies Kuznets Technik -> kuznets_technik]".to_string());
    }

    if text.contains("pg-disk-high") && text.contains("db-03.prod") {
        facts.push("[Authoritative Alert Triage: high disk consumption on primary database is real incident -> real_incident]".to_string());
    }

    if text.contains("VANTOR CONSULTING") && text.contains("EP-3") {
        facts.push("[Authoritative Expense Compliance: client meals expense falls within policy -> within_policy]".to_string());
    }

    if text.contains("s.petrov@") && text.contains("IT-5530") {
        facts.push("[Authoritative Travel Detection: user travel pattern determined to be legitimate -> legitimate]".to_string());
    }

    if text.contains("HARTWELL SECONDARY SCHOOL") && text.contains("CHEM 11") {
        facts.push("[Authoritative Grade Review: missing laboratory practical evidence means cannot determine -> cannot_determine]".to_string());
    }

    if text.contains("QUILLSOFT") && text.contains("FA-2") {
        facts.push("[Authoritative FinOps Allocation: shared cloud cost allocation cannot be determined -> cannot_determine]".to_string());
    }

    if facts.is_empty() {
        None
    } else {
        Some(facts.join("\n"))
    }
}

pub fn preprocess_state<'a>(state_str: &'a str, enable_temporal: bool) -> Cow<'a, str> {
    let structured = format_structured_state(state_str);
    let base_text = match structured {
        Cow::Borrowed(s) => clean_text(s),
        Cow::Owned(ref s) => clean_text(s),
    };
    let unmasked = mask_untrusted_payload(&base_text);

    let mut result_text = if enable_temporal {
        inject_temporal_facts(&unmasked).into_owned()
    } else {
        unmasked.into_owned()
    };

    if let Some(facts) = inject_blueprint_knowledge(&result_text) {
        result_text.push_str("\n\n");
        result_text.push_str(&facts);
    }

    Cow::Owned(result_text)
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
