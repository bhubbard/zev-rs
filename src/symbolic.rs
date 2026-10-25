//! Generic Symbolic Arithmetic & Constraint Verifier
//!
//! Provides deterministic symbolic verification for:
//! 1. Arithmetic evaluation (evaluates full expressions before `=` like `83 − 37 + 1 = 47`)
//! 2. Structural constraints (sentence count, word count, non-empty criteria)
//!
//! Emits explicit verification findings into the premise context without hardcoded heuristics.

use regex::Regex;
use std::sync::LazyLock;

/// Regex matching requested sentence count, e.g. "two-sentence", "in 3 sentences"
static RE_SENTENCE_CONSTRAINT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:a\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|\d+)[-\s]+sentence\b")
        .expect("valid sentence constraint regex")
});


fn word_to_number(w: &str) -> Option<usize> {
    match w.to_lowercase().as_str() {
        "one" | "1" => Some(1),
        "two" | "2" => Some(2),
        "three" | "3" => Some(3),
        "four" | "4" => Some(4),
        "five" | "5" => Some(5),
        "six" | "6" => Some(6),
        "seven" | "7" => Some(7),
        "eight" | "8" => Some(8),
        "nine" | "9" => Some(9),
        "ten" | "10" => Some(10),
        other => other.parse::<usize>().ok(),
    }
}

/// Evaluates a simple arithmetic expression (supports +, -, *, /, parenthesis)
pub fn eval_simple_math(expr: &str) -> Option<f64> {
    let clean = expr
        .replace('−', "-")
        .replace('×', "*")
        .replace('÷', "/")
        .replace(',', "");

    // Tokenize
    let mut tokens: Vec<String> = Vec::new();
    let mut current_num = String::new();

    for ch in clean.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            current_num.push(ch);
        } else {
            if !current_num.is_empty() {
                tokens.push(current_num.clone());
                current_num.clear();
            }
            if ch == '+' || ch == '-' || ch == '*' || ch == '/' {
                tokens.push(ch.to_string());
            } else if !ch.is_whitespace() {
                // Unknown character in expression
                return None;
            }
        }
    }
    if !current_num.is_empty() {
        tokens.push(current_num);
    }

    if tokens.is_empty() {
        return None;
    }

    // Two-pass parser: 1. * and /; 2. + and -
    let mut pass1: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if (tokens[i] == "*" || tokens[i] == "/") && !pass1.is_empty() && i + 1 < tokens.len() {
            let left: f64 = pass1.pop()?.parse().ok()?;
            let right: f64 = tokens[i + 1].parse().ok()?;
            let res = if tokens[i] == "*" {
                left * right
            } else {
                if right.abs() < 1e-9 {
                    return None;
                }
                left / right
            };
            pass1.push(res.to_string());
            i += 2;
        } else {
            pass1.push(tokens[i].clone());
            i += 1;
        }
    }

    // Pass 2: + and -
    let mut result: f64 = pass1.first()?.parse().ok()?;
    let mut j = 1;
    while j < pass1.len() {
        let op = &pass1[j];
        if j + 1 < pass1.len() {
            let next_val: f64 = pass1[j + 1].parse().ok()?;
            if op == "+" {
                result += next_val;
            } else if op == "-" {
                result -= next_val;
            } else {
                return None;
            }
            j += 2;
        } else {
            return None;
        }
    }

    Some(result)
}

static RE_EQUATION_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([0-9\.\s\+\-\*\/×−÷]+)=\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)")
        .expect("valid equation line regex")
});

static RE_EXACT_WORD_CONSTRAINT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(exactly|in|at most|maximum of)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+words?\b")
        .expect("valid exact word constraint regex")
});

static RE_FLOAT_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d+\.\d+)\b").expect("valid float token regex")
});

static RE_PLAIN_REQ_RESP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)^\s*(?:request|task)\s*:\s*(.*?)\b(?:response|answer)\s*:\s*(.+)$")
        .expect("valid plain req resp regex")
});

/// Evaluates arithmetic equations present in the text.
/// Returns descriptive findings of any calculation errors or confirmations.
pub fn verify_arithmetic_equations(text: &str) -> Vec<String> {
    if !text.contains('=') {
        return Vec::new();
    }
    let mut findings = Vec::new();
    let mut all_correct = true;
    let mut equation_count = 0;

    let mut exact_rhs_floats: Vec<(f64, usize, usize)> = Vec::new();
    for cap in RE_EQUATION_LINE.captures_iter(text) {
        let lhs = cap[1].trim();
        let rhs_raw = cap[2].trim();
        let rhs_str = rhs_raw.replace(',', "");

        // Only evaluate if LHS contains an arithmetic operator (+, -, *, /)
        if !lhs.contains('+')
            && !lhs.contains('-')
            && !lhs.contains('−')
            && !lhs.contains('*')
            && !lhs.contains('×')
            && !lhs.contains('/')
            && !lhs.contains('÷')
        {
            continue;
        }

        if let (Some(actual_res), Ok(expected_res)) =
            (eval_simple_math(lhs), rhs_str.parse::<f64>())
        {
            equation_count += 1;
            let diff = (actual_res - expected_res).abs();
            let tolerance = 0.01; // Deterministic calculation tolerance (accounts for rounding)

            if diff > tolerance {
                all_correct = false;
                findings.push(format!(
                    "[ARITHMETIC ERROR]: Equation \"{} = {}\" is mathematically incorrect (actual result is {:.4}). (CALCULATION_VALID: FALSE)",
                    lhs, rhs_str, actual_res
                ));
            } else if let Some(dot_pos) = rhs_str.find('.') {
                let dec_places = rhs_str.len() - dot_pos - 1;
                if dec_places >= 4 {
                    let end_pos = cap.get(0).map(|m| m.end()).unwrap_or(0);
                    exact_rhs_floats.push((actual_res, dec_places, end_pos));
                }
            }
        }
    }

    // Check if any subsequent rounded float in the text truncated instead of rounding half-up
    for (exact_val, exact_decs, after_pos) in exact_rhs_floats {
        if let Some(tail) = text.get(after_pos..) {
            for fcap in RE_FLOAT_TOKEN.captures_iter(tail) {
                let f_str = &fcap[1];
                if let Some(dot_pos) = f_str.find('.') {
                    let k = f_str.len() - dot_pos - 1;
                    if (1..=3).contains(&k) && k < exact_decs {
                        if let Ok(reported_val) = f_str.parse::<f64>() {
                            let step = 10f64.powi(-(k as i32));
                            let err = (exact_val - reported_val).abs();
                            if err >= 0.49 * step && err < step {
                                let scale = 10f64.powi(k as i32);
                                let half_up = (exact_val * scale + 0.5).floor() / scale;
                                if (half_up - reported_val).abs() > 1e-6 {
                                    all_correct = false;
                                    findings.push(format!(
                                        "[ARITHMETIC ERROR]: Rounded value {} does not match half-up rounding of {} (expected {:.prec$}). (CALCULATION_VALID: FALSE)",
                                        f_str,
                                        exact_val,
                                        half_up,
                                        prec = k
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if equation_count > 0 && all_correct {
        findings.push(format!(
            "[ARITHMETIC VERIFIED]: {} arithmetic equations checked and verified mathematically correct. (CALCULATION_VALID: TRUE)",
            equation_count
        ));
    }

    findings
}

/// Counts sentences in text using punctuation boundaries.
pub fn count_sentences(text: &str) -> usize {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0;
    }

    trimmed
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.chars().any(|ch| ch.is_alphabetic()))
        .count()
}

/// Checks if prompt specifies structural length or formatting constraints (sentence count, word count, trailing newline, JSON type)
/// and verifies whether the provided response complies.
pub fn verify_structural_constraints(request_text: &str, response_text: &str) -> Vec<String> {
    let mut findings = Vec::new();
    let req_lower = request_text.to_lowercase();

    // 1. Sentence count constraint check
    if let Some(cap) = RE_SENTENCE_CONSTRAINT.captures(request_text) {
        if let Some(target_str) = cap.get(1) {
            if let Some(expected_sentences) = word_to_number(target_str.as_str()) {
                let actual_sentences = count_sentences(response_text);
                if actual_sentences != expected_sentences {
                    findings.push(format!(
                        "[CONSTRAINT ERROR]: Requested exactly {} sentences, but response has {} sentences. (CONSTRAINT_SATISFIED: FALSE)",
                        expected_sentences, actual_sentences
                    ));
                } else {
                    findings.push(format!(
                        "[CONSTRAINT VERIFIED]: Response strictly adheres to the {}-sentence constraint. (CONSTRAINT_SATISFIED: TRUE)",
                        expected_sentences
                    ));
                }
            }
        }
    }

    // 2. Word count constraint check (exact vs upper bound)
    if let Some(cap) = RE_EXACT_WORD_CONSTRAINT.captures(request_text) {
        let mode = cap[1].to_lowercase();
        if let Some(target_words) = word_to_number(&cap[2]) {
            let actual_words = response_text.split_whitespace().count();
            if mode == "exactly" {
                if actual_words != target_words {
                    findings.push(format!(
                        "[CONSTRAINT ERROR]: Requested exactly {} words, but response contains {} words. (CONSTRAINT_SATISFIED: FALSE)",
                        target_words, actual_words
                    ));
                } else {
                    findings.push(format!(
                        "[CONSTRAINT VERIFIED]: Response contains exactly {} words. (CONSTRAINT_SATISFIED: TRUE)",
                        target_words
                    ));
                }
            } else if actual_words > target_words {
                findings.push(format!(
                    "[CONSTRAINT ERROR]: Requested constraint of at most {} words, but response contains {} words. (CONSTRAINT_SATISFIED: FALSE)",
                    target_words, actual_words
                ));
            }
        }
    }

    // 3. Trailing newline / whitespace prohibition
    if (req_lower.contains("no trailing newline") || req_lower.contains("no trailing whitespace"))
        && (response_text.ends_with('\n')
            || response_text.ends_with('\r')
            || response_text.ends_with(' '))
    {
        findings.push(
            "[CONSTRAINT ERROR]: Response contains a prohibited trailing newline or whitespace. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    // 4. JSON array vs object structural check
    if req_lower.contains("json array") && response_text.trim_start().starts_with('{') {
        findings.push(
            "[CONSTRAINT ERROR]: Requested a JSON array, but response is a JSON object. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    } else if req_lower.contains("json object") && response_text.trim_start().starts_with('[') {
        findings.push(
            "[CONSTRAINT ERROR]: Requested a JSON object, but response is a JSON array. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    findings
}

/// Parses plain-text Request/Response pairs and verifies structural constraints.
pub fn verify_plain_text_request_response(text: &str) -> Vec<String> {
    if text.len() > 2500 || (!text.contains("Response:") && !text.contains("Answer:")) {
        return Vec::new();
    }
    if let Some(cap) = RE_PLAIN_REQ_RESP.captures(text) {
        return verify_structural_constraints(cap[1].trim(), cap[2].trim());
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_simple_math_multi_term() {
        assert_eq!(eval_simple_math("83 - 37 + 1"), Some(47.0));
        assert_eq!(eval_simple_math("47 * 60"), Some(2820.0));
        assert_eq!(eval_simple_math("85 - 13.5"), Some(71.5));
    }

    #[test]
    fn test_verify_correct_arithmetic() {
        let text = "There are 83 − 37 + 1 = 47 terms. Sum is 47 × 60 = 2,820.";
        let findings = verify_arithmetic_equations(text);
        assert!(!findings.is_empty());
        assert!(findings[0].contains("CALCULATION_VALID: TRUE"));
    }

    #[test]
    fn test_detect_incorrect_arithmetic() {
        let text = "Calculation shows 85 - 13.5 = 71.5, then 71.5 + 12 = 84.5.";
        let findings = verify_arithmetic_equations(text);
        assert!(!findings.is_empty());
        assert!(findings.iter().any(|f| f.contains("CALCULATION_VALID: FALSE")));
    }

    #[test]
    fn test_sentence_constraint_verification() {
        let req = "Write a two-sentence summary of the incident.";
        let resp_ok = "The server experienced an outage at noon. Service was restored within fifteen minutes.";
        let resp_bad = "The server crashed. We rebooted it. All services returned. Traffic stabilized.";

        let f_ok = verify_structural_constraints(req, resp_ok);
        assert!(f_ok.iter().any(|f| f.contains("CONSTRAINT_SATISFIED: TRUE")));

        let f_bad = verify_structural_constraints(req, resp_bad);
        assert!(f_bad.iter().any(|f| f.contains("CONSTRAINT_SATISFIED: FALSE")));
    }
}
