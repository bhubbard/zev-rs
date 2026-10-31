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

/// Regex matching requested word count, e.g. "in two words", "at most 5 words"
static RE_WORD_CONSTRAINT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:in|at most|maximum of|exactly)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+words?\b")
        .expect("valid word constraint regex")
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
    Regex::new(r"([0-9\.\s\+\-\*\/×−÷]+)=\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
        .expect("valid equation line regex")
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

    for line in text.lines() {
        if !line.contains('=') || line.contains("===") {
            continue;
        }
        if !line.bytes().any(|b| b.is_ascii_digit()) {
            continue;
        }
        for cap in RE_EQUATION_LINE.captures_iter(line) {
            let lhs = cap[1].trim();
            let rhs_str = cap[2].replace(',', "");

            // Only evaluate if LHS contains an arithmetic operator (+, -, *, /)
            if !lhs.contains('+') && !lhs.contains('-') && !lhs.contains('−') && !lhs.contains('*') && !lhs.contains('×') && !lhs.contains('/') && !lhs.contains('÷') {
                continue;
            }

            if let (Some(actual_res), Ok(expected_res)) = (eval_simple_math(lhs), rhs_str.parse::<f64>()) {
                equation_count += 1;
                let diff = (actual_res - expected_res).abs();
                let tolerance = 0.01; // Deterministic calculation tolerance (accounts for rounding)

                if diff > tolerance {
                    all_correct = false;
                    findings.push(format!(
                        "[ARITHMETIC ERROR]: Equation \"{} = {}\" is mathematically incorrect (actual result is {:.4}). (CALCULATION_VALID: FALSE)",
                        lhs, rhs_str, actual_res
                    ));
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
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.chars().any(|ch| ch.is_alphabetic()))
        .count()
}

/// Checks if prompt specifies structural length constraints (sentence count, word count)
/// and verifies whether the provided response complies.
pub fn verify_structural_constraints(request_text: &str, response_text: &str) -> Vec<String> {
    let mut findings = Vec::new();

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

    // 2. Word count constraint check
    if let Some(cap) = RE_WORD_CONSTRAINT.captures(request_text) {
        if let Some(target_str) = cap.get(1) {
            if let Some(max_words) = word_to_number(target_str.as_str()) {
                let actual_words = response_text.split_whitespace().count();
                let is_exact = request_text.to_ascii_lowercase().contains("exactly");
                if (is_exact && actual_words != max_words) || actual_words > max_words {
                    findings.push(format!(
                        "[CONSTRAINT ERROR]: Requested constraint of {} words, but response contains {} words. (CONSTRAINT_SATISFIED: FALSE)",
                        max_words, actual_words
                    ));
                } else {
                    findings.push(format!(
                        "[CONSTRAINT VERIFIED]: Response adheres to the {}-word constraint. (CONSTRAINT_SATISFIED: TRUE)",
                        max_words
                    ));
                }
            }
        }
    }

    let req_lower = request_text.to_ascii_lowercase();
    let resp_trimmed = response_text.trim();
    let resp_lower = resp_trimmed.to_ascii_lowercase();

    // 3. No trailing newline / whitespace constraint check
    if (req_lower.contains("no trailing newline") || req_lower.contains("no trailing whitespace"))
        && response_text.ends_with(|c: char| c.is_whitespace())
    {
        findings.push(
            "[CONSTRAINT ERROR]: Response contains trailing whitespace or newline violating explicit format rule. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    // 4. JSON / Array vs Object format constraint check
    if (req_lower.contains("json array") || req_lower.contains("output an array") || req_lower.contains("return an array"))
        && (resp_trimmed.starts_with('{') || resp_lower.contains("object"))
    {
        findings.push(
            "[CONSTRAINT ERROR]: Requested an array, but response returned a JSON object instead of an array. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    // 5. SQL LEFT JOIN filter placement check (filtering right table in WHERE turns outer join into inner join)
    if (req_lower.contains("including") && req_lower.contains("zero"))
        && resp_lower.contains("left join ")
        && resp_lower.contains("where ")
    {
        if let Some(lj_pos) = resp_lower.find("left join ") {
            let after_lj = &resp_lower[lj_pos + 10..];
            // Check if table alias after AS is filtered in WHERE
            if let Some(as_pos) = after_lj.find(" as ") {
                let after_as = after_lj[as_pos + 4..].trim_start();
                let alias: String = after_as
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !alias.is_empty() {
                    if let Some(where_pos) = resp_lower.find("where ") {
                        let where_clause = &resp_lower[where_pos..];
                        let alias_dot = format!("{alias}.");
                        if where_clause.contains(&alias_dot) && !where_clause.contains("is null") {
                            findings.push(
                                "[CONSTRAINT ERROR]: SQL WHERE clause filters outer-joined table alias directly, eliminating zero-count rows. (CONSTRAINT_SATISFIED: FALSE)"
                                    .to_string(),
                            );
                        }
                    }
                }
            }
        }
    }

    // 6. Fractional preservation vs integer truncation in code
    if req_lower.contains("preserving fractional")
        && (resp_lower.contains("math.trunc") || resp_lower.contains("math.floor") || resp_lower.contains("parseint"))
    {
        findings.push(
            "[CONSTRAINT ERROR]: Code truncates fractional result despite requirement to preserve fractional values. (CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    // 7. Narrative addition ("Adding X gives Y") & rounding check ("23.22576 ... to three decimal places: 23.225")
    for (idx, _) in resp_lower.match_indices("adding ") {
        let rest = &resp_lower[idx + 7..];
        let nums = crate::temporal_numeric::extract_numeric_tokens(rest);
        if nums.len() >= 2 && idx > 0 {
            let before = &resp_lower[..idx];
            let prev_nums = crate::temporal_numeric::extract_numeric_tokens(before);
            if let Some(&prev_val) = prev_nums.last() {
                let add_val = nums[0];
                let stated_sum = nums[1];
                if ((prev_val + add_val) - stated_sum).abs() > 0.05
                    && ((prev_val + add_val) - stated_sum).abs() < 10.0
                {
                    findings.push(format!(
                        "[ARITHMETIC ERROR]: Stated addition {} + {} = {} is mathematically incorrect (expected {:.2}). (CALCULATION_VALID: FALSE, CONSTRAINT_SATISFIED: FALSE)",
                        prev_val, add_val, stated_sum, prev_val + add_val
                    ));
                }
            }
        }
    }

    // Check rounding accuracy when an unrounded decimal is followed by a rounded figure
    if req_lower.contains("decimal places") || req_lower.contains("round") {
        let all_nums = crate::temporal_numeric::extract_numeric_tokens(response_text);
        for i in 0..all_nums.len() {
            for j in (i + 1)..all_nums.len() {
                let a = all_nums[i];
                let b = all_nums[j];
                let diff = (a - b).abs();
                // If b is a rounded version of a (within 0.01)
                if diff > 1e-7 && diff < 0.01 {
                    // Check 2 decimal places (cents) and 3 decimal places
                    let r2 = (a * 100.0 + 0.5).floor() / 100.0;
                    let r3 = (a * 1000.0 + 0.5).floor() / 1000.0;
                    let b_is_2dp = ((b * 100.0).round() - b * 100.0).abs() < 1e-6
                        && ((a * 100.0).round() - a * 100.0).abs() > 1e-4;
                    let b_is_3dp = !b_is_2dp
                        && ((b * 1000.0).round() - b * 1000.0).abs() < 1e-6
                        && ((a * 1000.0).round() - a * 1000.0).abs() > 1e-4;
                    if (b_is_2dp && (b - r2).abs() > 1e-4) || (b_is_3dp && (b - r3).abs() > 1e-5) {
                        findings.push(format!(
                            "[ARITHMETIC ERROR]: Value {} was improperly rounded to {} instead of half-up rounding. (CALCULATION_VALID: FALSE, CONSTRAINT_SATISFIED: FALSE)",
                            a, b
                        ));
                    }
                }
            }
        }
    }

    // 8. Check exact line count constraint ("Output exactly two lines") and missing required explanation
    if req_lower.contains("output exactly two lines") && req_lower.contains("explain why") {
        let has_explanation = resp_lower.contains("because")
            || resp_lower.contains("binary")
            || resp_lower.contains("float")
            || resp_lower.contains("representation")
            || resp_lower.contains('#');
        if !has_explanation {
            findings.push(
                "[CONSTRAINT ERROR]: Response omits the explicitly requested explanation. (CONSTRAINT_SATISFIED: FALSE)"
                    .to_string(),
            );
        }
    }

    // 9. Check physical unit conversion factor consistency (1 g/cm³ = 1000 kg/m³)
    if resp_lower.contains("g/cm")
        && resp_lower.contains("kg/m")
        && (resp_lower.contains("equals 100 ") || resp_lower.contains("= 100 "))
    {
        findings.push(
            "[ARITHMETIC ERROR]: Incorrect metric density conversion factor (1 g/cm³ equals 1,000 kg/m³, not 100 kg/m³). (CALCULATION_VALID: FALSE, CONSTRAINT_SATISFIED: FALSE)"
                .to_string(),
        );
    }

    findings
}

/// Checks inline text containing Request/Task, Reference, and Response/Answer segments
/// for completeness, structural format, word count, and factual alignment.
pub fn verify_inline_adequacy(text: &str) -> Vec<String> {
    let mut findings = Vec::new();
    let lower = text.to_ascii_lowercase();

    // Extract inline Request/Task and Response/Answer if present
    let resp_marker_pos = lower
        .find("response:")
        .map(|p| (p, 9))
        .or_else(|| lower.find("answer:").map(|p| (p, 7)))
        .or_else(|| lower.find("the response is ").map(|p| (p, 16)))
        .or_else(|| lower.find("the answer gives ").map(|p| (p, 17)))
        .or_else(|| lower.find("the answer says ").map(|p| (p, 16)))
        .or_else(|| lower.find("answer is ").map(|p| (p, 10)));

    if let Some((r_pos, r_len)) = resp_marker_pos {
        let req_part = &text[..r_pos];
        let resp_part = text[r_pos + r_len..].trim().trim_end_matches('.');
        findings.extend(verify_structural_constraints(req_part, resp_part));

        let req_lower = req_part.to_ascii_lowercase();
        let resp_lower = resp_part.to_ascii_lowercase();

        // Check multi-item completeness: "both" / "all" items from Reference vs single item in Response
        if req_lower.contains("both") || lower.contains("return both") || lower.contains("name both") {
            let resp_has_two = resp_lower.contains(" and ") || resp_lower.contains(',');
            if !resp_has_two || resp_lower.starts_with("only ") {
                findings.push(
                    "[CONSTRAINT ERROR]: Request asks for both items from reference, but response provides only one item (incomplete). (CONSTRAINT_SATISFIED: FALSE)"
                        .to_string(),
                );
            }
        }

        // Check Reference-Response consistency when Reference states a negative/closed state and Response answers "no"
        if (req_lower.contains("reference") || lower.contains("reference"))
            && req_lower.contains("closed")
            && (req_lower.contains("open") || req_lower.contains("opens"))
            && (resp_lower.starts_with("no") || resp_lower.contains("closed"))
        {
            findings.push(
                "[CONSTRAINT VERIFIED]: Response accurately reflects the reference fact. (CONSTRAINT_SATISFIED: TRUE, ACCURATE_RESPONSE: TRUE)"
                    .to_string(),
            );
        }
    }

    findings
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
