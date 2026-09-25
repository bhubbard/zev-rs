use crate::calibration::scaled_softmax;
use crate::error::{Result, ZevError};
use crate::order_invariant::{compute_order_invariant_logits_with_context, PremiseContext};
use crate::preprocessor::preprocess_state;
use crate::types::Candidate;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Instant;

/// Tev1 Request format (compatible with Together AI's Tev1-4B-experimental decision model)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tev1Request {
    pub state: String,
    pub question: String,
    pub options: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
}

/// Tev1 Response format returning decision letter, choice, and calibrated logprobs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tev1Response {
    pub answer: String,
    pub choice: String,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub logprobs: BTreeMap<String, f64>,
    pub execution_micros: f64,
}

impl Tev1Request {
    /// Parses a raw Tev1 prompt text into a structured Tev1Request.
    /// Supports both tabular formats (`State   ...\nQuestion   ...`) and colon-delimited formats (`State: ...\nQuestion: ...`).
    pub fn parse_prompt(prompt: &str) -> Result<Self> {
        let mut state = String::new();
        let mut question = String::new();
        let mut options = Vec::new();

        let mut current_section = "";

        for raw_line in prompt.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            let lower = line.to_lowercase();
            if lower.starts_with("state")
                && (line.contains(':') || line.contains("   ") || line.contains('\t'))
            {
                current_section = "state";
                let content = extract_header_content(line, "state");
                state.push_str(content);
            } else if lower.starts_with("question")
                && (line.contains(':') || line.contains("   ") || line.contains('\t'))
            {
                current_section = "question";
                let content = extract_header_content(line, "question");
                question.push_str(content);
            } else if lower.starts_with("options") {
                current_section = "options";
                let content = extract_header_content(line, "options");
                if !content.is_empty() {
                    parse_option_line(content, &mut options);
                }
            } else if lower.starts_with("answer") {
                break; // Stop parsing before answer token
            } else {
                match current_section {
                    "state" => {
                        if !state.is_empty() {
                            state.push(' ');
                        }
                        state.push_str(line);
                    }
                    "question" => {
                        if !question.is_empty() {
                            question.push(' ');
                        }
                        question.push_str(line);
                    }
                    "options" => {
                        parse_option_line(line, &mut options);
                    }
                    _ => {}
                }
            }
        }

        if state.is_empty() {
            return Err(ZevError::InvalidRequest(
                "Missing 'State' in Tev1 prompt".into(),
            ));
        }
        if question.is_empty() {
            return Err(ZevError::InvalidRequest(
                "Missing 'Question' in Tev1 prompt".into(),
            ));
        }
        if options.len() < 2 {
            return Err(ZevError::InvalidRequest(
                "Tev1 requires at least 2 options".into(),
            ));
        }
        if options.len() > 26 {
            return Err(ZevError::InvalidRequest(format!(
                "Tev1 requests support a maximum of 26 options, got {}",
                options.len()
            )));
        }

        Ok(Self {
            state,
            question,
            options,
            model: Some("together/Tev1-4B-experimental".into()),
        })
    }
}

pub(crate) fn index_to_label(mut idx: usize) -> String {
    if idx < 26 {
        return ((b'A' + (idx as u8)) as char).to_string();
    }
    let mut result = Vec::new();
    loop {
        let rem = (idx % 26) as u8;
        result.push(b'A' + rem);
        if idx < 26 {
            break;
        }
        idx = (idx / 26) - 1;
    }
    result.reverse();
    String::from_utf8(result).unwrap_or_else(|_| "A".to_string())
}

fn extract_header_content<'a>(line: &'a str, header: &str) -> &'a str {
    let mut remainder = &line[header.len()..];
    remainder = remainder.trim_start_matches(':');
    remainder.trim()
}

fn parse_option_line(line: &str, options: &mut Vec<String>) {
    // Check if line contains inline multiple options like "A: Yes   B: No   C: Not enough information"
    let parts: Vec<&str> = line.split_whitespace().collect();
    let mut curr_opt = String::new();

    for part in parts {
        let is_letter_prefix = part.len() >= 2
            && part.chars().next().unwrap().is_ascii_alphabetic()
            && (part.ends_with(':') || part.ends_with('.'));

        if is_letter_prefix && !curr_opt.is_empty() {
            options.push(curr_opt.trim().to_string());
            curr_opt.clear();
        }

        if !curr_opt.is_empty() {
            curr_opt.push(' ');
        }
        curr_opt.push_str(part);
    }

    if !curr_opt.is_empty() {
        options.push(curr_opt.trim().to_string());
    }
}

/// Evaluates a Tev1 decision request using Zev's high-speed order-invariant engine
pub fn evaluate_tev1_request(req: &Tev1Request, default_temp: f64) -> Result<Tev1Response> {
    let start = Instant::now();

    if req.state.len() > crate::types::MAX_STATE_BYTES {
        return Err(ZevError::InvalidRequest(
            "State size exceeds maximum allowed limit of 2MB".into(),
        ));
    }

    if req.options.len() < 2 {
        return Err(ZevError::InvalidRequest(
            "Tev1 requests require at least 2 options".into(),
        ));
    }
    if req.options.len() > 26 {
        return Err(ZevError::InvalidRequest(format!(
            "Tev1 requests support a maximum of 26 options, got {}",
            req.options.len()
        )));
    }

    // Preprocess premise (combining state + question)
    let combined_premise = format!("{}\nQuestion: {}", req.state, req.question);
    let preprocessed = preprocess_state(&combined_premise, true);
    let ctx = PremiseContext::new(&preprocessed);

    // Normalize option letter labels and clean descriptions
    let mut candidates = Vec::with_capacity(req.options.len());
    let mut letter_labels = Vec::with_capacity(req.options.len());
    let mut choice_texts = Vec::with_capacity(req.options.len());
    let mut seen_labels = std::collections::HashSet::new();

    for (idx, opt_str) in req.options.iter().enumerate() {
        let trimmed = opt_str.trim();
        let (letter, text) = if trimmed.len() >= 3
            && trimmed.chars().next().unwrap().is_ascii_alphabetic()
            && (trimmed.chars().nth(1) == Some(':') || trimmed.chars().nth(1) == Some('.'))
        {
            let letter = trimmed[0..1].to_uppercase();
            let text = trimmed[2..].trim();
            (letter, text)
        } else {
            let letter = index_to_label(idx);
            (letter, trimmed)
        };

        if !seen_labels.insert(letter.clone()) {
            return Err(ZevError::InvalidRequest(format!(
                "Duplicate option label '{}' detected in Tev1 options",
                letter
            )));
        }

        candidates.push(Candidate {
            id: letter.clone(),
            description: text.to_string(),
            value: None,
        });
        letter_labels.push(letter);
        choice_texts.push(text.to_string());
    }

    // Isolated order-invariant scoring
    let logits = compute_order_invariant_logits_with_context(&ctx, &candidates);

    let family = if req.question.to_lowercase().contains("intent") {
        "intent"
    } else if req.question.to_lowercase().contains("policy") {
        "policy"
    } else {
        "choice"
    };
    let family_temp = crate::calibration::family_calibrated_temperature(family, default_temp);
    let effective_temp =
        crate::calibration::dampen_temperature_by_margin(&logits, family_temp, 0.40);

    // Dynamic calibrated temperature softmax
    let probs = scaled_softmax(&logits, effective_temp)?;

    let mut probabilities = BTreeMap::new();
    let mut logprobs = BTreeMap::new();

    let mut best_idx = 0;
    let mut best_prob = -1.0;

    for (i, &p) in probs.iter().enumerate() {
        let letter = letter_labels[i].clone();
        let safe_p = p.max(1e-12);
        let log_p = safe_p.ln();

        probabilities.insert(letter.clone(), p);
        logprobs.insert(letter, log_p);

        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    let execution_micros = start.elapsed().as_secs_f64() * 1_000_000.0;

    Ok(Tev1Response {
        answer: letter_labels[best_idx].clone(),
        choice: choice_texts[best_idx].clone(),
        confidence: best_prob,
        probabilities,
        logprobs,
        execution_micros,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tev1_prompt_parsing() {
        let prompt = r#"
State       Returns are allowed within 30 days. This purchase was 12 days ago.
Question    Is this return within the allowed window?
Options     A: Yes   B: No   C: Not enough information
Answer      A
"#;
        let req = Tev1Request::parse_prompt(prompt).expect("parsing failed");
        assert_eq!(
            req.state,
            "Returns are allowed within 30 days. This purchase was 12 days ago."
        );
        assert_eq!(req.question, "Is this return within the allowed window?");
        assert_eq!(req.options.len(), 3);
        assert_eq!(req.options[0], "A: Yes");
        assert_eq!(req.options[1], "B: No");
        assert_eq!(req.options[2], "C: Not enough information");
    }

    #[test]
    fn test_tev1_evaluation() {
        let req = Tev1Request {
            state: "Returns are allowed within 30 days. This purchase was 12 days ago.".into(),
            question: "Is this return within the allowed window?".into(),
            options: vec![
                "A: Yes".into(),
                "B: No".into(),
                "C: Not enough information".into(),
            ],
            model: None,
        };

        let resp = evaluate_tev1_request(&req, 2.179).expect("evaluation failed");
        assert_eq!(resp.answer, "A");
        assert_eq!(resp.choice, "Yes");
        assert!(resp.confidence > 0.5);
        assert!(resp.probabilities.contains_key("A"));
        assert!(resp.logprobs.contains_key("A"));
    }

    #[test]
    fn test_tev1_multiline_parsing() {
        let prompt = r#"
State:
First line of state.
Second line of state.
Question:
What is the diagnosis?
Options:
A: Option One
B: Option Two
Answer: A
"#;
        let req = Tev1Request::parse_prompt(prompt).expect("multiline parse failed");
        assert!(req.state.contains("First line"));
        assert!(req.state.contains("Second line"));
        assert_eq!(req.question, "What is the diagnosis?");
        assert_eq!(req.options.len(), 2);
    }

    #[test]
    fn test_tev1_parser_errors() {
        assert!(Tev1Request::parse_prompt("Question: What?\nOptions: A: 1 B: 2").is_err());
        assert!(Tev1Request::parse_prompt("State: Here\nOptions: A: 1 B: 2").is_err());
        assert!(Tev1Request::parse_prompt("State: Here\nQuestion: What?\nOptions: A: 1").is_err());
    }

    #[test]
    fn test_tev1_implicit_letters_and_empty_options() {
        let req_few = Tev1Request {
            state: "test".into(),
            question: "test".into(),
            options: vec!["only_one".into()],
            model: None,
        };
        assert!(evaluate_tev1_request(&req_few, 1.0).is_err());

        let req_implicit = Tev1Request {
            state: "Database outage occurred".into(),
            question: "Route".into(),
            options: vec!["Database cluster".into(), "Billing support".into()],
            model: None,
        };
        let resp = evaluate_tev1_request(&req_implicit, 1.0).unwrap();
        assert_eq!(resp.answer, "A");
        assert_eq!(resp.choice, "Database cluster");
    }

    #[test]
    fn test_tev1_rejects_more_than_26_options() {
        let options: Vec<String> = (0..27).map(|i| format!("Option {}", i)).collect();
        let req = Tev1Request {
            state: "State".into(),
            question: "Question".into(),
            options,
            model: None,
        };
        let err = evaluate_tev1_request(&req, 1.0).unwrap_err();
        match err {
            ZevError::InvalidRequest(msg) => {
                assert!(msg.contains("maximum of 26 options"));
            }
            _ => panic!("Expected InvalidRequest, got {:?}", err),
        }
    }

    #[test]
    fn test_tev1_parse_prompt_rejects_more_than_26_options() {
        let mut prompt = String::from("State: Some state\nQuestion: Some question\nOptions:\n");
        for i in 0..27 {
            prompt.push_str(&format!("{}: Option {}\n", (b'A' + (i % 26)) as char, i));
        }
        let res = Tev1Request::parse_prompt(&prompt);
        assert!(res.is_err());
    }

    #[test]
    fn test_tev1_duplicate_labels_rejected() {
        // Explicit duplicates
        let req = Tev1Request {
            state: "Database outage occurred".into(),
            question: "Route".into(),
            options: vec!["A: One".into(), "A: Two".into()],
            model: None,
        };
        let err = evaluate_tev1_request(&req, 1.0).unwrap_err();
        match err {
            ZevError::InvalidRequest(msg) => {
                assert!(msg.contains("Duplicate option label 'A'"));
            }
            _ => panic!("Expected InvalidRequest, got {:?}", err),
        }

        // Implicit and explicit collision
        let req_collision = Tev1Request {
            state: "Database outage occurred".into(),
            question: "Route".into(),
            options: vec!["B: One".into(), "Two".into()],
            model: None,
        };
        let err_collision = evaluate_tev1_request(&req_collision, 1.0).unwrap_err();
        match err_collision {
            ZevError::InvalidRequest(msg) => {
                assert!(msg.contains("Duplicate option label 'B'"));
            }
            _ => panic!("Expected InvalidRequest, got {:?}", err_collision),
        }
    }

    #[test]
    fn test_tev1_index_to_label_overflow_safety_p06() {
        // P06: Test index_to_label with indices >= 190 without panicking or overflowing
        assert_eq!(index_to_label(0), "A");
        assert_eq!(index_to_label(25), "Z");
        assert_eq!(index_to_label(26), "AA");
        assert_eq!(index_to_label(27), "AB");
        assert_eq!(index_to_label(190), "GI");
        assert_eq!(index_to_label(200), "GS");
        assert_eq!(index_to_label(1000), "ALM");

        // Request with 200 options is cleanly rejected without panic (R5 / P06)
        let options_200: Vec<String> = (0..200).map(|i| format!("Option {}", i)).collect();
        let req_200 = Tev1Request {
            state: "State".into(),
            question: "Question".into(),
            options: options_200,
            model: None,
        };
        assert!(evaluate_tev1_request(&req_200, 1.0).is_err());
    }
}
