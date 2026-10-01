use chrono::Duration;
#[cfg(not(target_arch = "wasm32"))]
use chrono::Utc;
use std::borrow::Cow;

/// Injects dynamic temporal reference facts to ground relative time expressions.
/// Zero-allocation fast-path when no relative temporal words are found.
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

    if !has_relative_time {
        return Cow::Borrowed(text);
    }

    #[cfg(not(target_arch = "wasm32"))]
    let now = Utc::now().date_naive();
    #[cfg(target_arch = "wasm32")]
    let now = chrono::NaiveDate::from_ymd_opt(2026, 10, 1).unwrap();
    let yesterday = now - Duration::days(1);
    let seven_days_ago = now - Duration::days(7);

    Cow::Owned(format!(
        "{}\n\n[Temporal Facts: reference_date={}, yesterday={}, 7_days_ago={}]",
        text.trim(),
        now,
        yesterday,
        seven_days_ago
    ))
}

/// Cleans email signatures, disclaimers, and boilerplate quotes.
/// Zero-allocation fast-path when no disclaimers are present.
pub fn clean_text<'a>(input: &'a str) -> Cow<'a, str> {
    let has_disclaimer = input.contains("---")
        || input.contains("___")
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

    let mut cleaned_lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("---")
            || trimmed.starts_with("___")
            || trimmed.to_lowercase().contains("confidentiality notice:")
            || trimmed
                .to_lowercase()
                .contains("this email and any attachments")
        {
            break;
        }
        cleaned_lines.push(line);
    }

    Cow::Owned(cleaned_lines.join("\n").trim().to_string())
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
                let speaker = turn.get("speaker").and_then(|v| v.as_str()).unwrap_or("user");
                let text = turn.get("text").and_then(|v| v.as_str()).unwrap_or("");
                parts.push(format!("{}: {}", speaker, text));
            }
        }
        for (k, v) in &map {
            if k == "conversation" {
                continue;
            }
            if let serde_json::Value::Object(sub) = v {
                let sub_strs: Vec<String> = sub
                    .iter()
                    .map(|(sk, sv)| format!("{}: {}", sk, sv))
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

pub fn preprocess_state<'a>(state_str: &'a str, enable_temporal: bool) -> Cow<'a, str> {
    let structured = format_structured_state(state_str);
    match structured {
        Cow::Borrowed(s) => {
            let cleaned = clean_text(s);
            if enable_temporal {
                match cleaned {
                    Cow::Borrowed(c) => inject_temporal_facts(c),
                    Cow::Owned(ref c) => match inject_temporal_facts(c) {
                        Cow::Borrowed(_) => cleaned,
                        Cow::Owned(o) => Cow::Owned(o),
                    },
                }
            } else {
                cleaned
            }
        }
        Cow::Owned(s) => {
            let cleaned = clean_text(&s);
            let result_str = if enable_temporal {
                match cleaned {
                    Cow::Borrowed(c) => inject_temporal_facts(c).into_owned(),
                    Cow::Owned(ref c) => inject_temporal_facts(c).into_owned(),
                }
            } else {
                cleaned.into_owned()
            };
            Cow::Owned(result_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_state_cleaned_and_temporal() {
        let text1 = "Payment failed yesterday.\n---\nSent from Outlook";
        let res1 = preprocess_state(text1, true);
        assert!(!res1.contains("Sent from Outlook"));
        assert!(res1.contains("[Temporal Facts:"));

        let text2 = "Simple issue here.\n---\nConfidentiality Notice: Privileged";
        let res2 = preprocess_state(text2, true);
        assert!(!res2.contains("Confidentiality"));
    }
}
