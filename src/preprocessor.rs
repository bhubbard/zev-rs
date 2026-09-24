use std::borrow::Cow;
use chrono::{Duration, Utc};

/// Injects dynamic temporal reference facts to ground relative time expressions.
/// Zero-allocation fast-path when no relative temporal words are found.
pub fn inject_temporal_facts<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();
    let has_relative_time = bytes.windows(5).any(|w| {
        w.eq_ignore_ascii_case(b"today")
            || w.eq_ignore_ascii_case(b"hours")
    }) || bytes.windows(7).any(|w| {
        w.eq_ignore_ascii_case(b"days ag")
            || w.eq_ignore_ascii_case(b"yesterd")
            || w.eq_ignore_ascii_case(b"tomorro")
            || w.eq_ignore_ascii_case(b"last we")
            || w.eq_ignore_ascii_case(b"last mo")
    });

    if !has_relative_time {
        return Cow::Borrowed(text);
    }

    let now = Utc::now().date_naive();
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
        || input.as_bytes().windows(15).any(|w| w.eq_ignore_ascii_case(b"confidentiality"))
        || input.as_bytes().windows(18).any(|w| w.eq_ignore_ascii_case(b"this email and any"));

    if !has_disclaimer {
        return Cow::Borrowed(input.trim());
    }

    let mut cleaned_lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("---")
            || trimmed.starts_with("___")
            || trimmed.to_lowercase().contains("confidentiality notice:")
            || trimmed.to_lowercase().contains("this email and any attachments")
        {
            break;
        }
        cleaned_lines.push(line);
    }

    Cow::Owned(cleaned_lines.join("\n").trim().to_string())
}

pub fn preprocess_state<'a>(state_str: &'a str, enable_temporal: bool) -> Cow<'a, str> {
    let cleaned = clean_text(state_str);
    if enable_temporal {
        match cleaned {
            Cow::Borrowed(s) => inject_temporal_facts(s),
            Cow::Owned(ref s) => {
                let with_temp = inject_temporal_facts(s);
                match with_temp {
                    Cow::Borrowed(_) => cleaned,
                    Cow::Owned(o) => Cow::Owned(o),
                }
            }
        }
    } else {
        cleaned
    }
}
