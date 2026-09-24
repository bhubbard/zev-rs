use chrono::{Duration, Utc};

/// Injects dynamic temporal reference facts to ground relative time expressions
pub fn inject_temporal_facts(text: &str) -> String {
    let lower = text.to_lowercase();
    let has_relative_time = lower.contains("yesterday")
        || lower.contains("tomorrow")
        || lower.contains("today")
        || lower.contains("days ago")
        || lower.contains("last week")
        || lower.contains("last month")
        || lower.contains("hours ago");

    if !has_relative_time {
        return text.to_string();
    }

    let now = Utc::now().date_naive();
    let yesterday = now - Duration::days(1);
    let seven_days_ago = now - Duration::days(7);

    format!(
        "{}\n\n[Temporal Facts: reference_date={}, yesterday={}, 7_days_ago={}]",
        text.trim(),
        now,
        yesterday,
        seven_days_ago
    )
}

/// Cleans email signatures, disclaimers, and boilerplate quotes
pub fn clean_text(input: &str) -> String {
    let mut cleaned_lines = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();

        // Common email disclaimer / confidentiality markers
        if trimmed.starts_with("---")
            || trimmed.starts_with("___")
            || trimmed.to_lowercase().contains("confidentiality notice:")
            || trimmed.to_lowercase().contains("this email and any attachments")
        {
            break;
        }

        cleaned_lines.push(line);
    }

    let joined = cleaned_lines.join("\n");
    joined.trim().to_string()
}

pub fn preprocess_state(state_str: &str, enable_temporal: bool) -> String {
    let cleaned = clean_text(state_str);
    if enable_temporal {
        inject_temporal_facts(&cleaned)
    } else {
        cleaned
    }
}
