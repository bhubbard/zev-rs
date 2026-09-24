use std::collections::HashSet;
use crate::types::OptionDef;

/// Shortlists candidate options using lightweight token-set cosine overlap.
/// If options <= max_slots, returns the list unchanged.
pub fn shortlist_options(options: &[OptionDef], state: &str, max_slots: usize) -> Vec<OptionDef> {
    if options.len() <= max_slots {
        return options.to_vec();
    }

    let state_tokens: HashSet<String> = state
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 2)
        .collect();

    let mut scored: Vec<(f64, &OptionDef)> = options
        .iter()
        .map(|opt| {
            let opt_text = format!("{} {}", opt.id, opt.description).to_lowercase();
            let opt_tokens: HashSet<String> = opt_text
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() > 2)
                .collect();

            let intersection = state_tokens.intersection(&opt_tokens).count();
            let denom = ((state_tokens.len() * opt_tokens.len()) as f64).sqrt().max(1.0);
            let score = intersection as f64 / denom;

            (score, opt)
        })
        .collect();

    // Sort descending by score
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Keep top max_slots
    scored.into_iter().take(max_slots).map(|(_, opt)| opt.clone()).collect()
}
