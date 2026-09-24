use std::collections::HashSet;
use crate::types::OptionDef;

/// Shortlists candidate options using lightweight token-set overlap with ID boost and canonical tie-breaking.
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
            let mut matches = 0.0;
            let mut opt_token_count = 0usize;

            // Direct ID match & constituent ID token match
            let id_lower = opt.id.to_lowercase();
            if state_tokens.contains(&id_lower) {
                matches += 4.0;
            }
            for part in id_lower.split(['_', '-']) {
                if part.len() > 2 {
                    opt_token_count += 1;
                    if state_tokens.contains(part) {
                        matches += 2.5;
                    }
                }
            }

            // Description tokens
            for word in opt.description.split_whitespace() {
                let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
                if trimmed.len() > 2 {
                    opt_token_count += 1;
                    let word_lower = trimmed.to_lowercase();
                    if state_tokens.contains(&word_lower) {
                        // Weighted by length (discriminative tokens carry more weight)
                        let weight = 1.0 + (word_lower.len() as f64 - 3.0).clamp(0.0, 3.0) * 0.2;
                        matches += weight;
                    }
                }
            }

            let denom = ((state_tokens.len().max(1) * opt_token_count.max(1)) as f64).sqrt().max(1.0);
            let score = matches / denom;

            (score, opt)
        })
        .collect();

    // Sort descending by score, breaking ties canonically by ID
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.id.cmp(&b.1.id))
    });

    // Keep top max_slots
    scored.into_iter().take(max_slots).map(|(_, opt)| opt.clone()).collect()
}
