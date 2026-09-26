//! Semantic Abstention Registry and Pattern Matching.
//!
//! Inspired by Mapika/decider's abstention pattern matching.
//! Identifies natural language options and candidate descriptions representing
//! abstentions, uncertain responses, and "none of the above" choices.

use crate::types::UNKNOWN;

pub const ABSTAIN_PREFIXES: &[&str] = &[
    "none of the above",
    "none of these",
    "none of the other",
    "none of the options",
    "not listed",
    "no suitable",
    "does not apply",
    "cannot tell",
    "cannot be determined",
    "insufficient information",
    "not enough info",
    "not enough information",
    "not applicable",
    "neither of these",
    "neither of the above",
    "unable to determine",
];

pub const ABSTAIN_EXACT: &[&str] = &[
    "other",
    "other / not covered",
    "other/not covered",
    "something else",
    "neither of these",
    "neither",
    "none",
    "n/a",
    "na",
    "unknown",
    "unsure",
    "undecided",
    "abstain",
    "abstained",
    "cannot tell",
    "__insufficient__",
    "insufficient_evidence",
];

/// Returns true if the text matches an abstention exact keyword or prefix.
pub fn is_abstain_text(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    let clean = t
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .trim();
    if ABSTAIN_EXACT.iter().any(|&exact| clean == exact) {
        return true;
    }
    ABSTAIN_PREFIXES
        .iter()
        .any(|&prefix| clean.starts_with(prefix))
}

/// Returns true if either the candidate id or its description is an abstention option.
pub fn is_abstain_candidate(id: &str, description: &str) -> bool {
    id == UNKNOWN || is_abstain_text(id) || is_abstain_text(description)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstain_text_matching() {
        assert!(is_abstain_text("none of the above"));
        assert!(is_abstain_text("None of the above."));
        assert!(is_abstain_text("not listed"));
        assert!(is_abstain_text("Other"));
        assert!(is_abstain_text("unsure"));
        assert!(is_abstain_text("does not apply to this situation"));
        assert!(is_abstain_text("cannot tell from the context provided"));

        // Regular non-abstaining candidates
        assert!(!is_abstain_text("billing specialist"));
        assert!(!is_abstain_text("technical support issue"));
        assert!(!is_abstain_text("escalate to manager"));
    }

    #[test]
    fn test_abstain_candidate() {
        assert!(is_abstain_candidate("opt_3", "None of the above"));
        assert!(is_abstain_candidate("other", "Category not listed"));
        assert!(is_abstain_candidate(UNKNOWN, "insufficient evidence"));
        assert!(!is_abstain_candidate("billing", "Billing support"));
    }
}
