//! Premise windowing and retrieval-guided context extraction for long inputs.
//!
//! Especially critical for long legal contracts (ContractNLI), clinical records,
//! and long technical documents where 95%+ of the text is irrelevant boilerplate.

const PROMPT_STOPWORDS: &[&str] = &[
    "classify",
    "relationship",
    "between",
    "contract",
    "this",
    "hypothesis",
    "which",
    "party",
    "shall",
    "that",
    "from",
    "with",
    "have",
    "been",
    "these",
    "following",
    "criteria",
    "select",
    "choose",
    "accordance",
    "relevant",
];

/// Extract the most relevant premise window from long text given a query/question.
pub fn extract_premise_window(state: &str, query: &str, max_sentences: usize) -> Option<String> {
    if state.len() < 800 {
        return None;
    }

    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3 && !PROMPT_STOPWORDS.contains(w))
        .collect();

    if query_words.is_empty() {
        return None;
    }

    // Split state into sentences
    let mut sentences = Vec::new();
    for line in state.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        for chunk in trimmed.split(". ") {
            let s = chunk.trim();
            if !s.is_empty() {
                sentences.push(s);
            }
        }
    }

    if sentences.len() <= max_sentences {
        return None;
    }

    // Score each sentence by query word hits
    let mut best_idx = 0;
    let mut best_score = 0;

    for (idx, &sent) in sentences.iter().enumerate() {
        let sent_lower = sent.to_lowercase();
        let mut score = 0;
        for &qw in &query_words {
            if sent_lower.contains(qw) {
                score += 1;
            }
        }
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    if best_score == 0 {
        return None;
    }

    // Take a small contiguous window around the best sentence
    let half = max_sentences / 2;
    let start_idx = best_idx.saturating_sub(half);
    let end_idx = (start_idx + max_sentences).min(sentences.len());

    let window: Vec<&str> = sentences[start_idx..end_idx].to_vec();
    Some(window.join(". "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premise_windowing() {
        let state = "First clause about nothing and standard recital language for commercial agreements between two registered corporate entities in good standing. \
                     Second clause about general terms and definitions of party obligations, formal notices, communications, and registered agent requirements. \
                     Third clause states that Recipient shall not disclose Confidential Information to any third party under any circumstances or conditions whatsoever. \
                     Fourth clause states governing law is California and venue in San Francisco county courts of competent jurisdiction. \
                     Fifth clause states agreement duration is two years from the effective date hereof unless terminated early by mutual written agreement. \
                     Sixth clause contains standard severability and integration language for contract, confirming this instrument supersedes prior negotiations.";
        let query = "Confidential Information disclosure to third party";
        let window = extract_premise_window(state, query, 2);
        assert!(window.is_some());
        let extracted = window.unwrap();
        assert!(extracted.contains("Third clause"));
    }
}
