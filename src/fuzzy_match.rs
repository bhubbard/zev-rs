//! Fuzzy string distance & phonetic matching algorithms.
//!
//! Synthesized from `call-to-lead-rs/src/matcher.rs`.
//!
//! Provides Jaro-Winkler string similarity and American Soundex phonetic encoding,
//! enabling zero-token candidate matching robust against typos, OCR errors, and phonetic variants.

/// Computes the Jaro-Winkler similarity between two strings, returning a score in [0.0, 1.0].
pub fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    if len1 == 0 || len2 == 0 {
        return 0.0;
    }
    if s1_chars == s2_chars {
        return 1.0;
    }

    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0usize;

    for i in 0..len1 {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(len2);

        for j in start..end {
            if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                continue;
            }
            s1_matches[i] = true;
            s2_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut transpositions = 0usize;
    let mut k = 0usize;

    for i in 0..len1 {
        if !s1_matches[i] {
            continue;
        }
        while !s2_matches[k] {
            k += 1;
        }
        if s1_chars[i] != s2_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let jaro =
        ((m / len1 as f64) + (m / len2 as f64) + ((m - (transpositions / 2) as f64) / m)) / 3.0;

    // Winkler prefix bonus (up to 4 matching initial characters, scaling factor 0.1)
    let mut prefix_len = 0usize;
    for (c1, c2) in s1_chars.iter().zip(s2_chars.iter()).take(4) {
        if c1 == c2 {
            prefix_len += 1;
        } else {
            break;
        }
    }

    jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
}

/// Computes the standard American Soundex phonetic code for an English token.
/// Returns a 4-character string: letter + 3 digits (e.g. "R163").
pub fn soundex_code(s: &str) -> String {
    let clean: String = s
        .chars()
        .filter(|c| c.is_alphabetic())
        .flat_map(|c| c.to_uppercase())
        .collect();

    if clean.is_empty() {
        return "0000".to_string();
    }

    let chars: Vec<char> = clean.chars().collect();
    let first = chars[0];

    let char_to_digit = |c: char| -> Option<char> {
        match c {
            'B' | 'F' | 'P' | 'V' => Some('1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
            'D' | 'T' => Some('3'),
            'L' => Some('4'),
            'M' | 'N' => Some('5'),
            'R' => Some('6'),
            _ => None, // A, E, I, O, U, H, W, Y
        }
    };

    let mut code = String::with_capacity(4);
    code.push(first);

    let mut prev_digit = char_to_digit(first);

    for &c in &chars[1..] {
        let curr_digit = char_to_digit(c);
        if let Some(digit) = curr_digit {
            if Some(digit) != prev_digit {
                code.push(digit);
                if code.len() == 4 {
                    break;
                }
            }
        }
        prev_digit = curr_digit;
    }

    // Pad with zeros to ensure exactly 4 characters
    while code.len() < 4 {
        code.push('0');
    }

    code
}

/// Checks whether two words share the same phonetic sound code.
pub fn phonetic_match(w1: &str, w2: &str) -> bool {
    if w1.is_empty() || w2.is_empty() {
        return false;
    }
    soundex_code(w1) == soundex_code(w2)
}

/// Fuzzy match utility for option IDs and synonyms against query tokens.
pub fn fuzzy_option_match(query_token: &str, candidate_id: &str, min_similarity: f64) -> bool {
    let q = query_token.to_lowercase();
    let c = candidate_id.to_lowercase();

    if q == c {
        return true;
    }

    if jaro_winkler_similarity(&q, &c) >= min_similarity {
        return true;
    }

    phonetic_match(&q, &c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_winkler_similarity() {
        assert!(jaro_winkler_similarity("martha", "marhta") > 0.94);
        assert!(jaro_winkler_similarity("dwayne", "duane") > 0.80);
        assert!(jaro_winkler_similarity("dixon", "dicksonx") > 0.80);
    }

    #[test]
    fn test_soundex_code() {
        assert_eq!(soundex_code("Robert"), "R163");
        assert_eq!(soundex_code("Rupert"), "R163");
        assert_eq!(soundex_code("Rubin"), "R150");
        assert!(phonetic_match("Robert", "Rupert"));
    }

    #[test]
    fn test_fuzzy_option_match() {
        assert!(fuzzy_option_match("apendicitis", "appendicitis", 0.85));
        assert!(fuzzy_option_match("canceling", "cancellation", 0.70));
    }
}
