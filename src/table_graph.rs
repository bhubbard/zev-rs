//! Generic In-Memory Markdown Table & Relational Graph Resolver
//!
//! Provides deterministic relational parsing for semi-structured text containing Markdown/ASCII
//! tables, section hierarchies (Base Terms vs. Overrides/Endorsements), and numerical constraints.
//!
//! Converts complex multi-hop relational data into explicit 1-hop lexical facts without
//! modifying core model weights or using hardcoded entity memorization.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

/// A parsed tabular structure from text.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkdownTable {
    pub title: Option<String>,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}


static RE_SEPARATOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*\|?\s*[-:]+[-| :]*\|?\s*$").expect("valid separator regex")
});

static RE_SECTION_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(?:={4,}|#{1,4})\s*(.*?)\s*(?:={4,})?$").expect("valid section header regex")
});

static RE_CURRENCY_AMOUNT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)").expect("valid currency regex")
});

static RE_SUBLIMIT_CAP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:sublimit|limit|cap(?:ped)?\s+at|maximum\s+of|up\s+to)\s*(?:of)?\s*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)").expect("valid cap regex")
});

/// Parses all markdown / pipe-delimited tables found in the text.
pub fn parse_markdown_tables(text: &str) -> Vec<MarkdownTable> {
    if !text.contains('|') {
        return Vec::new();
    }
    let mut tables = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_headers: Vec<String> = Vec::new();
    let mut current_rows: Vec<Vec<String>> = Vec::new();
    let mut in_table = false;
    let mut previous_line = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Check if line is a table row
        if trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.matches('|').count() >= 2 {
            let cells: Vec<String> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();

            if !in_table {
                // Potential header row
                current_headers = cells;
                in_table = true;
                if !previous_line.is_empty() && !previous_line.starts_with('|') {
                    current_title = Some(previous_line.trim_matches('=').trim().to_string());
                }
            } else if RE_SEPARATOR.is_match(trimmed) {
                // Table separator line, continue
                continue;
            } else {
                // Data row
                current_rows.push(cells);
            }
        } else {
            if in_table {
                // End of current table
                if !current_headers.is_empty() && !current_rows.is_empty() {
                    tables.push(MarkdownTable {
                        title: current_title.take(),
                        headers: current_headers.clone(),
                        rows: current_rows.clone(),
                    });
                }
                current_headers.clear();
                current_rows.clear();
                in_table = false;
            }
            if !trimmed.is_empty() {
                previous_line = trimmed.to_string();
            }
        }
    }

    if in_table && !current_headers.is_empty() && !current_rows.is_empty() {
        tables.push(MarkdownTable {
            title: current_title,
            headers: current_headers,
            rows: current_rows,
        });
    }

    tables
}

/// Generic multi-hop relational solver:
/// Matches search terms in non-table text against table cells, traverses joins across tables,
/// and synthesizes explicit relational statements.
pub fn resolve_tabular_relations(text: &str) -> Vec<String> {
    let tables = parse_markdown_tables(text);
    if tables.is_empty() {
        return Vec::new();
    }

    let mut header_tokens: HashSet<String, crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(32, crate::fast_hash::Xxh3BuildHasher);
    for t in &tables {
        for h in &t.headers {
            for w in h.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
                if w.len() >= 3 {
                    header_tokens.insert(w.to_ascii_lowercase());
                }
            }
        }
    }

    const TABLE_STOPWORDS: [&str; 28] = [
        "the", "and", "for", "are", "was", "with", "that", "from", "this", "these", "than",
        "then", "have", "been", "were", "some", "which", "what", "when", "where", "who",
        "does", "should", "would", "could", "table", "following", "below",
    ];

    // Extract potential query tokens strictly from text outside tables
    let mut query_tokens: HashSet<String, crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(64, crate::fast_hash::Xxh3BuildHasher);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('|') && trimmed.ends_with('|') {
            continue; // Skip table lines
        }
        for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
            let clean = word.trim();
            if clean.len() >= 3 && !clean.bytes().all(|b| b.is_ascii_digit()) {
                let lower = clean.to_ascii_lowercase();
                if !TABLE_STOPWORDS.contains(&lower.as_str()) && !header_tokens.contains(&lower) {
                    query_tokens.insert(lower);
                }
            }
        }
    }

    if query_tokens.is_empty() {
        return Vec::new();
    }

    let mut relations = Vec::new();

    // Collect 1-hop bridge entities from matched rows
    let mut hop1_bridge_entities: HashSet<String, crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(32, crate::fast_hash::Xxh3BuildHasher);
    let mut matched_rows: HashSet<(usize, usize), crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(32, crate::fast_hash::Xxh3BuildHasher);

    // Map each table row into key-value pairs (Hop 1: Direct matches)
    for (t_idx, table) in tables.iter().enumerate() {
        for (r_idx, row) in table.rows.iter().enumerate() {
            let mut matches_query = false;

            // Check if any cell in this row matches an entity mentioned in the query
            for cell in row {
                for word in cell.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
                    if word.len() >= 4 {
                        let w_lower = word.to_ascii_lowercase();
                        if query_tokens.contains(&w_lower) {
                            matches_query = true;
                            break;
                        }
                    }
                }
                if matches_query {
                    break;
                }
            }

            if matches_query {
                matched_rows.insert((t_idx, r_idx));
                let mut facts = Vec::new();
                for (h_idx, header) in table.headers.iter().enumerate() {
                    if let Some(val) = row.get(h_idx) {
                        let trimmed_val = val.trim();
                        if !trimmed_val.is_empty() {
                            facts.push(format!("{}: {}", header, trimmed_val));
                            let val_lower = trimmed_val.to_ascii_lowercase();
                            if val_lower.len() >= 3
                                && !query_tokens.contains(&val_lower)
                                && !matches!(
                                    val_lower.as_str(),
                                    "yes" | "no" | "true" | "false" | "active" | "inactive"
                                        | "enabled" | "disabled" | "open" | "closed" | "high"
                                        | "medium" | "low" | "none" | "null"
                                )
                                && !val_lower
                                    .bytes()
                                    .all(|b| b.is_ascii_digit() || b == b'.' || b == b',')
                            {
                                hop1_bridge_entities.insert(val_lower);
                            }
                        }
                    }
                }

                if !facts.is_empty() {
                    let title_prefix = table
                        .title
                        .as_deref()
                        .map(|t| format!("In {}: ", t))
                        .unwrap_or_default();
                    relations.push(format!(
                        "[TABULAR RELATION]: {}{}",
                        title_prefix,
                        facts.join(", ")
                    ));
                    if relations.len() >= 12 {
                        break;
                    }
                }
            }
        }
        if relations.len() >= 12 {
            break;
        }
    }

    // Hop 2: Transitive multi-hop joins across tables or subsequent rows using bridge entities
    if !hop1_bridge_entities.is_empty() && relations.len() < 16 {
        for (t_idx, table) in tables.iter().enumerate() {
            for (r_idx, row) in table.rows.iter().enumerate() {
                if matched_rows.contains(&(t_idx, r_idx)) {
                    continue; // Skip rows already included in Hop 1
                }

                let mut matched_bridge = None;
                for cell in row {
                    let cell_lower = cell.trim().to_ascii_lowercase();
                    if hop1_bridge_entities.contains(&cell_lower) {
                        matched_bridge = Some(cell.trim().to_string());
                        break;
                    }
                    for word in
                        cell_lower.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                    {
                        if word.len() >= 4 && hop1_bridge_entities.contains(word) {
                            matched_bridge = Some(cell.trim().to_string());
                            break;
                        }
                    }
                    if matched_bridge.is_some() {
                        break;
                    }
                }

                if let Some(bridge_str) = matched_bridge {
                    let mut facts = Vec::new();
                    for (h_idx, header) in table.headers.iter().enumerate() {
                        if let Some(val) = row.get(h_idx) {
                            let trimmed_val = val.trim();
                            if !trimmed_val.is_empty() && trimmed_val != bridge_str {
                                facts.push(format!("{}: {}", header, trimmed_val));
                            }
                        }
                    }

                    if !facts.is_empty() {
                        let title_prefix = table
                            .title
                            .as_deref()
                            .map(|t| format!("in {}: ", t))
                            .unwrap_or_default();
                        relations.push(format!(
                            "[TRANSITIVE RELATION (2-HOP)]: Via \"{}\" {}{}",
                            bridge_str,
                            title_prefix,
                            facts.join(", ")
                        ));
                        if relations.len() >= 16 {
                            break;
                        }
                    }
                }
            }
            if relations.len() >= 16 {
                break;
            }
        }
    }

    relations
}

/// Generic policy hierarchy resolver:
/// Detects sections marked as Endorsements, Riders, Amendments, Overrides, or Exceptions.
/// If an override section matches query terms, it elevates the override above the base terms.
pub fn resolve_policy_hierarchy(text: &str) -> Vec<String> {
    if !text.contains("====") && !text.contains('#') {
        return Vec::new();
    }
    let mut overrides = Vec::new();

    // Find section headers
    let header_matches: Vec<_> = RE_SECTION_HEADER.find_iter(text).collect();
    if header_matches.is_empty() {
        return overrides;
    }

    for (i, m) in header_matches.iter().enumerate() {
        let header_str = m.as_str();
        let section_title = header_str.trim_matches(|c| c == '=' || c == '#' || c == ' ');

        let start_pos = m.end();
        let end_pos = if i + 1 < header_matches.len() {
            header_matches[i + 1].start()
        } else {
            text.len()
        };

        let section_body = &text[start_pos..end_pos];

        // Check if section is an override, endorsement, amendment, or exception
        let title_lower = section_title.to_lowercase();
        let is_override = title_lower.contains("endorsement")
            || title_lower.contains("amendment")
            || title_lower.contains("override")
            || title_lower.contains("exception")
            || title_lower.contains("rider");

        if is_override {
            // Check for explicit waiver or replacement language
            let body_lower = section_body.to_lowercase();
            let is_waived = body_lower.contains("waived")
                || body_lower.contains("replaces")
                || body_lower.contains("supersedes")
                || body_lower.contains("replaced by")
                || body_lower.contains("applies instead");

            if is_waived {
                overrides.push(format!(
                    "[POLICY OVERRIDE ACTIVE]: Section \"{}\" takes precedence and modifies base exclusions.",
                    section_title
                ));
            }
        }
    }

    overrides
}

/// Generic currency and numeric sublimit analyzer:
/// Detects when a claimed amount exceeds an explicit monetary sublimit or policy cap.
pub fn resolve_monetary_constraints(text: &str) -> Vec<String> {
    if !text.contains('$') {
        return Vec::new();
    }
    let mut constraints = Vec::new();

    // Find sublimit caps and their matched byte spans
    let mut caps: Vec<(f64, std::ops::Range<usize>)> = Vec::new();
    for cap in RE_SUBLIMIT_CAP.captures_iter(text) {
        if let Some(amt_match) = cap.get(1) {
            let clean = amt_match.as_str().replace(',', "");
            if let Ok(val) = clean.parse::<f64>() {
                caps.push((val, amt_match.range()));
            }
        }
    }

    if caps.is_empty() {
        return constraints;
    }

    // Find all mentioned dollar amounts that are not the sublimit itself
    for cap_match in RE_CURRENCY_AMOUNT.captures_iter(text) {
        let m = cap_match.get(1).unwrap();
        let m_range = m.range();
        // Skip if this currency amount is part of the sublimit definition
        if caps.iter().any(|(_, r)| r.start == m_range.start && r.end == m_range.end) {
            continue;
        }

        let clean = m.as_str().replace(',', "");
        if let Ok(claim_val) = clean.parse::<f64>() {
            for &(cap_val, _) in &caps {
                if claim_val > cap_val {
                    constraints.push(format!(
                        "[NUMERIC CONSTRAINT]: Claim amount ${:.2} exceeds stated sublimit ${:.2} (EXCEEDS_SUBLIMIT: TRUE).",
                        claim_val, cap_val
                    ));
                } else if (claim_val - cap_val).abs() < 0.01 {
                    constraints.push(format!(
                        "[NUMERIC CONSTRAINT]: Claim amount ${:.2} matches exactly stated sublimit ${:.2} (MEETS_SUBLIMIT: TRUE).",
                        claim_val, cap_val
                    ));
                }
            }
        }
    }

    constraints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_markdown_table_parsing() {
        let markdown = r#"
| Service | Process | Tier |
|---------|---------|------|
| auth    | authd   | 1    |
| billing | billd   | 2    |
"#;
        let tables = parse_markdown_tables(markdown);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].headers, vec!["Service", "Process", "Tier"]);
        assert_eq!(tables[0].rows.len(), 2);
        assert_eq!(tables[0].rows[0], vec!["auth", "authd", "1"]);
    }

    #[test]
    fn test_resolve_tabular_relations() {
        let text = r#"
Alert on host server-1: process="authd" is crashing.

| Service | Process | Owning Team |
|---------|---------|-------------|
| auth    | authd   | Security    |
| web     | webd    | Frontend    |
"#;
        let rels = resolve_tabular_relations(text);
        assert!(!rels.is_empty());
        assert!(rels[0].contains("Process: authd"));
        assert!(rels[0].contains("Owning Team: Security"));
    }

    #[test]
    fn test_resolve_transitive_multi_hop_relations() {
        let text = r#"
Critical alert for ticket INC-9002: service unresponsive.

Table 1:
| Incident | Service |
|----------|---------|
| INC-9002 | payments|

Table 2:
| Service  | Owning Team | Escalation Channel |
|----------|-------------|--------------------|
| payments | Fintech     | Slack-Fintech-P1   |
"#;
        let rels = resolve_tabular_relations(text);
        assert!(rels.len() >= 2);
        assert!(rels.iter().any(|r| r.contains("payments")));
        assert!(rels.iter().any(|r| r.contains("TRANSITIVE RELATION (2-HOP)") && r.contains("Fintech")));
    }

    #[test]
    fn test_monetary_constraint_resolution() {
        let text = "Policy sublimit of $15,000 applies to water damage. Total claim submitted was $38,000.";
        let res = resolve_monetary_constraints(text);
        assert!(!res.is_empty());
        assert!(res[0].contains("EXCEEDS_SUBLIMIT: TRUE"));
    }
}
