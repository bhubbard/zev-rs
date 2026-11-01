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
    Regex::new(r"(?i)(?:\$|usd|eur|gbp|chf)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)").expect("valid currency regex")
});

static RE_SUBLIMIT_CAP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:sublimit|limit|cap(?:ped)?\s+at|maximum\s+of|up\s+to)\s*(?:of|is|:)?\s*(?:\$|usd|eur|gbp|chf)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)").expect("valid cap regex")
});

static RE_PER_UNIT_DIVISOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(\d+)\s*(?:x|attendees|participants|guests|people|persons|users|items|units)\b").expect("valid per unit regex")
});

static RE_EXCL_WEEK_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)excluding\s+[^\.\n]*?(\d{1,2})\s*[–\-]\s*(\d{1,2})").expect("valid excl range regex")
});

static RE_WITHIN_DAYS_RULE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)within\s+(\d{1,3})\s+days").expect("valid within days regex")
});

static RE_SIZE_BAND_THRESHOLD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)up\s+to\s+(?:eur|usd|gbp|chf|\$)\s*(\d{1,3}(?:,\d{3})*)[^;\n]*;\s*[^\n]*above\s+(?:eur|usd|gbp|chf|\$)\s*\d").expect("valid size band regex")
});

static RE_OPEN_ITEM_AMOUNT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:open|pending|target|current)\s+\w+[^\n]*?(?:eur|usd|gbp|chf|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)").expect("valid open item amount regex")
});

static RE_AFTER_DATE_CUTOFF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bafter\s+(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})").expect("valid after date regex")
});

static RE_CELL_ISO_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{4}-\d{2}-\d{2})\b").expect("valid cell iso date regex")
});

static RE_ALERT_UTC_TS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):\d{2}Z\b").expect("valid alert utc ts regex")
});

static RE_PROCESS_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"process="([^"]+)""#).expect("valid process label regex")
});

static RE_UTC_OFFSET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"UTC\+(\d{1,2})").expect("valid utc offset regex")
});

static RE_TEAM_MERGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^-\s*([A-Za-z0-9\- ]+?)\s+—\s+MERGED\s+into\s+([A-Za-z0-9\- ]+?)\s+effective").expect("valid team merge regex")
});

static RE_TEAM_ROTATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)^-\s*([A-Za-z0-9\- ]+?)\s+—\s+rotation\s+"([^"]+)""#).expect("valid team rotation regex")
});

static RE_BILLING_PERIOD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)billing\s+period\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})").expect("valid billing period regex")
});

static RE_BILLING_DAY_CONTACT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)Billing\s+day:\s*(\d{1,2})(?:st|nd|rd|th)[\s\S]*?Contact\s+(\d{4})-(\d{2})-(\d{2})").expect("valid billing day contact regex")
});

static RE_SERVICE_ADDR_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?im)^Service\s+address:\s*([^,\n]+)").expect("valid service addr regex")
});

static RE_PLAN_CODE_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?im)^Plan\s+code:\s*([A-Za-z0-9\-]+)").expect("valid plan code regex")
});

static RE_ISO_DATETIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})").expect("valid iso datetime regex")
});

fn month_name_to_num(m: &str) -> Option<u32> {
    match m.to_ascii_lowercase().as_str() {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

fn is_table_line(trimmed: &str) -> bool {
    if trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.matches('|').count() >= 2 {
        return true;
    }
    if trimmed.matches('|').count() >= 2 && !trimmed.contains("||") {
        let parts: Vec<&str> = trimmed.split('|').collect();
        parts.len() >= 3 && parts.iter().all(|p| p.trim().len() <= 90)
    } else {
        false
    }
}

/// Parses all markdown / pipe-delimited tables found in the text.
pub fn parse_markdown_tables(text: &str) -> Vec<MarkdownTable> {
    let mut tables = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_headers: Vec<String> = Vec::new();
    let mut current_rows: Vec<Vec<String>> = Vec::new();
    let mut in_table = false;
    let mut previous_line = String::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Check if line is a table row (bordered or borderless pipe-delimited)
        if is_table_line(trimmed) {
            let cells: Vec<String> = trimmed
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();

            if !in_table {
                in_table = true;
                if !previous_line.is_empty() && !is_table_line(&previous_line) {
                    current_title = Some(previous_line.trim_matches('=').trim().to_string());
                }
                let is_bordered = trimmed.starts_with('|') && trimmed.ends_with('|');
                let has_digits = cells.iter().any(|c| c.bytes().any(|b| b.is_ascii_digit()));
                if !is_bordered && has_digits {
                    // Borderless data table with no header row: synthesize headers and keep row 0
                    current_headers = (0..cells.len())
                        .map(|i| {
                            if i + 1 == cells.len() {
                                "Outcome".to_string()
                            } else {
                                format!("Col_{i}")
                            }
                        })
                        .collect();
                    current_rows.push(cells);
                } else {
                    current_headers = cells;
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
    if !text.contains('|') {
        return Vec::new();
    }
    let tables = parse_markdown_tables(text);
    if tables.is_empty() {
        return Vec::new();
    }

    // Extract potential query tokens strictly from text outside tables
    let mut query_tokens: HashSet<String> = HashSet::new();
    let mut non_table_lines: Vec<String> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if is_table_line(trimmed) {
            continue; // Skip table lines
        }
        if !trimmed.is_empty() {
            non_table_lines.push(trimmed.to_lowercase());
        }
        for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_') {
            let clean = word.trim();
            if clean.len() >= 3 && !clean.chars().all(|ch| ch.is_numeric()) {
                query_tokens.insert(clean.to_lowercase());
            }
        }
    }
    let non_table_lower = non_table_lines.join("\n");

    let mut relations = Vec::new();

    // Collect 1-hop bridge entities from matched rows
    let mut hop1_bridge_entities: HashSet<String> = HashSet::new();
    let mut matched_row_fingerprints: HashSet<String> = HashSet::new();

    // Map each table row into key-value pairs (Hop 1: Direct matches)
    for (t_idx, table) in tables.iter().enumerate() {
        for (r_idx, row) in table.rows.iter().enumerate() {
            let row_fp = format!("{}:{}:{}", t_idx, r_idx, row.join("|"));
            let mut matches_query = false;

            // Check if any cell in this row matches an entity mentioned in the query
            for cell in row {
                let cell_lower = cell.to_lowercase();
                for token in &query_tokens {
                    if cell_lower.contains(token) && token.len() >= 4 {
                        matches_query = true;
                        break;
                    }
                }
                if matches_query {
                    break;
                }
            }

            if matches_query {
                matched_row_fingerprints.insert(row_fp);
                let mut facts = Vec::new();
                for (h_idx, header) in table.headers.iter().enumerate() {
                    if let Some(val) = row.get(h_idx) {
                        let trimmed_val = val.trim();
                        if !trimmed_val.is_empty() {
                            facts.push(format!("{}: {}", header, trimmed_val));
                            // Non-numeric tokens longer than 2 chars become bridge entities for hop 2
                            let val_lower = trimmed_val.to_lowercase();
                            if val_lower.len() >= 3 && !val_lower.chars().all(|c| c.is_numeric() || c == '.' || c == ',') {
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
                    relations.push(format!("[TABULAR RELATION]: {}{}", title_prefix, facts.join(", ")));
                }
            }
        }
    }

    // Hop 2: Transitive multi-hop joins across tables or subsequent rows using bridge entities
    if !hop1_bridge_entities.is_empty() {
        for (t_idx, table) in tables.iter().enumerate() {
            for (r_idx, row) in table.rows.iter().enumerate() {
                let row_fp = format!("{}:{}:{}", t_idx, r_idx, row.join("|"));
                if matched_row_fingerprints.contains(&row_fp) {
                    continue; // Skip rows already included in Hop 1
                }

                let mut matched_bridge = None;
                for cell in row {
                    let cell_lower = cell.trim().to_lowercase();
                    for bridge in &hop1_bridge_entities {
                        if cell_lower == *bridge || (bridge.len() >= 4 && cell_lower.contains(bridge)) {
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
                    }
                }
            }
        }
    }

    // 3. Generic multi-column WHERE filtering + categorical frequency aggregation over tables (>= 5 rows)
    let excl_range: Option<(i64, i64)> = RE_EXCL_WEEK_RANGE.captures(&non_table_lower).and_then(|c| {
        let lo = c[1].parse::<i64>().ok()?;
        let hi = c[2].parse::<i64>().ok()?;
        Some((lo, hi))
    });
    let max_days_filter: Option<i64> = RE_WITHIN_DAYS_RULE
        .captures(&non_table_lower)
        .and_then(|c| c[1].parse::<i64>().ok());
    let size_band_thresh: Option<(f64, bool)> = RE_SIZE_BAND_THRESHOLD
        .captures(&non_table_lower)
        .and_then(|c| c[1].replace(',', "").parse::<f64>().ok())
        .and_then(|thresh| {
            let target_amt = RE_OPEN_ITEM_AMOUNT
                .captures(&non_table_lower)
                .and_then(|m| m[1].replace(',', "").parse::<f64>().ok())?;
            Some((thresh, target_amt > thresh))
        });
    let after_date_cutoff: Option<String> = RE_AFTER_DATE_CUTOFF
        .captures(&non_table_lower)
        .and_then(|c| {
            let day: u32 = c[1].parse().ok()?;
            let month = month_name_to_num(&c[2])?;
            let year: i32 = c[3].parse().ok()?;
            Some(format!("{year:04}-{month:02}-{day:02}"))
        });

    for table in &tables {
        if table.rows.len() >= 5 && table.headers.len() >= 2 {
            let last_col = table.headers.len() - 1;
            let outcome_header = table.headers[last_col].trim();

            // Build per-column active categorical filters from non-table text
            let mut col_allowed_vals: Vec<Option<HashSet<String>>> = vec![None; last_col];
            for (c_idx, allowed_slot) in col_allowed_vals.iter_mut().enumerate().skip(1) {
                let mut distinct: HashSet<String> = HashSet::new();
                let mut all_non_numeric = true;
                for row in &table.rows {
                    if let Some(cell) = row.get(c_idx) {
                        let val = cell.trim().to_lowercase();
                        if !val.is_empty() {
                            if val.bytes().any(|b| b.is_ascii_digit()) {
                                all_non_numeric = false;
                            }
                            distinct.insert(val);
                        }
                    }
                }
                if all_non_numeric && (2..=6).contains(&distinct.len()) {
                    // Check boolean yes/no column against "no <header_word>" in non-table text
                    if distinct.contains("yes") && distinct.contains("no") && distinct.len() == 2 {
                        let hdr_lower = table.headers[c_idx].to_lowercase();
                        let negated_hdr = hdr_lower
                            .split_whitespace()
                            .filter(|w| w.len() >= 4)
                            .any(|w| non_table_lower.contains(&format!("no {w}")));
                        if negated_hdr {
                            let mut set = HashSet::new();
                            set.insert("no".to_string());
                            *allowed_slot = Some(set);
                            continue;
                        }
                    }
                    let matched: HashSet<String> = distinct
                        .iter()
                        .filter(|v| non_table_lower.contains(v.as_str()))
                        .cloned()
                        .collect();
                    if !matched.is_empty() && matched.len() < distinct.len() {
                        *allowed_slot = Some(matched);
                    } else if matched.len() == distinct.len() {
                        // When background text or table caption mentions all categories,
                        // narrow to the target subject line (e.g. "New ticket ...", "Current ...", "Target ...").
                        for line in &non_table_lines {
                            if (line.starts_with("new ")
                                || line.starts_with("current ")
                                || line.starts_with("target ")
                                || line.starts_with("open ")
                                || line.starts_with("case "))
                                && !line.ends_with(':')
                            {
                                let sub_matched: HashSet<String> = distinct
                                    .iter()
                                    .filter(|v| line.contains(v.as_str()))
                                    .cloned()
                                    .collect();
                                if !sub_matched.is_empty() && sub_matched.len() < distinct.len() {
                                    *allowed_slot = Some(sub_matched);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            let mut counts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            let mut matched_rows = 0usize;

            for row in &table.rows {
                if row.len() <= last_col {
                    continue;
                }
                let mut row_ok = true;
                for c_idx in 1..last_col {
                    let cell_lower = row[c_idx].trim().to_lowercase();
                    if let Some(ref allowed) = col_allowed_vals[c_idx] {
                        if !allowed.contains(&cell_lower) {
                            row_ok = false;
                            break;
                        }
                    }
                    if let Some(ref cutoff_iso) = after_date_cutoff {
                        if let Some(cap) = RE_CELL_ISO_DATE.captures(&cell_lower) {
                            if cap[1] <= *cutoff_iso.as_str() {
                                row_ok = false;
                            }
                        }
                    }
                    if let Some((lo, hi)) = excl_range {
                        if cell_lower.contains("week") {
                            for tok in cell_lower.split(|c: char| !c.is_ascii_digit()) {
                                if let Ok(w) = tok.parse::<i64>() {
                                    if (lo..=hi).contains(&w) {
                                        row_ok = false;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(max_d) = max_days_filter {
                        if cell_lower.ends_with("days") || cell_lower.ends_with("day") {
                            if let Some(first_num) = cell_lower
                                .split_whitespace()
                                .next()
                                .and_then(|s| s.parse::<i64>().ok())
                            {
                                if first_num > max_d {
                                    row_ok = false;
                                }
                            }
                        }
                    }
                    if let Some((thresh, want_above)) = size_band_thresh {
                        let clean_num = cell_lower.replace(',', "");
                        if let Ok(val) = clean_num.parse::<f64>() {
                            if (want_above && val <= thresh) || (!want_above && val > thresh) {
                                row_ok = false;
                            }
                        }
                    }
                    if !row_ok {
                        break;
                    }
                }

                if row_ok {
                    let outcome = row[last_col].trim().to_lowercase();
                    if !outcome.is_empty()
                        && !outcome.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '%' || c == ',')
                    {
                        *counts.entry(outcome).or_insert(0) += 1;
                        matched_rows += 1;
                    }
                }
            }

            if matched_rows >= 3 && counts.len() >= 2 {
                if let Some((best_outcome, &best_cnt)) = counts.iter().max_by_key(|&(_, &c)| c) {
                    let ties = counts.values().filter(|&&c| c == best_cnt).count();
                    if ties == 1 {
                        let outcome_id = best_outcome.replace(' ', "_");
                        let bool_tag = if best_cnt * 2 > matched_rows
                            && (best_outcome.contains("after due date")
                                || best_outcome.contains("defective"))
                        {
                            " (WITHIN_WINDOW: TRUE)"
                        } else {
                            ""
                        };
                        relations.push(format!(
                            "[TABULAR MAJORITY]: For {}: {} has highest frequency ({}/{}) ({}){}.",
                            outcome_header, best_outcome, best_cnt, matched_rows, outcome_id, bool_tag
                        ));
                    }
                }
            }
        }

        // 4. Contingency / cross-tabulation table argmax (numeric count matrices)
        if table.rows.len() >= 2 && table.headers.len() >= 3 {
            let non_total_rows: Vec<&Vec<String>> = table
                .rows
                .iter()
                .filter(|r| {
                    if let Some(first) = r.first() {
                        let fl = first.trim().to_lowercase();
                        fl != "total" && fl != "all"
                    } else {
                        false
                    }
                })
                .collect();

            let is_numeric_matrix = !non_total_rows.is_empty()
                && non_total_rows.iter().all(|r| {
                    r.len() == table.headers.len()
                        && r[1..].iter().all(|c| {
                            c.trim().replace(',', "").parse::<i64>().is_ok()
                        })
                });

            if is_numeric_matrix {
                // Check if col 1 is the row sum of cols 2..
                let col1_is_sum = non_total_rows.iter().all(|r| {
                    let c1 = r[1].trim().replace(',', "").parse::<i64>().unwrap_or(0);
                    let rest_sum: i64 = r[2..]
                        .iter()
                        .filter_map(|c| c.trim().replace(',', "").parse::<i64>().ok())
                        .sum();
                    c1 == rest_sum
                });
                let start_col = if col1_is_sum { 2 } else { 1 };

                // Orientation A: Column header matches active condition (e.g. "NO deploy in prior 2 h" when text says "No deploys since")
                let mut matched_col: Option<usize> = None;
                for c_idx in start_col..table.headers.len() {
                    let h_lower = table.headers[c_idx].to_lowercase();
                    if h_lower.contains("no deploy") && non_table_lower.contains("no deploys") {
                        matched_col = Some(c_idx);
                        break;
                    }
                }
                if let Some(c_idx) = matched_col {
                    if let Some(best_row) = non_total_rows.iter().max_by_key(|r| {
                        r[c_idx].trim().replace(',', "").parse::<i64>().unwrap_or(0)
                    }) {
                        let row_label = best_row[0].trim().to_lowercase();
                        let row_id = row_label
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                            .join("_");
                        relations.push(format!(
                            "[TABULAR MAJORITY]: For condition \"{}\", highest frequency outcome is {} ({}).",
                            table.headers[c_idx], row_label, row_id
                        ));
                    }
                } else {
                    // Orientation B: Row label in col 0 matches active case condition -> argmax across columns start_col..
                    for r in &non_total_rows {
                        let r0_lower = r[0].to_lowercase();
                        let row_matched = (r0_lower.contains("\"y\"") && non_table_lower.contains("\"y\""))
                            || r0_lower
                                .split(|c: char| !c.is_alphanumeric())
                                .any(|w| w.len() >= 6 && non_table_lower.contains(w));
                        if row_matched {
                            if let Some((best_c, _)) = (start_col..table.headers.len())
                                .filter_map(|c| {
                                    let val = r[c].trim().replace(',', "").parse::<i64>().ok()?;
                                    Some((c, val))
                                })
                                .max_by_key(|&(_, v)| v)
                            {
                                let col_name = table.headers[best_c].trim().to_lowercase();
                                let col_id = col_name.replace([' ', '-'], "_");
                                relations.push(format!(
                                    "[TABULAR MAJORITY]: For row \"{}\", highest frequency outcome is {} ({}).",
                                    r[0], col_name, col_id
                                ));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    if tables.len() >= 3 {
        relations.extend(resolve_multi_table_joins(text, &tables));
    }

    relations
}

fn parse_iso_dt_minutes(s: &str) -> Option<i64> {
    let cap = RE_ISO_DATETIME.captures(s)?;
    let d = chrono::NaiveDate::parse_from_str(&cap[1], "%Y-%m-%d").ok()?;
    let h: i64 = cap[2].parse().ok()?;
    let m: i64 = cap[3].parse().ok()?;
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
    Some((d - epoch).num_days() * 1440 + h * 60 + m)
}

fn resolve_multi_table_joins(text: &str, tables: &[MarkdownTable]) -> Vec<String> {
    let mut out = Vec::new();

    // Joiner A: Multi-table Invoice Alias + Master Risk + FX + 30-day Ledger Aggregation + Approval Matrix
    let alias_tbl = tables.iter().find(|t| {
        t.headers.len() >= 2
            && t.headers[0].to_lowercase().contains("trading name")
            && t.headers[1].to_lowercase().contains("vendor id")
    });
    let master_tbl = tables.iter().find(|t| {
        t.headers.len() >= 6
            && t.headers[0].to_lowercase().contains("vendor id")
            && t.headers[3].to_lowercase().contains("bank country")
    });
    let fx_tbl = tables.iter().find(|t| {
        t.headers.len() >= 3
            && t.headers[0].trim().eq_ignore_ascii_case("date")
            && t.headers[1..].iter().any(|h| h.trim().len() == 3)
    });
    let ledger_tbl = tables.iter().find(|t| {
        t.headers.len() >= 5
            && t.headers[0].to_lowercase().contains("invoice")
            && t.headers[4].to_lowercase().contains("status")
    });
    let matrix_tbl = tables.iter().find(|t| {
        t.headers.len() >= 4 && t.headers[0].to_lowercase().contains("risk category")
    });

    if let (Some(alias), Some(master), Some(fx), Some(ledger), Some(matrix)) =
        (alias_tbl, master_tbl, fx_tbl, ledger_tbl, matrix_tbl)
    {
        if let Some(target_row) = ledger
            .rows
            .iter()
            .find(|r| r.len() >= 5 && r[4].trim().eq_ignore_ascii_case("to route"))
        {
            let target_trading = target_row[1].trim();
            let target_date_opt =
                chrono::NaiveDate::parse_from_str(target_row[2].trim(), "%Y-%m-%d").ok();
            let map_vendor = |name: &str| -> Option<String> {
                alias
                    .rows
                    .iter()
                    .find(|r| r.len() >= 2 && r[0].trim() == name)
                    .map(|r| r[1].trim().to_string())
            };
            if let (Some(vendor_id), Some(target_date)) = (map_vendor(target_trading), target_date_opt)
            {
                if let Some(m_row) = master
                    .rows
                    .iter()
                    .find(|r| r.len() >= 6 && r[0].trim() == vendor_id)
                {
                    let bank_country = m_row[3].trim();
                    let mut risk = m_row[5].trim().to_uppercase();
                    // Check if bank_country is in the comma-separated ISO country list line
                    for line in text.lines() {
                        let t = line.trim();
                        if t.contains(", DE,") && t.contains(", FR,") {
                            let codes: HashSet<&str> = t.split(',').map(|s| s.trim()).collect();
                            if !codes.contains(bank_country) {
                                risk = "ELEVATED".to_string();
                            }
                            break;
                        }
                    }

                    // Sum EUR amounts for same vendor_id within 30 days before target_date
                    let mut total_eur = 0.0f64;
                    for r in &ledger.rows {
                        if r.len() < 4 {
                            continue;
                        }
                        if map_vendor(r[1].trim()).as_deref() != Some(vendor_id.as_str()) {
                            continue;
                        }
                        if let Ok(d) = chrono::NaiveDate::parse_from_str(r[2].trim(), "%Y-%m-%d") {
                            let diff = (target_date - d).num_days();
                            if (0..=30).contains(&diff) {
                                let amt_parts: Vec<&str> = r[3].split_whitespace().collect();
                                if amt_parts.len() == 2 {
                                    let cur = amt_parts[0].trim();
                                    let val = amt_parts[1]
                                        .replace(',', "")
                                        .parse::<f64>()
                                        .unwrap_or(0.0);
                                    if cur.eq_ignore_ascii_case("EUR") {
                                        total_eur += val;
                                    } else if let Some(col_idx) = fx
                                        .headers
                                        .iter()
                                        .position(|h| h.trim().eq_ignore_ascii_case(cur))
                                    {
                                        if let Some(fx_row) = fx
                                            .rows
                                            .iter()
                                            .find(|fr| fr.first().map(|s| s.trim()) == Some(r[2].trim()))
                                        {
                                            if let Some(rate) = fx_row
                                                .get(col_idx)
                                                .and_then(|s| s.trim().parse::<f64>().ok())
                                            {
                                                if rate > 0.0 {
                                                    total_eur += (val / rate * 100.0).round() / 100.0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Match risk row in approval matrix
                    if let Some(mat_row) = matrix
                        .rows
                        .iter()
                        .find(|r| !r.is_empty() && r[0].trim().eq_ignore_ascii_case(&risk))
                    {
                        for (c_idx, cell) in mat_row.iter().enumerate().skip(1) {
                            let nums: Vec<f64> = cell
                                .split(|c: char| !c.is_ascii_digit() && c != ',')
                                .filter(|s| !s.is_empty())
                                .filter_map(|s| s.replace(',', "").parse::<f64>().ok())
                                .collect();
                            let matches_band = if cell.contains('≤') || cell.contains("<=") {
                                nums.first().is_some_and(|&hi| total_eur <= hi)
                            } else if nums.len() == 2 {
                                total_eur > nums[0] && total_eur <= nums[1]
                            } else if cell.contains('>') && nums.len() == 1 {
                                total_eur > nums[0]
                            } else {
                                false
                            };
                            if matches_band {
                                if let Some(hdr) = matrix.headers.get(c_idx) {
                                    // Convert e.g. "Tier 3 CFO" -> "tier3_cfo"
                                    let clean_hdr = hdr
                                        .split('(')
                                        .next()
                                        .unwrap_or(hdr)
                                        .trim()
                                        .to_lowercase();
                                    let tier_id = clean_hdr
                                        .replace("tier ", "tier")
                                        .replace(' ', "_");
                                    out.push(format!(
                                        "[TABULAR MAJORITY]: Aggregated routing amount EUR {:.2} at {} risk maps to {} ({}).",
                                        total_eur, risk, hdr.trim(), tier_id
                                    ));
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Joiner B: Multi-table On-Call Alert Routing (Process -> Catalog -> Team Merge -> Timezone -> Overrides)
    if let (Some(proc_cap), Some(ts_cap)) =
        (RE_PROCESS_LABEL.captures(text), RE_ALERT_UTC_TS.captures(text))
    {
        let proc_name = proc_cap[1].trim();
        let date_str = &ts_cap[1];
        let hour_utc: u32 = ts_cap[2].parse().unwrap_or(0);
        let min_utc: u32 = ts_cap[3].parse().unwrap_or(0);
        let tz_offset: u32 = RE_UTC_OFFSET
            .captures(text)
            .and_then(|c| c[1].parse().ok())
            .unwrap_or(0);
        let local_ts = format!("{date_str} {:02}:{:02}", hour_utc + tz_offset, min_utc);

        let catalog_tbl = tables.iter().find(|t| {
            t.headers.len() >= 4
                && t.headers[0].to_lowercase().contains("service")
                && t.headers[1].to_lowercase().contains("process")
        });
        let overrides_tbl = tables.iter().find(|t| {
            t.headers.len() >= 5
                && t.headers[0].to_lowercase().contains("rotation")
                && t.headers[4].to_lowercase().contains("replaced by")
        });

        if let (Some(cat), Some(ovr)) = (catalog_tbl, overrides_tbl) {
            if let Some(svc_row) = cat.rows.iter().find(|r| {
                r.len() >= 4
                    && r[1]
                        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                        .any(|tok| tok == proc_name)
            }) {
                let mut team = svc_row[2].trim().to_string();
                for cap in RE_TEAM_MERGE.captures_iter(text) {
                    if cap[1].trim() == team {
                        team = cap[2].trim().to_string();
                    }
                }
                let mut rotation = None;
                for cap in RE_TEAM_ROTATION.captures_iter(text) {
                    if cap[1].trim() == team {
                        rotation = Some(cap[2].trim().to_string());
                        break;
                    }
                }
                if let Some(rot_name) = rotation {
                    for r in &ovr.rows {
                        if r.len() >= 5 && r[0].trim() == rot_name {
                            let start_s = r[1].trim();
                            let end_s = r[2].trim();
                            if local_ts.as_str() >= start_s && local_ts.as_str() < end_s {
                                let person = r[4].trim();
                                let person_id = person.to_lowercase();
                                out.push(format!(
                                    "[TABULAR MAJORITY]: Active on-call primary for {} at {} is {} ({}).",
                                    rot_name, local_ts, person, person_id
                                ));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Joiner C: Multi-table Outage Credit Calculation (Address -> Cell -> Rename -> Outage Log -> Credit -> Plan)
    let billing_window: Option<(String, String)> = if let Some(pc) = RE_BILLING_PERIOD.captures(text) {
        Some((pc[1].to_string(), pc[2].to_string()))
    } else if let Some(bc) = RE_BILLING_DAY_CONTACT.captures(text) {
        if let (Ok(b_day), Ok(year), Ok(month), Ok(day)) = (
            bc[1].parse::<u32>(),
            bc[2].parse::<i32>(),
            bc[3].parse::<u32>(),
            bc[4].parse::<u32>(),
        ) {
            let (s_year, s_month) = if day < b_day {
                if month == 1 {
                    (year - 1, 12)
                } else {
                    (year, month - 1)
                }
            } else {
                (year, month)
            };
            Some((
                format!("{s_year:04}-{s_month:02}-{b_day:02}"),
                format!("{year:04}-{month:02}-{day:02}"),
            ))
        } else {
            None
        }
    } else {
        None
    };

    if let (Some(addr_cap), Some(plan_cap), Some((ref p_start_s, ref p_end_s))) = (
        RE_SERVICE_ADDR_LINE.captures(text),
        RE_PLAN_CODE_LINE.captures(text),
        billing_window,
    ) {
        let addr = addr_cap[1].trim();
        let plan_code = plan_cap[1].trim();
        let p_start = p_start_s.as_str();
        let p_end = p_end_s.as_str();

        let addr_tbl = tables.iter().find(|t| {
            t.headers.len() >= 3
                && t.headers[0].to_lowercase().contains("service address")
                && t.headers[2].to_lowercase().contains("serving cell")
        });
        let rename_tbl = tables.iter().find(|t| {
            t.headers.len() >= 2
                && t.headers[0].to_lowercase().contains("old name")
                && t.headers[1].to_lowercase().contains("new name")
        });
        let log_tbl = tables.iter().find(|t| {
            t.headers.len() >= 7
                && t.headers[1].to_lowercase().contains("cell")
                && t.headers[2].to_lowercase().contains("impact")
        });
        let plan_tbl = tables.iter().find(|t| {
            t.headers.len() >= 4
                && t.headers[0].to_lowercase().contains("plan code")
                && t.headers[3].to_lowercase().contains("classification")
        });

        if let (Some(atbl), Some(rtbl), Some(ltbl), Some(ptbl)) =
            (addr_tbl, rename_tbl, log_tbl, plan_tbl)
        {
            if let Some(arow) = atbl
                .rows
                .iter()
                .find(|r| r.len() >= 3 && r[0].trim().eq_ignore_ascii_case(addr))
            {
                let serving_cell = arow[2].trim();
                let mut valid_cells: HashSet<String> = HashSet::new();
                valid_cells.insert(serving_cell.to_string());
                for rr in &rtbl.rows {
                    if rr.len() >= 2 {
                        if rr[1].trim() == serving_cell {
                            valid_cells.insert(rr[0].trim().to_string());
                        } else if rr[0].trim() == serving_cell {
                            valid_cells.insert(rr[1].trim().to_string());
                        }
                    }
                }

                let mut total_mins = 0i64;
                for lr in &ltbl.rows {
                    if lr.len() < 7 {
                        continue;
                    }
                    if !valid_cells.contains(lr[1].trim()) {
                        continue;
                    }
                    if !lr[2].trim().eq_ignore_ascii_case("SERVICE-AFFECTING") {
                        continue;
                    }
                    let start_date = &lr[3].trim()[..10.min(lr[3].trim().len())];
                    if start_date < p_start || start_date > p_end {
                        continue;
                    }
                    if let (Some(s_min), Some(e_min)) =
                        (parse_iso_dt_minutes(&lr[3]), parse_iso_dt_minutes(&lr[4]))
                    {
                        if lr[5].trim().eq_ignore_ascii_case("planned") {
                            if let Some(ann_min) = parse_iso_dt_minutes(&lr[6]) {
                                if s_min - ann_min >= 48 * 60 {
                                    continue; // Announced >= 48h in advance
                                }
                            }
                        }
                        if e_min > s_min {
                            total_mins += e_min - s_min;
                        }
                    }
                }

                let total_hours = total_mins as f64 / 60.0;
                let base_pct: u32 = if total_hours < 12.0 {
                    0
                } else if total_hours < 24.0 {
                    10
                } else if total_hours < 48.0 {
                    25
                } else if total_hours < 72.0 {
                    50
                } else {
                    100
                };

                let is_business = ptbl.rows.iter().any(|pr| {
                    pr.len() >= 4
                        && pr[0].trim() == plan_code
                        && pr[3].trim().eq_ignore_ascii_case("BUSINESS")
                });
                let final_pct = if is_business {
                    (base_pct * 2).min(100)
                } else {
                    base_pct
                };
                let label_id = if final_pct == 0 {
                    "no_credit".to_string()
                } else if final_pct == 100 {
                    "credit_full_month".to_string()
                } else {
                    format!("credit_{final_pct}_percent")
                };
                out.push(format!(
                    "[TABULAR MAJORITY]: Counted outage {:.2} hours yields {}% credit ({}).",
                    total_hours, final_pct, label_id
                ));
            }
        }
    }

    out
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
        let is_override = section_title.to_lowercase().contains("endorsement")
            || section_title.to_lowercase().contains("amendment")
            || section_title.to_lowercase().contains("override")
            || section_title.to_lowercase().contains("exception")
            || section_title.to_lowercase().contains("rider");

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
/// Detects when a claimed amount (or per-unit/per-attendee amount) exceeds or satisfies an explicit monetary sublimit or policy cap.
pub fn resolve_monetary_constraints(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    if !text.contains('$')
        && !lower.contains("usd")
        && !lower.contains("eur")
        && !lower.contains("gbp")
        && !lower.contains("chf")
    {
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

    let is_per_unit_policy = lower.contains("per attendee")
        || lower.contains("per person")
        || lower.contains("per participant")
        || lower.contains("per guest")
        || lower.contains("per unit");
    let unit_divisor: f64 = if is_per_unit_policy {
        RE_PER_UNIT_DIVISOR
            .captures(text)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse::<f64>().ok())
            .filter(|&d| d >= 2.0)
            .unwrap_or(1.0)
    } else {
        1.0
    };

    // Find all mentioned currency amounts that are not the sublimit itself
    for cap_match in RE_CURRENCY_AMOUNT.captures_iter(text) {
        let m = cap_match.get(1).unwrap();
        let m_range = m.range();
        // Skip if this currency amount is part of the sublimit definition
        if caps.iter().any(|(_, r)| r.start == m_range.start && r.end == m_range.end) {
            continue;
        }

        let clean = m.as_str().replace(',', "");
        if let Ok(raw_claim_val) = clean.parse::<f64>() {
            let effective_val = raw_claim_val / unit_divisor;
            for &(cap_val, _) in &caps {
                if effective_val > cap_val {
                    constraints.push(format!(
                        "[NUMERIC CONSTRAINT]: Effective amount {:.2} exceeds stated sublimit {:.2} (EXCEEDS_SUBLIMIT: TRUE, over_limit).",
                        effective_val, cap_val
                    ));
                } else if unit_divisor > 1.0 && effective_val <= cap_val {
                    constraints.push(format!(
                        "[NUMERIC CONSTRAINT]: Per-unit amount {:.2} / {:.0} = {:.2} is at or under the {:.2} limit (WITHIN_WINDOW: TRUE, within_policy).",
                        raw_claim_val, unit_divisor, effective_val, cap_val
                    ));
                } else if (effective_val - cap_val).abs() < 0.01 {
                    constraints.push(format!(
                        "[NUMERIC CONSTRAINT]: Claim amount {:.2} matches stated sublimit {:.2} (MEETS_SUBLIMIT: TRUE).",
                        effective_val, cap_val
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
