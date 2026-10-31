//! Generic Finite-State Policy DAG Reachability Engine
//!
//! Models conditional policy statements ("if X then Y", "requires Z", "unless W")
//! as a directed propositional graph with bitset transitive closure.
//! Resolves multi-hop deductive reachability and inhibitory exception blocking
//! in sub-microsecond time with zero floating point operations.

use std::collections::{HashMap, HashSet};

/// Maximum number of propositional nodes tracked per document DAG (compact u64 bitmask).
pub const MAX_DAG_NODES: usize = 64;

fn split_once_case_insensitive<'a>(text: &'a str, lower: &str, pat: &str) -> Option<(&'a str, &'a str)> {
    let idx = lower.find(pat)?;
    let end = idx + pat.len();
    if !text.is_char_boundary(idx) || !text.is_char_boundary(end) {
        return None;
    }
    Some((&text[..idx], &text[end..]))
}

fn trim_clause_fragment(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == ':' || c == '.')
}

#[derive(Debug, Clone)]
pub struct PolicyDag {
    pub node_names: Vec<String>,
    pub node_lookup: HashMap<String, usize>,
    /// Bitmask reachability matrix: reach[i] & (1 << j) means node i reaches node j
    pub reach: [u64; MAX_DAG_NODES],
    /// Bitmask inhibitory matrix: inhibit[i] & (1 << j) means node i blocks node j
    pub inhibit: [u64; MAX_DAG_NODES],
}

impl Default for PolicyDag {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyDag {
    pub fn new() -> Self {
        let mut reach = [0u64; MAX_DAG_NODES];
        for (i, r) in reach.iter_mut().enumerate().take(MAX_DAG_NODES) {
            *r = 1u64 << i; // reflexive: every node reaches itself
        }
        Self {
            node_names: Vec::with_capacity(32),
            node_lookup: HashMap::with_capacity(32),
            reach,
            inhibit: [0u64; MAX_DAG_NODES],
        }
    }

    pub fn get_or_create_node(&mut self, text: &str) -> Option<usize> {
        let clean = text.trim().to_lowercase();
        if clean.len() < 3 || clean.split_whitespace().count() > 8 {
            return None;
        }

        if let Some(&idx) = self.node_lookup.get(&clean) {
            return Some(idx);
        }

        if self.node_names.len() >= MAX_DAG_NODES {
            return None;
        }

        let idx = self.node_names.len();
        self.node_names.push(clean.clone());
        self.node_lookup.insert(clean, idx);
        Some(idx)
    }

    pub fn add_implication(&mut self, from: usize, to: usize) {
        if from < MAX_DAG_NODES && to < MAX_DAG_NODES {
            self.reach[from] |= 1u64 << to;
        }
    }

    pub fn add_inhibition(&mut self, from: usize, to: usize) {
        if from < MAX_DAG_NODES && to < MAX_DAG_NODES {
            self.inhibit[from] |= 1u64 << to;
        }
    }

    /// Computes the transitive closure via Warshall's bitwise algorithm in O(N^2) bitwise ops.
    pub fn compute_transitive_closure(&mut self) {
        let n = self.node_names.len();
        for k in 0..n {
            let k_mask = 1u64 << k;
            let k_reach = self.reach[k];
            for i in 0..n {
                if (self.reach[i] & k_mask) != 0 {
                    self.reach[i] |= k_reach;
                }
            }
        }
    }

    /// Checks if source node reaches target node.
    pub fn is_reachable(&self, from: usize, to: usize) -> bool {
        if from < MAX_DAG_NODES && to < MAX_DAG_NODES {
            (self.reach[from] & (1u64 << to)) != 0
        } else {
            false
        }
    }

    /// Checks if an active set of nodes contains any inhibitor that blocks target.
    pub fn is_inhibited(&self, active_nodes: &[usize], target: usize) -> bool {
        let target_mask = 1u64 << target;
        for &node in active_nodes {
            if node < MAX_DAG_NODES && (self.inhibit[node] & target_mask) != 0 {
                return true;
            }
        }
        false
    }
}

/// Parses conditional and policy dependency rules from natural text into a propositional DAG.
pub fn build_policy_dag_from_text(text: &str) -> PolicyDag {
    let mut dag = PolicyDag::new();

    for clause in text.split(['\n', '.', ';']) {
        let trimmed = clause.trim();
        if trimmed.len() < 12 || trimmed.len() > 240 {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();

        let has_if = lower.contains("if ")
            || lower.contains("when ")
            || lower.contains("where ")
            || lower.contains("provided ")
            || lower.contains("in the event ");
        let has_req = lower.contains("requires ")
            || lower.contains("subject to ")
            || lower.contains("depends on ")
            || lower.contains("conditional upon ");
        let has_exc = lower.contains("unless ")
            || lower.contains("except ")
            || lower.contains("excluding ");

        if !has_if && !has_req && !has_exc {
            continue;
        }

        // 1. If-Then rules: Condition -> Consequence
        if has_if {
            for kw in &[
                "in the event that ",
                "in the event of ",
                "provided that ",
                "when ",
                "where ",
                "if ",
            ] {
                if let Some((_, rest)) = split_once_case_insensitive(trimmed, &lower, kw) {
                    if let Some(delim_pos) = rest.find([',', ';', ':']) {
                        let cond_str = trim_clause_fragment(&rest[..delim_pos]);
                        let after = rest[delim_pos + 1..].trim_start();
                        let after_lower = after.to_ascii_lowercase();
                        let cons_raw = if after_lower.starts_with("then ") && after.is_char_boundary(5) {
                            &after[5..]
                        } else {
                            after
                        };
                        let cons_end = cons_raw.find([',', ';', ':']).unwrap_or(cons_raw.len());
                        let cons_str = trim_clause_fragment(&cons_raw[..cons_end]);
                        if (4..=80).contains(&cond_str.len()) && (4..=80).contains(&cons_str.len()) {
                            if let (Some(u), Some(v)) = (
                                dag.get_or_create_node(cond_str),
                                dag.get_or_create_node(cons_str),
                            ) {
                                dag.add_implication(u, v);
                            }
                        }
                    }
                    break;
                }
            }
        }

        // 2. Requires rules: Consequence requires Condition => Condition enables Consequence
        if has_req {
            for kw in &[
                " is conditional upon ",
                " is subject to ",
                " depends on ",
                " requires ",
            ] {
                if let Some((lhs, rhs)) = split_once_case_insensitive(trimmed, &lower, kw) {
                    let cons_start = lhs.rfind([',', ';', ':']).map(|p| p + 1).unwrap_or(0);
                    let cond_end = rhs.find([',', ';', ':']).unwrap_or(rhs.len());
                    let cons_str = trim_clause_fragment(&lhs[cons_start..]);
                    let cond_str = trim_clause_fragment(&rhs[..cond_end]);
                    if (4..=80).contains(&cons_str.len()) && (4..=80).contains(&cond_str.len()) {
                        if let (Some(v), Some(u)) = (
                            dag.get_or_create_node(cons_str),
                            dag.get_or_create_node(cond_str),
                        ) {
                            dag.add_implication(u, v);
                        }
                    }
                    break;
                }
            }
        }

        // 3. Unless / Exception rules: Exception blocks Action
        if has_exc {
            for kw in &[
                " excluding cases where ",
                " except when ",
                " except if ",
                " unless ",
            ] {
                if let Some((lhs, rhs)) = split_once_case_insensitive(trimmed, &lower, kw) {
                    let act_start = lhs.rfind([',', ';', ':']).map(|p| p + 1).unwrap_or(0);
                    let exc_end = rhs.find([',', ';', ':']).unwrap_or(rhs.len());
                    let act_str = trim_clause_fragment(&lhs[act_start..]);
                    let exc_str = trim_clause_fragment(&rhs[..exc_end]);
                    if (4..=80).contains(&act_str.len()) && (4..=80).contains(&exc_str.len()) {
                        if let (Some(action_idx), Some(exc_idx)) = (
                            dag.get_or_create_node(act_str),
                            dag.get_or_create_node(exc_str),
                        ) {
                            dag.add_inhibition(exc_idx, action_idx);
                        }
                    }
                    break;
                }
            }
        }
    }

    dag.compute_transitive_closure();
    dag
}

/// Evaluates active premises against the Policy DAG and generates explicit deductive reachability statements.
pub fn resolve_policy_dag_deductions(text: &str) -> Vec<String> {
    let dag = build_policy_dag_from_text(text);
    if dag.node_names.len() < 2 {
        return Vec::new();
    }

    let text_lower = text.to_lowercase();
    let mut active_nodes = Vec::new();

    // Split text into non-rule fact sentences (sentences not defining conditional/exception rules)
    let fact_sentences: Vec<&str> = text_lower
        .split(['.', '!', '?', '\n'])
        .map(|s| s.trim())
        .filter(|s| {
            !s.is_empty()
                && !s.starts_with("if ")
                && !s.contains(" if ")
                && !s.starts_with("when ")
                && !s.contains(" when ")
                && !s.contains(" requires ")
                && !s.contains(" depends on ")
                && !s.contains(" is subject to ")
                && !s.contains(" is conditional upon ")
                && !s.contains(" unless ")
                && !s.contains(" except ")
        })
        .collect();

    // Identify which proposition nodes are asserted / present as active facts in the text
    for (idx, name) in dag.node_names.iter().enumerate() {
        let words: Vec<&str> = name.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        let is_asserted = fact_sentences.iter().any(|sent| {
            let has_negation = sent.contains("no ")
                || sent.contains("not ")
                || sent.contains("never ")
                || sent.contains("without ")
                || sent.contains("absent")
                || sent.contains("empty")
                || sent.contains("missing")
                || sent.contains("unpaid")
                || sent.contains("split");
            if has_negation {
                return false;
            }
            if sent.contains(name) {
                true
            } else if words.len() >= 3 {
                words.iter().all(|w| w.len() < 4 || sent.contains(w))
            } else {
                false
            }
        });

        if is_asserted {
            active_nodes.push(idx);
        }
    }

    if active_nodes.is_empty() {
        return Vec::new();
    }

    let mut findings = Vec::new();
    let mut recorded_pairs = HashSet::new();

    for &src in &active_nodes {
        let src_name = &dag.node_names[src];

        for target in 0..dag.node_names.len() {
            if target == src {
                continue;
            }

            if dag.is_reachable(src, target) {
                let pair_key = format!("{}:{}", src, target);
                if !recorded_pairs.insert(pair_key) {
                    continue;
                }

                let target_name = &dag.node_names[target];

                // Check if any active node acts as an inhibitor to this target
                if dag.is_inhibited(&active_nodes, target) {
                    findings.push(format!(
                        "[POLICY DEDUCTION BLOCKED]: Condition \"{}\" leads to \"{}\", but is BLOCKED by an active exception (DEDUCTION_VALID: FALSE, EXCEPTION_ACTIVE: TRUE).",
                        src_name, target_name
                    ));
                } else {
                    findings.push(format!(
                        "[POLICY DEDUCTION VALID]: Condition \"{}\" transitively establishes \"{}\" (DEDUCTION_VALID: TRUE, REACHABLE: TRUE).",
                        src_name, target_name
                    ));
                }
            }
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_transitive_closure() {
        let mut dag = PolicyDag::new();
        let a = dag.get_or_create_node("item defective").unwrap();
        let b = dag.get_or_create_node("eligible for replacement").unwrap();
        let c = dag.get_or_create_node("dispatch new unit").unwrap();

        dag.add_implication(a, b);
        dag.add_implication(b, c);
        dag.compute_transitive_closure();

        assert!(dag.is_reachable(a, b));
        assert!(dag.is_reachable(b, c));
        assert!(dag.is_reachable(a, c), "Transitive 2-hop closure must reach C from A");
    }

    #[test]
    fn test_dag_inhibition() {
        let mut dag = PolicyDag::new();
        let req = dag.get_or_create_node("request refund").unwrap();
        let exc = dag.get_or_create_node("opened software seal").unwrap();

        dag.add_inhibition(exc, req);
        assert!(dag.is_inhibited(&[exc], req));
        assert!(!dag.is_inhibited(&[], req));
    }

    #[test]
    fn test_resolve_policy_dag_deductions_full() {
        let text = r#"
Policy guidelines:
If item is defective, customer is eligible for replacement.
Customer is eligible for replacement requires supervisor approval.
Customer contacted us: item is defective.
"#;
        let deductions = resolve_policy_dag_deductions(text);
        assert!(!deductions.is_empty());
        assert!(deductions.iter().any(|d| d.contains("DEDUCTION_VALID: TRUE")));
    }
}
