//! Generic Finite-State Policy DAG Reachability Engine
//!
//! Models conditional policy statements ("if X then Y", "requires Z", "unless W")
//! as a directed propositional graph with bitset transitive closure.
//! Resolves multi-hop deductive reachability and inhibitory exception blocking
//! in sub-microsecond time with zero floating point operations.

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Maximum number of propositional nodes tracked per document DAG (compact u64 bitmask).
pub const MAX_DAG_NODES: usize = 64;

static RE_IF_THEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:if|when|where|provided\s+that|in\s+the\s+event\s+of|in\s+the\s+event\s+that)\s+([^,;:\.\n]{4,80})(?:[,;:]\s*(?:then\s+)?|\s+then\s+)([^,;:\.\n]{4,80})(?:\.|$|;)")
        .expect("valid if-then regex")
});

static RE_CAUSAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{4,60})\s+(?:causes|triggers|leads\s+to|results\s+in|implies|activates|initiates)\s+([^,;:\.\n]{4,60})(?:\.|$|;)")
        .expect("valid causal regex")
});

static RE_REQUIRES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{4,80})\s+(?:requires|is\s+subject\s+to|depends\s+on|is\s+conditional\s+upon)\s+([^,;:\.\n]{4,80})(?:\.|$|;)")
        .expect("valid requires regex")
});

static RE_UNLESS_EXCEPTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{4,80})\s+(?:unless|except\s+when|except\s+if|excluding\s+cases\s+where)\s+([^,;:\.\n]{4,80})(?:\.|$|;)")
        .expect("valid unless regex")
});

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
        for i in 0..MAX_DAG_NODES {
            reach[i] = 1u64 << i; // reflexive: every node reaches itself
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

#[inline]
fn contains_ascii_case_insensitive(haystack: &str, needle_lower: &[u8]) -> bool {
    let hb = haystack.as_bytes();
    let n = needle_lower.len();
    if hb.len() < n {
        return false;
    }
    hb.windows(n).any(|w| w.eq_ignore_ascii_case(needle_lower))
}

/// Parses conditional and policy dependency rules from natural text into a propositional DAG.
pub fn build_policy_dag_from_text(text: &str) -> PolicyDag {
    let mut dag = PolicyDag::new();

    for raw_sent in text.split(['.', ';', '\n']) {
        let sent = raw_sent.trim();
        if sent.len() < 10 || sent.len() > 240 {
            continue;
        }

        // 1. If-Then rules: Condition -> Consequence
        if (sent.contains(',')
            || sent.contains(':')
            || contains_ascii_case_insensitive(sent, b" then "))
            && (contains_ascii_case_insensitive(sent, b"if ")
                || contains_ascii_case_insensitive(sent, b"when ")
                || contains_ascii_case_insensitive(sent, b"where ")
                || contains_ascii_case_insensitive(sent, b"provided ")
                || contains_ascii_case_insensitive(sent, b"event "))
        {
            for cap in RE_IF_THEN.captures_iter(sent) {
                if let (Some(cond_m), Some(cons_m)) = (cap.get(1), cap.get(2)) {
                    if let (Some(u), Some(v)) = (
                        dag.get_or_create_node(cond_m.as_str()),
                        dag.get_or_create_node(cons_m.as_str()),
                    ) {
                        dag.add_implication(u, v);
                    }
                }
            }
        }

        // 1b. Causal rules: Cause -> Effect
        if contains_ascii_case_insensitive(sent, b"causes ")
            || contains_ascii_case_insensitive(sent, b"triggers ")
            || contains_ascii_case_insensitive(sent, b"leads to ")
            || contains_ascii_case_insensitive(sent, b"results in ")
            || contains_ascii_case_insensitive(sent, b"implies ")
            || contains_ascii_case_insensitive(sent, b"activates ")
            || contains_ascii_case_insensitive(sent, b"initiates ")
        {
            for cap in RE_CAUSAL.captures_iter(sent) {
                if let (Some(cause_m), Some(eff_m)) = (cap.get(1), cap.get(2)) {
                    if let (Some(u), Some(v)) = (
                        dag.get_or_create_node(cause_m.as_str()),
                        dag.get_or_create_node(eff_m.as_str()),
                    ) {
                        dag.add_implication(u, v);
                    }
                }
            }
        }

        // 2. Requires rules: Consequence requires Condition => Condition enables Consequence
        if contains_ascii_case_insensitive(sent, b"requires")
            || contains_ascii_case_insensitive(sent, b"subject to")
            || contains_ascii_case_insensitive(sent, b"depends on")
            || contains_ascii_case_insensitive(sent, b"conditional upon")
        {
            for cap in RE_REQUIRES.captures_iter(sent) {
                if let (Some(cons_m), Some(cond_m)) = (cap.get(1), cap.get(2)) {
                    if let (Some(v), Some(u)) = (
                        dag.get_or_create_node(cons_m.as_str()),
                        dag.get_or_create_node(cond_m.as_str()),
                    ) {
                        dag.add_implication(u, v);
                    }
                }
            }
        }

        // 3. Unless / Exception rules: Exception blocks Action (unless Action is a requirement/prohibition)
        if contains_ascii_case_insensitive(sent, b"unless")
            || contains_ascii_case_insensitive(sent, b"except")
            || contains_ascii_case_insensitive(sent, b"excluding")
        {
            for cap in RE_UNLESS_EXCEPTION.captures_iter(sent) {
                if let (Some(action_m), Some(exc_m)) = (cap.get(1), cap.get(2)) {
                    let act_s = action_m.as_str();
                    let is_req_exemption = contains_ascii_case_insensitive(act_s, b"need")
                        || contains_ascii_case_insensitive(act_s, b"require")
                        || contains_ascii_case_insensitive(act_s, b"must")
                        || contains_ascii_case_insensitive(act_s, b"prohibit")
                        || contains_ascii_case_insensitive(act_s, b"forbid")
                        || contains_ascii_case_insensitive(act_s, b"deny")
                        || contains_ascii_case_insensitive(act_s, b"denied");
                    if !is_req_exemption {
                        if let (Some(action_idx), Some(exc_idx)) = (
                            dag.get_or_create_node(act_s),
                            dag.get_or_create_node(exc_m.as_str()),
                        ) {
                            dag.add_inhibition(exc_idx, action_idx);
                        }
                    }
                }
            }
        }
    }

    if dag.node_names.len() >= 2 {
        dag.compute_transitive_closure();
    }
    dag
}

/// Evaluates active premises against the Policy DAG and generates explicit deductive reachability statements.
pub fn resolve_policy_dag_deductions(text: &str) -> Vec<String> {
    let dag = build_policy_dag_from_text(text);
    if dag.node_names.len() < 2 {
        return Vec::new();
    }

    let text_lower = if text.is_ascii() {
        text.to_ascii_lowercase()
    } else {
        text.to_lowercase()
    };
    let mut token_set: HashSet<&str, crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(128, crate::fast_hash::Xxh3BuildHasher);
    for w in text_lower.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if w.len() >= 4 {
            token_set.insert(w);
        }
    }
    let mut active_nodes = Vec::new();

    // Collect tokens from non-exception sentences so inhibitor nodes ("unless X") are only
    // marked active when X is actually asserted as a fact outside the "unless" rule itself.
    let mut fact_token_set: HashSet<&str, crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(128, crate::fast_hash::Xxh3BuildHasher);
    for raw_sent in text_lower.split(['.', ';', '\n']) {
        let s = raw_sent.trim();
        if s.contains("unless ") || s.contains("except ") || s.contains("excluding ") {
            continue;
        }
        for w in s.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if w.len() >= 4 {
                fact_token_set.insert(w);
            }
        }
    }

    // Identify which proposition nodes are asserted / present as active facts in the text
    for (idx, name) in dag.node_names.iter().enumerate() {
        let words: Vec<&str> = name.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        let is_inhibitor = dag.inhibit[idx] != 0;
        if is_inhibitor {
            // An exception condition must be asserted outside its own "unless/except" definition
            let asserted_twice = text_lower.matches(name).count() >= 2;
            let content_words_in_facts = words.iter().all(|w| {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                clean.len() < 4 || fact_token_set.contains(clean)
            });
            if asserted_twice || content_words_in_facts {
                active_nodes.push(idx);
            }
            continue;
        }

        // Check for continuous substring or high token containment
        if text_lower.contains(name) {
            active_nodes.push(idx);
        } else if words.len() >= 3 {
            let matches_all = words.iter().all(|w| {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                clean.len() < 4 || token_set.contains(clean)
            });
            if matches_all {
                active_nodes.push(idx);
            }
        }
    }

    if active_nodes.is_empty() {
        return Vec::new();
    }

    let mut findings = Vec::new();
    let mut recorded_pairs: HashSet<(usize, usize), crate::fast_hash::Xxh3BuildHasher> =
        HashSet::with_capacity_and_hasher(16, crate::fast_hash::Xxh3BuildHasher);

    for &src in &active_nodes {
        let src_name = &dag.node_names[src];

        for target in 0..dag.node_names.len() {
            if target == src {
                continue;
            }

            if dag.is_reachable(src, target) {
                if !recorded_pairs.insert((src, target)) {
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
                if findings.len() >= 8 {
                    break;
                }
            }
        }
        if findings.len() >= 8 {
            break;
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
