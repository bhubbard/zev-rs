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
    Regex::new(r"(?i)\b(?:if|when|where|provided\s+that|in\s+the\s+event\s+of|in\s+the\s+event\s+that)\s+([^,;:\.\n]{4,80})[,;:]\s+(?:then\s+)?([^,;:\.\n]{4,80})(?:\.|$|;)")
        .expect("valid if-then regex")
});

static RE_REQUIRES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{3,80})\s+(?:requires?|are\s+subject\s+to|is\s+subject\s+to|depends?\s+on|is\s+conditional\s+upon|are\s+conditional\s+upon)\s+([^,;:\.\n]{3,80})(?:\.|$|;)")
        .expect("valid requires regex")
});

static RE_UNLESS_EXCEPTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{3,80})\s+(?:unless|except\s+when|except\s+if|excluding\s+cases\s+where)\s+([^,;:\.\n]{3,80})(?:\.|$|;)")
        .expect("valid unless regex")
});

static RE_OVERRIDES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b([^,;:\.\n]{3,80})\s+(?:overrides?|supersedes?|prohibits?|blocks?|revokes?)\s+([^,;:\.\n]{3,80})(?:\.|$|;)")
        .expect("valid overrides regex")
});

#[derive(Debug, Clone)]
pub struct PolicyDag {
    pub node_names: Vec<String>,
    pub node_lookup: HashMap<String, usize>,
    /// Bitmask reachability matrix: reach[i] & (1 << j) means node i reaches node j
    pub reach: [u64; MAX_DAG_NODES],
    /// Bitmask inhibitory matrix: inhibit[i] & (1 << j) means node i blocks node j
    pub inhibit: [u64; MAX_DAG_NODES],
    /// Required preconditions: (action_node_idx, required_condition_idx)
    pub required_preconditions: Vec<(usize, usize)>,
}

impl Default for PolicyDag {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyDag {
    pub fn new() -> Self {
        let mut reach = [0u64; MAX_DAG_NODES];
        for (i, item) in reach.iter_mut().enumerate().take(MAX_DAG_NODES) {
            *item = 1u64 << i; // reflexive: every node reaches itself
        }
        Self {
            node_names: Vec::with_capacity(32),
            node_lookup: HashMap::with_capacity(32),
            reach,
            inhibit: [0u64; MAX_DAG_NODES],
            required_preconditions: Vec::new(),
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
        let has_req = lower.contains("require")
            || lower.contains("subject to ")
            || lower.contains("depends on ")
            || lower.contains("depend on ")
            || lower.contains("conditional upon ");
        let has_exc =
            lower.contains("unless ") || lower.contains("except ") || lower.contains("excluding ");
        let has_ovr = lower.contains("override")
            || lower.contains("supersede")
            || lower.contains("prohibit")
            || lower.contains("block")
            || lower.contains("revoke");

        if !has_if && !has_req && !has_exc && !has_ovr {
            continue;
        }

        // 1. If-Then rules: Condition -> Consequence
        if has_if {
            for cap in RE_IF_THEN.captures_iter(trimmed) {
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

        // 2. Requires rules: Consequence requires Condition => Condition enables Consequence (and Modus Tollens: !Condition blocks Consequence)
        if has_req {
            for cap in RE_REQUIRES.captures_iter(trimmed) {
                if let (Some(cons_m), Some(cond_m)) = (cap.get(1), cap.get(2)) {
                    if let Some(v) = dag.get_or_create_node(cons_m.as_str()) {
                        let cond_raw = cond_m.as_str();
                        // Split conjunctive preconditions ("a receipt and purchase within 30 days")
                        for sub_cond in cond_raw.split(" and ").flat_map(|s| s.split(" AND ")) {
                            if let Some(u) = dag.get_or_create_node(sub_cond.trim()) {
                                dag.add_implication(u, v);
                                dag.required_preconditions.push((v, u));
                            }
                        }
                    }
                }
            }
        }

        // 3. Unless / Exception rules: Exception blocks Action
        if has_exc {
            for cap in RE_UNLESS_EXCEPTION.captures_iter(trimmed) {
                if let (Some(action_m), Some(exc_m)) = (cap.get(1), cap.get(2)) {
                    if let (Some(action_idx), Some(exc_idx)) = (
                        dag.get_or_create_node(action_m.as_str()),
                        dag.get_or_create_node(exc_m.as_str()),
                    ) {
                        dag.add_inhibition(exc_idx, action_idx);
                    }
                }
            }
        }

        // 4. Overrides / Prohibits rules: Inhibitor overrides/blocks Target
        if has_ovr {
            for cap in RE_OVERRIDES.captures_iter(trimmed) {
                if let (Some(inhib_m), Some(target_m)) = (cap.get(1), cap.get(2)) {
                    if let (Some(inhib_idx), Some(target_idx)) = (
                        dag.get_or_create_node(inhib_m.as_str()),
                        dag.get_or_create_node(target_m.as_str()),
                    ) {
                        dag.add_inhibition(inhib_idx, target_idx);
                    }
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

    // Build fact context by excluding hypothetical ("if/when...") antecedents and ("unless/except...") tails
    let mut fact_clauses = Vec::new();
    for clause in text.split(['\n', '.', ';']) {
        let trimmed = clause.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower.starts_with("if ")
            || lower.starts_with("when ")
            || lower.starts_with("where ")
            || lower.starts_with("provided ")
            || lower.starts_with("in the event ")
        {
            continue;
        }
        // Strip hypothetical exception tail ("unless ...", "except ...", "excluding ...")
        let fact_part = if let Some(idx) = lower
            .find(" unless ")
            .or_else(|| lower.find(" except "))
            .or_else(|| lower.find(" excluding "))
        {
            &trimmed[..idx]
        } else {
            trimmed
        };
        fact_clauses.push(fact_part);
    }
    let fact_text = fact_clauses.join(". ");
    let fact_lower = fact_text.to_lowercase();
    let fact_ctx = crate::order_invariant::PremiseContext::new(&fact_text);
    let ctx = crate::order_invariant::PremiseContext::new(text);
    let mut active_nodes = Vec::new();

    // Identify which proposition nodes are asserted / present as active facts in the text
    for (idx, name) in dag.node_names.iter().enumerate() {
        let words: Vec<&str> = name.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }

        // Check for continuous substring or high token/stem containment in fact clauses
        if fact_lower.contains(name) {
            active_nodes.push(idx);
        } else {
            let content_words: Vec<&str> = words
                .iter()
                .copied()
                .filter(|w| w.len() >= 4 && *w != "temporary" && *w != "all" && *w != "their")
                .collect();
            if !content_words.is_empty()
                && content_words.iter().all(|w| {
                    (fact_ctx.contains_bounded(w) || fact_ctx.contains_stem(w))
                        && !fact_ctx.is_negated(w)
                        && !fact_ctx.is_stem_negated(w)
                })
            {
                active_nodes.push(idx);
            }
        }
    }

    let mut findings = Vec::new();

    // Modus Tollens check on required preconditions:
    // If action V requires precondition U, and U is negated/violated in the premise, V is blocked.
    for &(action_idx, req_idx) in &dag.required_preconditions {
        let action_name = &dag.node_names[action_idx];
        let req_name = &dag.node_names[req_idx];

        // Case A: Negative precondition ("absence of X", "no X", "without X") violated by presence of X
        let neg_prefix_target = req_name
            .strip_prefix("absence of ")
            .or_else(|| req_name.strip_prefix("the absence of "))
            .or_else(|| req_name.strip_prefix("no "))
            .or_else(|| req_name.strip_prefix("without "));

        if let Some(forbidden_noun) = neg_prefix_target {
            let fn_clean = forbidden_noun.trim();
            if (ctx.contains_bounded(fn_clean) || ctx.contains_stem(fn_clean))
                && !ctx.is_negated(fn_clean)
                && !ctx.is_stem_negated(fn_clean)
            {
                findings.push(format!(
                    "[POLICY DEDUCTION BLOCKED]: Action \"{}\" requires \"{}\", which is violated in the state (DEDUCTION_VALID: FALSE, EXCEPTION_ACTIVE: TRUE).",
                    action_name, req_name
                ));
                continue;
            }
        }

        // Case B: Affirmative precondition ("a receipt", "overdue status") whose core noun is negated in the state ("has no receipt")
        let core_words: Vec<&str> = req_name
            .split_whitespace()
            .filter(|w| w.len() >= 4 && *w != "within" && *w != "days")
            .collect();
        if !core_words.is_empty()
            && core_words
                .iter()
                .any(|w| ctx.is_negated(w) || ctx.is_stem_negated(w))
        {
            findings.push(format!(
                "[POLICY DEDUCTION BLOCKED]: Action \"{}\" requires \"{}\", which is negated or missing in the state (DEDUCTION_VALID: FALSE, EXCEPTION_ACTIVE: TRUE).",
                action_name, req_name
            ));
        }
    }

    // Direct inhibition check: if an inhibitor node is active in the premise (outside its own rule definition)
    for target in 0..dag.node_names.len() {
        if dag.is_inhibited(&active_nodes, target) {
            let target_name = &dag.node_names[target];
            findings.push(format!(
                "[POLICY DEDUCTION BLOCKED]: Target \"{}\" is overridden or blocked by an active inhibition rule (DEDUCTION_VALID: FALSE, EXCEPTION_ACTIVE: TRUE).",
                target_name
            ));
        }
    }

    if active_nodes.is_empty() {
        return findings;
    }

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
        assert!(
            dag.is_reachable(a, c),
            "Transitive 2-hop closure must reach C from A"
        );
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
        assert!(deductions
            .iter()
            .any(|d| d.contains("DEDUCTION_VALID: TRUE")));
    }
}
