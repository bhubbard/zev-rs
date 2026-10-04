//! Anti-Overfitting and Benchmark Hygiene Guard
//!
//! Enforces zero-tolerance against hardcoded benchmark overrides, test-set memorization,
//! and artificial logit manipulation in the production engine and preprocessor.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use zev::{ChoiceQuestion, OptionDef, Policy, Question, ZevEngine, ZevRequest};

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, files);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
}

#[test]
fn test_no_benchmark_case_ids_in_src() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);
    assert!(!files.is_empty(), "src directory must contain .rs files");

    let forbidden_case_id_patterns = [
        "hard-opus",
        "hard-sol",
        "WorkflowEval-",
        "case_hard_",
        "opus-a-",
        "opus-b-",
        "opus-c-",
        "sol-a-",
        "sol-b-",
        "sol-c-",
    ];

    for file in &files {
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        for pat in &forbidden_case_id_patterns {
            assert!(
                !content.contains(pat),
                "OVERFITTING GUARD VIOLATION: File {} contains forbidden benchmark test-case ID '{}'",
                file.display(),
                pat
            );
        }
    }
}

#[test]
fn test_no_benchmark_entity_memorization_in_src() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);
    assert!(!files.is_empty(), "src directory must contain .rs files");

    // Dataset-specific entities and benchmark phrases from JevBench, SimpleBench, and
    // other suites that should NEVER appear as hardcoded strings in general engine logic.
    let forbidden_dataset_entities = [
        "ALDERMOOR",
        "NS-2026-131",
        "WL-4471902",
        "WESERLINK",
        "Train MV-184",
        "CLM-6081",
        "V-EMBER",
        "CASTELLAN FOODS",
        "VELANT OPTICS",
        "MERIDIAN CARD SERVICES",
        "WAYFARER ASSURANCE",
        "PELAGOS STREAMING",
        "BRIGHTWATER OUTDOOR",
        "RQ-26-09-3318",
        "KESTREL MARINE",
        "LATTICEPAY",
        "Vantorre Landscaping",
        "Leonie Marsh",
        "Shiba Park Tower",
        "TACROVEX",
        "HALVERSTON COLLEGE",
        "KORRIDAN ELECTRONICS",
        "OSTERLAND GARDEN",
        "CLOUDMERE SRE",
        "ELMBROOK FAMILY",
        "LUMEN SOCIAL",
        "BRAEMONT PUMPS",
        "Kuznets Technik",
        "HARTWELL SECONDARY",
        "inject_blueprint_knowledge",
        // SimpleBench cheats:
        "ice cubes in a frying pan",
        "diverts up the stairs",
        "global nuclear war",
        // JevBench phrase cheats:
        "load rule lr-7",
        "closed on sunday",
        "replacing the earlier courier",
        "send it to my new office instead",
        "pet dragon",
        "failing parser",
        "standalone python",
        "reschedule my meeting",
        "attached contract",
        "daily export",
        "irreversibly deleted",
        "cannot sign in",
        "nonessential function impaired",
        "every function works",
        "thanks for explaining",
        "understand the policy",
        "proof is absent",
        "dispute is open",
        "suspension blocks",
        "only red",
        "stop renewing",
        "cancelled yesterday",
        "depot pickup",
        "did not change the booking",
    ];

    for file in &files {
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        for entity in &forbidden_dataset_entities {
            assert!(
                !content.contains(entity),
                "OVERFITTING GUARD VIOLATION: File {} contains hardcoded benchmark entity/phrase '{}'",
                file.display(),
                entity
            );
        }
    }
}

#[test]
fn test_no_cheat_functions_in_src() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);
    assert!(!files.is_empty(), "src directory must contain .rs files");

    let forbidden_function_names = [
        "evaluate_ordinal_severity_ladder",
        "detect_confirmed_delivery_extraction",
        "detect_customer_intent_action",
        "evaluate_cumulative_budget_alert",
        "boost_routing_specialist_associations",
        "detect_constraint_violation",
        "detect_policy_precondition_violation",
        "apply_mention_vs_request_intent_filter",
        "inject_blueprint_knowledge",
    ];

    for file in &files {
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        for fn_name in &forbidden_function_names {
            assert!(
                !content.contains(fn_name),
                "OVERFITTING GUARD VIOLATION: File {} contains hardcoded benchmark cheat function '{}'",
                file.display(),
                fn_name
            );
        }
    }
}

#[test]
fn test_no_artificial_logit_boost_bypasses() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);

    let suspicious_patterns = [
        "+= 6.0",
        "+= 8.0",
        "+= 20.",
        "+= 25.",
        "+= 30.",
        "+= 50.",
        "+= 100.",
        "line.find(\"-> \")",
        "blueprint resolution override",
    ];

    for file in &files {
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        for pat in &suspicious_patterns {
            assert!(
                !content.contains(pat),
                "OVERFITTING GUARD VIOLATION: File {} contains artificial logit boost bypass pattern '{}'",
                file.display(),
                pat
            );
        }
    }
}

#[test]
fn test_entity_perturbation_robustness() {
    let engine = ZevEngine::default();

    let options = vec![
        OptionDef {
            id: "billing".into(),
            description: "Payment processing, invoice inquiry, and refund dispute".into(),
        },
        OptionDef {
            id: "tech_support".into(),
            description: "Server errors, bugs, and API downtime".into(),
        },
        OptionDef {
            id: "sales".into(),
            description: "New enterprise contracts, plan upgrades, and pricing".into(),
        },
    ];

    let question = Question::Choice(ChoiceQuestion {
        instructions: "Categorize the incoming customer ticket".into(),
        options,
        policy: Policy {
            allow_abstain: false,
            ..Default::default()
        },
    });

    let mut questions = BTreeMap::new();
    questions.insert("ticket_type".into(), question);

    // Scenario A: Original entity
    let req_a = ZevRequest {
        state: serde_json::json!("Customer John Doe from Acme Corp called: our latest invoice shows a charge for $450 that we disputed last week. Need an immediate refund."),
        questions: questions.clone(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    // Scenario B: Perturbed entities (different customer, company, amount)
    let req_b = ZevRequest {
        state: serde_json::json!("Customer Sarah Connor from Cyberdyne Systems called: our latest invoice shows a charge for $920 that we disputed last week. Need an immediate refund."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let res_a = engine.evaluate(&req_a).expect("eval a");
    let res_b = engine.evaluate(&req_b).expect("eval b");

    let ans_a = &res_a.answers["ticket_type"];
    let ans_b = &res_b.answers["ticket_type"];

    // Both should yield consistent decision direction (billing)
    assert_eq!(ans_a.decision, ans_b.decision, "Decisions must remain invariant under entity perturbations");
    assert_eq!(
        ans_a.decision,
        Some(serde_json::Value::String("billing".into()))
    );
    let prob_diff = (ans_a.probabilities["billing"] - ans_b.probabilities["billing"]).abs();
    assert!(prob_diff < 0.20, "Confidence spread under entity perturbation should be reasonable, got diff: {}", prob_diff);
}
