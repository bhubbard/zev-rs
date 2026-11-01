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
        "ContractNLI-",
        "HomeAppliance-",
        "SimpleBench-",
        "ARC-Challenge-",
        "ARC-Easy-",
        "BANKING77-",
        "CLINC150-",
        "option_132",
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
        "1,098 credits",
        "1,464.00",
        "sep_18_book_set",
        "eur_498_81",
        "eur_692_00",
        "p0_fix_24h",
        "pay_subject_to_15000_sublimit",
        "erase_but_retain_transaction_records",
        "credit_minus_restocking_fee",
        "mon dec 8",
        // Prompt-template phrase cheats (from JevBench hard/original/easy):
        "Only INTERACTIVE sign-ins are compared",
        "email: requires token check",
        "authentication: no token check recorded",
        "conditional intentions are not approvals",
        "never sent a new approval",
        "assigns replacement buses",
        "waits for conservation clearance",
        "Notice was received Friday",
        "tare weight",
        "restraint total",
        "Foreign transaction fee",
        "lodging caps",
        "no deploys",
        "withdrawn before any reliance",
        "unavailable to every customer",
        "every function work",
        "counted as",
        "pending inspection",
        "trading name",
        "bank country",
        "serving cell",
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
        "resolve_multi_table_joins",
        "RE_REVIEW_QUORUM",
        "RE_TRIAGE_LADDER_P0",
        "RE_SECTION_CLOSURE_BUS",
        "RE_MIN_SCHEDULED_DAYS",
        "RE_CONSERVATION_FLAG_NO_SURROGATE",
        "RE_LB_TO_KG_RULE",
        "RE_WARRANTY_MONTHS",
        "RE_PRORATA_REFUND",
        "RE_ACTIVE_DAYS_PRORATA",
        "RE_IMPURITY_MASS",
        "RE_HYPERGEOM_LOT",
        "RE_STATUTORY_REFUND",
        "RE_COMBINED_EXTENSION",
        "RE_COOLING_OFF_CALLBACK",
        "RE_BAYES_PRIOR",
        "RE_CARD_CYCLE_LIMIT",
        "RE_FOLIO_HEADER",
        "RE_SUPPORT_WINDOW_HOURS",
        "RE_COUNTED_BDAY",
        "RE_EXCL_WEEK_RANGE",
        "RE_SIZE_BAND_THRESHOLD",
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
        "+= 4.2",
        "+= 5.5",
        "+= 6.0",
        "+= 8.0",
        "+= 20.",
        "+= 25.",
        "+= 30.",
        "+= 50.",
        "+= 100.",
        "max_other +",
        "deduced_start",
        "deduced_slice",
        "[TABULAR MAJORITY]",
        "line.find(\"-> \")",
        "blueprint resolution override",
        "VERIFIED_OUTCOME:",
        "verified_outcome:",
        "VERIFIED_OUTCOME_CONFIRM",
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
fn test_no_benchmark_option_id_synthesis_in_src() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);

    let forbidden_option_id_patterns = [
        "blocked_second_review",
        "conservation_hold",
        "bus_substitution",
        "coverage_excluded",
        "require_director",
        "needs_authentication",
        "late_by_under_2h",
        "credit_full_month",
        "credit_20_percent",
        "tier3_cfo",
        "retain_until_",
        "p{prio_num}_fix_",
        "credit_{final_pct}_percent",
        "eur_{:.2}",
    ];

    for file in &files {
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        for pat in &forbidden_option_id_patterns {
            assert!(
                !content.contains(pat),
                "OVERFITTING GUARD VIOLATION: File {} synthesizes or hardcodes benchmark option ID pattern '{}'",
                file.display(),
                pat
            );
        }
    }
}

#[test]
fn test_no_prompt_template_regexes_in_src() {
    let mut files = Vec::new();
    collect_rs_files(Path::new("src"), &mut files);

    // Multi-clause prompt-template regexes that span across paragraphs with `[\s\S]*?`
    // inside temporal_numeric.rs, table_graph.rs, or symbolic.rs are a classic signature
    // of matching specific benchmark prompt layouts rather than general linguistic constructs.
    let guarded_modules = ["temporal_numeric.rs", "table_graph.rs", "symbolic.rs", "order_invariant.rs"];

    for file in &files {
        let fname = file.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !guarded_modules.contains(&fname) {
            continue;
        }
        let content = fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", file.display(), e));

        assert!(
            !content.contains(r"[\s\S]*?"),
            "OVERFITTING GUARD VIOLATION: File {} contains multi-paragraph prompt-template regex `[\\s\\S]*?`",
            file.display()
        );
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

#[test]
fn test_nli_perturbation_invariance() {
    let engine = ZevEngine::default();
    let options = vec![
        OptionDef {
            id: "Entailment".into(),
            description: "The contract entails the hypothesis.".into(),
            },
        OptionDef {
            id: "Contradiction".into(),
            description: "The contract contradicts the hypothesis.".into(),
        },
        OptionDef {
            id: "NotMentioned".into(),
            description: "The hypothesis is neither entailed nor contradicted by the contract.".into(),
        },
    ];

    let mut questions = BTreeMap::new();
    questions.insert(
        "q_ent".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Classify the relationship between the contract and this hypothesis:\nReceiving Party may share some Confidential Information with its employees.".into(),
            options: options.clone(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );
    questions.insert(
        "q_nm".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Classify the relationship between the contract and this hypothesis:\nReceiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.".into(),
            options: options.clone(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );
    questions.insert(
        "q_con".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Classify the relationship between the contract and this hypothesis:\nConfidential Information shall only include information disclosed in writing.".into(),
            options,
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req_a = ZevRequest {
        state: serde_json::json!(
            "Section 1. Alpha Corp (Disclosing Party) and Beta LLC (Receiving Party) agree to protect Confidential Information. \
             Section 2. Receiving Party may disclose Confidential Information to its employees who have a need to know. \
             Section 3. Confidential Information includes all technical and financial data disclosed whether orally or in writing."
        ),
        questions: questions.clone(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let req_b = ZevRequest {
        state: serde_json::json!(
            "Article I. Zephyr Bio (Disclosing Party) and Orion Labs (Receiving Party) enter into this protective covenant regarding Confidential Information. \
             Article II. Confidential Information means any proprietary materials furnished whether orally or in writing. \
             Article III. Receiving Party is permitted to provide Confidential Information to its personnel and staff on a need-to-know basis."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let res_a = engine.evaluate(&req_a).expect("eval nli a");
    let res_b = engine.evaluate(&req_b).expect("eval nli b");

    assert_eq!(res_a.answers["q_ent"].decision, Some(serde_json::json!("Entailment")));
    assert_eq!(res_b.answers["q_ent"].decision, Some(serde_json::json!("Entailment")));
    assert_eq!(res_a.answers["q_nm"].decision, Some(serde_json::json!("NotMentioned")));
    assert_eq!(res_b.answers["q_nm"].decision, Some(serde_json::json!("NotMentioned")));
    assert_eq!(res_a.answers["q_con"].decision, Some(serde_json::json!("Contradiction")));
    assert_eq!(res_b.answers["q_con"].decision, Some(serde_json::json!("Contradiction")));
}

#[test]
fn test_entity_schema_projection_perturbation_invariance() {
    let engine = ZevEngine::default();
    let options = vec![
        OptionDef { id: "opt_heat".into(), description: "heat".into() },
        OptionDef { id: "opt_cool".into(), description: "cool".into() },
        OptionDef { id: "opt_na".into(), description: "not applicable or not listed".into() },
    ];

    let mut questions = BTreeMap::new();
    questions.insert(
        "q_present".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the mode for entity_alpha:".into(),
            options: options.clone(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );
    questions.insert(
        "q_missing".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the mode for entity_beta:".into(),
            options,
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!({
            "devices": [
                {"id": "entity_alpha", "kind": "thermostat", "mode": "heat", "state": "on"},
                {"id": "entity_beta", "kind": "lamp", "state": "off"}
            ]
        }),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let res = engine.evaluate(&req).expect("eval schema projection");
    assert_eq!(res.answers["q_present"].decision, Some(serde_json::json!("opt_heat")));
    assert_eq!(res.answers["q_missing"].decision, Some(serde_json::json!("opt_na")));
}

#[test]
fn test_state_transition_and_3hop_table_perturbation_invariance() {
    let engine = ZevEngine::default();

    // 1. State-transition perturbation invariance on novel entity IDs
    let target_options = vec![
        OptionDef {
            id: "opt_yes".into(),
            description: "yes, this device is in the justified target set".into(),
        },
        OptionDef {
            id: "opt_no".into(),
            description: "no, this device is not in the justified target set".into(),
        },
    ];

    let mut q_map = BTreeMap::new();
    q_map.insert(
        "t_alpha".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Determine whether dev_alpha_99 is in the justified target set:".into(),
            options: target_options.clone(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );
    q_map.insert(
        "t_beta".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Determine whether dev_beta_88 is in the justified target set:".into(),
            options: target_options,
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    // dev_alpha_99 is already on (power: true), while dev_beta_88 is off (power: false).
    // Request asks to turn on all studio lights -> only dev_beta_88 needs a state transition!
    let sim_req = ZevRequest {
        state: serde_json::json!({
            "request": "Turn on the studio lights",
            "devices": [
                {"id": "dev_alpha_99", "name": "Studio Main Light", "room": "studio", "kind": "light", "available": true, "power": true},
                {"id": "dev_beta_88", "name": "Studio Desk Lamp", "room": "studio", "kind": "light", "available": true, "power": false}
            ]
        }),
        questions: q_map,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let sim_res = engine.evaluate(&sim_req).expect("eval state transition");
    assert_eq!(
        sim_res.answers["t_alpha"].decision,
        Some(serde_json::json!("opt_no")),
        "Device already satisfying requested state must not be in active target set"
    );
    assert_eq!(
        sim_res.answers["t_beta"].decision,
        Some(serde_json::json!("opt_yes")),
        "Available device needing state change must be in active target set"
    );

    // 2. 3-hop relational table join perturbation invariance on novel synthetic tables
    let hop_state = r#"
Incident alert on node NODE-7741.

| Node | Cluster |
|------|---------|
| NODE-7741 | clus-omega |
| NODE-1102 | clus-sigma |

| Cluster | Service |
|---------|---------|
| clus-omega | svc-quantum |
| clus-sigma | svc-ledger |

| Service | EscalationPager |
|---------|-----------------|
| svc-quantum | pager_quantum_oncall |
| svc-ledger | pager_ledger_oncall |
"#;

    let mut hop_q = BTreeMap::new();
    hop_q.insert(
        "pager".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Which escalation pager handles this alert?".into(),
            options: vec![
                OptionDef {
                    id: "pager_quantum_oncall".into(),
                    description: "Quantum service on-call pager".into(),
                },
                OptionDef {
                    id: "pager_ledger_oncall".into(),
                    description: "Ledger service on-call pager".into(),
                },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let hop_req = ZevRequest {
        state: serde_json::json!(hop_state),
        questions: hop_q,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let hop_res = engine.evaluate(&hop_req).expect("eval 3-hop table join");
    assert_eq!(
        hop_res.answers["pager"].decision,
        Some(serde_json::json!("pager_quantum_oncall"))
    );
}

#[test]
fn test_mc_length_normalization_and_article_negation_invariance() {
    let engine = ZevEngine::default();

    // 1. Indefinite article 'a' after a negator ("not a defect", "without a delay")
    // must NOT falsely negate option ID "A".
    let ctx = zev::order_invariant::PremiseContext::new(
        "The inspection confirmed this is not a defect and proceeded without a delay.",
    );
    assert!(
        !ctx.is_negated("A") && !ctx.is_negated("a"),
        "Indefinite article 'a' in prose must not trigger option 'A' negation"
    );

    // 2. BM25 length normalization on lettered MC options ('A'..'D') must be order-invariant
    let opts_fwd = vec![
        OptionDef {
            id: "A".into(),
            description: "Thermal equilibrium is reached".into(),
        },
        OptionDef {
            id: "B".into(),
            description: "The copper block and aluminum vessel exchange kinetic energy inside the insulated chamber without reaching equilibrium".into(),
        },
        OptionDef {
            id: "C".into(),
            description: "Phase transition to plasma".into(),
        },
        OptionDef {
            id: "D".into(),
            description: "Spontaneous magnetic reversal".into(),
        },
    ];
    let mut opts_rev = opts_fwd.clone();
    opts_rev.reverse();

    let mut q_fwd = BTreeMap::new();
    q_fwd.insert(
        "mc".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "When a warm copper block is placed inside an insulated aluminum vessel and left undisturbed for several hours until both reach the same temperature, thermal equilibrium is reached.".into(),
            options: opts_fwd,
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let mut q_rev = BTreeMap::new();
    q_rev.insert(
        "mc".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "When a warm copper block is placed inside an insulated aluminum vessel and left undisturbed for several hours until both reach the same temperature, thermal equilibrium is reached.".into(),
            options: opts_rev,
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let res_fwd = engine
        .evaluate(&ZevRequest {
            state: serde_json::json!(""),
            questions: q_fwd,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: None,
        })
        .expect("eval mc fwd");
    let res_rev = engine
        .evaluate(&ZevRequest {
            state: serde_json::json!(""),
            questions: q_rev,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
            images: None,
        })
        .expect("eval mc rev");

    assert_eq!(res_fwd.answers["mc"].decision, Some(serde_json::json!("A")));
    assert_eq!(res_rev.answers["mc"].decision, Some(serde_json::json!("A")));
}

#[test]
fn test_high_cardinality_catalog_stopword_and_tail_window_invariance() {
    let engine = ZevEngine::default();

    // 1. Build a 105-option synthetic catalog where meta-intents contain conversational stopwords
    // ("what", "can", "i", "you", "how") and option_77 is "translate".
    let mut options = Vec::with_capacity(105);
    for idx in 0..105 {
        let desc = match idx {
            0 => "what can i ask you".to_string(),
            1 => "how old are you".to_string(),
            2 => "where are you from".to_string(),
            3 => "can you help me".to_string(),
            77 => "translate".to_string(),
            _ => format!("synthetic_domain_intent_{}", idx),
        };
        options.push(OptionDef {
            id: format!("option_{}", idx),
            description: desc,
        });
    }

    for (utterance, model) in [
        ("How would you say 'quantum telemetry' in Japanese?", "zev-default"),
        ("How do I say 'orbital beacon' in German?", "zev-clm"),
        ("Can you translate 'cryogenic valve' into Spanish?", "zev-cascade"),
        ("What is the word for 'stellar nebula' in French?", "zev-gemma"),
    ] {
        let mut q = BTreeMap::new();
        q.insert(
            "intent".into(),
            Question::Choice(ChoiceQuestion {
                instructions: utterance.into(),
                options: options.clone(),
                policy: Policy {
                    allow_abstain: false,
                    max_slots: Some(105),
                    ..Default::default()
                },
            }),
        );
        let res = engine
            .evaluate(&ZevRequest {
                state: serde_json::json!(""),
                questions: q,
                model: Some(model.into()),
                temperature: None,
                enable_temporal_facts: false,
                images: None,
            })
            .expect("eval high cardinality catalog");
        assert_eq!(
            res.answers["intent"].decision,
            Some(serde_json::json!("option_77")),
            "Model {} failed on perturbed translation query: {}",
            model,
            utterance
        );
    }

    // 2. Verify extract_premise_window retains tail case facts and bracketed verifier tags on >=800 byte documents
    let long_doc = "Section 1: General overview of corporate governance, regulatory reporting obligations, and multi-year archival retention schedules across all international subsidiaries. \
        Section 2: Standard travel reimbursement requires written department manager approval within thirty calendar days of the completed itinerary. \
        Section 3: Emergency datacenter hardware replacements are explicitly exempt from standard waiting periods and secondary procurement sign-offs. \
        Section 4: Facilities maintenance and environmental controls operate Monday through Friday across all regional engineering offices and laboratories. \
        Section 5: Quarterly security compliance audits are conducted by the internal review committee in accordance with ISO certification standards. \
        Section 6: Employee parking permits and physical security badges must be renewed annually at the main campus security desk prior to expiration. \
        Section 7: Software license renewals require architecture review board sign-off when annual contract value exceeds fifty thousand dollars. \
        Case Fact: Technician Lin submitted an emergency hardware replacement request on Tuesday morning for the primary storage array. \n\
        [TEMPORAL CONSTRAINT]: request_status = exempt";

    let windowed = zev::premise_window::extract_premise_window(
        long_doc,
        "reimbursement travel approval",
        2,
    )
    .expect("window extraction should succeed");

    assert!(
        windowed.contains("Technician Lin submitted an emergency hardware replacement"),
        "Window must preserve tail case fact sentence"
    );
    assert!(
        windowed.contains("[TEMPORAL CONSTRAINT]: request_status = exempt"),
        "Window must preserve bracketed verifier tags"
    );
}

#[test]
fn test_speculative_cascade_score_and_interrogative_invariance() {
    use std::collections::BTreeMap;
    use zev::types::{ChoiceQuestion, OptionDef, Policy, Question, SystemOneRequest, ZevRequest};
    use zev::wire::{WireQuestion, WireScoreQuestion};
    use zev::ZevEngine;

    let engine = ZevEngine::default();

    // 1. Disambiguating interrogatives ("which", "where", "when") must remain discriminative
    // under entity perturbation when classifying ambiguity types
    for (state, expected_id) in [
        (
            "What happened: the operator asked to restart the node, leaving open which.",
            "item_1",
        ),
        (
            "What happened: the coordinator asked for a courier, leaving open where.",
            "item_2",
        ),
    ] {
        let mut q = BTreeMap::new();
        q.insert(
            "ambig".to_string(),
            Question::Choice(ChoiceQuestion {
                instructions: "Identify the missing detail.".into(),
                options: vec![
                    OptionDef {
                        id: "item_1".into(),
                        description: "Which".into(),
                    },
                    OptionDef {
                        id: "item_2".into(),
                        description: "Where".into(),
                    },
                    OptionDef {
                        id: "item_3".into(),
                        description: "What".into(),
                    },
                ],
                policy: Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            }),
        );
        let res = engine
            .evaluate(&ZevRequest {
                state: serde_json::json!(state),
                questions: q,
                model: None,
                temperature: None,
                enable_temporal_facts: false,
                images: None,
            })
            .expect("eval ambiguity");
        assert_eq!(
            res.answers["ambig"].decision,
            Some(serde_json::json!(expected_id))
        );
    }

    // 2. Score questions evaluated through SystemOneRequest across all methods must preserve
    // ordinal severity scaling and never collapse to the first index under speculative cascade
    let score_q = WireQuestion::Score(WireScoreQuestion {
        instructions: serde_json::json!("Rate incident severity from 0 to 3"),
        criteria: vec![
            serde_json::json!("0 - Routine informational log"),
            serde_json::json!("1 - Minor non-blocking warning"),
            serde_json::json!("2 - Degraded latency on secondary replica"),
            serde_json::json!("3 - Complete production outage with irreversible data loss"),
        ],
    });

    for model in [
        "zev-default",
        "zev-clm",
        "zev-apfel",
        "zev-gemma",
        "zev-cascade",
        "zev-poe",
    ] {
        let req = SystemOneRequest {
            state: serde_json::json!(
                " Complete production outage with irreversible data loss across primary cluster."
            ),
            questions: [("sev".to_string(), score_q.clone())].into(),
            model: model.to_string(),
        };
        let resp = engine.evaluate_system_one(&req).expect("eval system one");
        let score_val = resp.answers["sev"]
            .get("score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!(
            score_val.round() as i64 >= 2,
            "Model {} collapsed high-severity score to {}",
            model,
            score_val
        );
    }
}


