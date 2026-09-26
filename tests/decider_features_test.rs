use std::collections::{BTreeMap, HashMap};
use zev::{
    fit_temperatures_by_type, generate_candidates, is_abstain_text, BooleanQuestion,
    ChoiceQuestion, OptionDef, Policy, Question, ScoreQuestion, TypeTemperatureConfig, ZevEngine,
    ZevRequest,
};

#[test]
fn test_per_type_temperature_calibration_resolution() {
    let custom_config = TypeTemperatureConfig::new(1.35, 2.10, 1.28, 1.15);
    let engine = ZevEngine::default().with_type_temperatures(custom_config);

    let state = "Customer reports an intermittent 502 bad gateway on checkout service";

    let mut questions = BTreeMap::new();
    questions.insert(
        "dept".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select responsible team".into(),
            options: vec![
                OptionDef {
                    id: "infra".into(),
                    description: "Infrastructure and gateway network issues".into(),
                },
                OptionDef {
                    id: "billing".into(),
                    description: "Payment disputes and invoices".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );
    questions.insert(
        "is_outage".into(),
        Question::Boolean(BooleanQuestion {
            instructions: "Is this an outage incident?".into(),
            true_description: "Yes, service interruption".into(),
            false_description: "No, normal operations".into(),
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );
    questions.insert(
        "severity".into(),
        Question::Score(ScoreQuestion {
            instructions: "Rate severity from minor to critical".into(),
            levels: vec![
                "P3 - minor performance hiccup".into(),
                "P2 - degraded checkout gateway".into(),
                "P1 - total site outage".into(),
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.evaluate(&req).unwrap();

    let ans_dept = resp.answers.get("dept").unwrap();
    let ans_outage = resp.answers.get("is_outage").unwrap();
    let ans_sev = resp.answers.get("severity").unwrap();

    // Verify each question type was evaluated using its distinct per-type calibrated temperature
    assert!(ans_dept.temperature > 0.0);
    assert!(ans_outage.temperature > 0.0);
    assert!(ans_sev.temperature > 0.0);

    // Boolean temperature is higher than Choice and Score temperature
    assert!(
        ans_outage.temperature > ans_dept.temperature,
        "Boolean temp ({}) should exceed Choice temp ({})",
        ans_outage.temperature,
        ans_dept.temperature
    );

    // Global override test: passing temperature: Some(1.75) sets all questions to 1.75 base
    let req_override = ZevRequest {
        state: serde_json::json!(state),
        questions: {
            let mut q = BTreeMap::new();
            q.insert(
                "dept".into(),
                Question::Choice(ChoiceQuestion {
                    instructions: "Select responsible team".into(),
                    options: vec![
                        OptionDef {
                            id: "infra".into(),
                            description: "Infrastructure and gateway network issues".into(),
                        },
                        OptionDef {
                            id: "billing".into(),
                            description: "Payment disputes and invoices".into(),
                        },
                    ],
                    policy: Policy {
                        allow_abstain: false,
                        ..Default::default()
                    },
                }),
            );
            q
        },
        model: None,
        temperature: Some(1.75),
        enable_temporal_facts: false,
    };
    let resp_override = engine.evaluate(&req_override).unwrap();
    let ans_override = resp_override.answers.get("dept").unwrap();
    // 1.75 * family (intent 0.90) = 1.575 (dampened if margin applies)
    assert!(
        ans_override.temperature >= 1.50 && ans_override.temperature <= 2.50,
        "Overridden temperature should reflect user-passed 1.75 base"
    );
}

#[test]
fn test_fit_temperatures_by_type_optimizer() {
    let mut data = HashMap::new();
    data.insert(
        "choice".to_string(),
        vec![
            (vec![3.2, 0.4, 0.1], 0),
            (vec![0.2, 2.9, 0.8], 1),
            (vec![0.1, 0.5, 3.0], 2),
        ],
    );
    data.insert(
        "boolean".to_string(),
        vec![(vec![2.5, 0.3], 0), (vec![0.1, 2.8], 1)],
    );

    let fitted = fit_temperatures_by_type(&data, 0.5, 4.0, 20);
    assert!(fitted.choice > 0.5 && fitted.choice < 4.0);
    assert!(fitted.boolean > 0.5 && fitted.boolean < 4.0);
    // Unprovided types retain defaults
    assert_eq!(fitted.score, 1.38);
    assert_eq!(fitted.numeric, 1.25);
}

#[test]
fn test_semantic_abstention_matcher_heuristics() {
    // Exact abstentions
    assert!(is_abstain_text("none"));
    assert!(is_abstain_text("other"));
    assert!(is_abstain_text("other / not covered"));
    assert!(is_abstain_text("unsure"));
    assert!(is_abstain_text("neither"));

    // Prefix abstentions
    assert!(is_abstain_text("None of the above"));
    assert!(is_abstain_text("none of these options"));
    assert!(is_abstain_text("not listed in the categories"));
    assert!(is_abstain_text("does not apply to this context"));
    assert!(is_abstain_text("cannot tell from available evidence"));
    assert!(is_abstain_text("insufficient information provided"));

    // Non-abstention queries
    assert!(!is_abstain_text("payment error"));
    assert!(!is_abstain_text("network timeout"));
    assert!(!is_abstain_text("none-too-pleased customer")); // false prefix edge case
}

#[test]
fn test_natural_language_abstention_option_deduplication() {
    let q_with_natural_abstain = Question::Choice(ChoiceQuestion {
        instructions: "Pick category".into(),
        options: vec![
            OptionDef {
                id: "billing".into(),
                description: "Invoicing and payment issues".into(),
            },
            OptionDef {
                id: "none".into(),
                description: "None of the above options".into(),
            },
        ],
        policy: Policy {
            allow_abstain: true,
            ..Default::default()
        },
    });

    let candidates = generate_candidates(&q_with_natural_abstain);
    // Should NOT contain synthetic __insufficient__ candidate because "None of the above" is present
    assert_eq!(candidates.len(), 2);
    assert!(candidates.iter().any(|c| c.id == "none"));
    assert!(!candidates.iter().any(|c| c.id == zev::UNKNOWN));

    let q_without_natural_abstain = Question::Choice(ChoiceQuestion {
        instructions: "Pick category".into(),
        options: vec![
            OptionDef {
                id: "billing".into(),
                description: "Invoicing and payment issues".into(),
            },
            OptionDef {
                id: "support".into(),
                description: "Technical helpdesk".into(),
            },
        ],
        policy: Policy {
            allow_abstain: true,
            ..Default::default()
        },
    });

    let candidates2 = generate_candidates(&q_without_natural_abstain);
    // Should contain synthetic __insufficient__ candidate
    assert_eq!(candidates2.len(), 3);
    assert!(candidates2.iter().any(|c| c.id == zev::UNKNOWN));
}

#[test]
fn test_natural_language_abstention_routing_and_decision_preservation() {
    let engine = ZevEngine::default();
    let state = "The user is asking about the recipe for chocolate cake.";

    let q = Question::Choice(ChoiceQuestion {
        instructions: "Route this IT helpdesk inquiry".into(),
        options: vec![
            OptionDef {
                id: "password_reset".into(),
                description: "Resetting corporate Active Directory password".into(),
            },
            OptionDef {
                id: "vpn_trouble".into(),
                description: "Corporate Cisco VPN tunnel connection failure".into(),
            },
            OptionDef {
                id: "none_of_above".into(),
                description: "None of the above: request is unrelated to IT helpdesk".into(),
            },
        ],
        policy: Policy {
            allow_abstain: true,
            max_unavailable_probability: 0.30,
            ..Default::default()
        },
    });

    let mut map = BTreeMap::new();
    map.insert("route".into(), q);

    let resp = engine
        .evaluate(&ZevRequest {
            state: serde_json::json!(state),
            questions: map,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
        })
        .unwrap();

    let ans = resp.answers.get("route").unwrap();
    // Model correctly selected the natural abstention option
    assert_eq!(ans.status, "insufficient_evidence");
    // Decision truthfully preserves the chosen user option ID "none_of_above"
    assert_eq!(
        ans.decision,
        Some(serde_json::Value::String("none_of_above".into()))
    );
}
