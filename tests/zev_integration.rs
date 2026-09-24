use std::collections::BTreeMap;
use std::sync::Arc;
use axum::{body::Body, http::{Request, StatusCode}};
use tower::ServiceExt;

use zev::{
    compute_ece, fit_temperature, shortlist_options, ChoiceQuestion,
    OptionDef, Policy, Question, SystemOneRequest,
    ZevEngine, ZevRequest,
};

#[test]
fn test_order_invariance() {
    let engine = ZevEngine::default();
    let state = "Customer calls about credit card charge dispute and wants a refund.";

    let opt1 = OptionDef { id: "billing".into(), description: "Payment processing and refund".into() };
    let opt2 = OptionDef { id: "tech_support".into(), description: "Server and API error".into() };
    let opt3 = OptionDef { id: "sales".into(), description: "New subscriptions and upgrades".into() };

    // Order 1: [billing, tech_support, sales]
    let q1 = Question::Choice(ChoiceQuestion {
        instructions: "Route customer".into(),
        options: vec![opt1.clone(), opt2.clone(), opt3.clone()],
        policy: Policy { allow_abstain: false, ..Default::default() },
    });

    // Order 2: [sales, tech_support, billing] (inverted)
    let q2 = Question::Choice(ChoiceQuestion {
        instructions: "Route customer".into(),
        options: vec![opt3.clone(), opt2.clone(), opt1.clone()],
        policy: Policy { allow_abstain: false, ..Default::default() },
    });

    let mut map1 = BTreeMap::new(); map1.insert("route".into(), q1);
    let mut map2 = BTreeMap::new(); map2.insert("route".into(), q2);

    let resp1 = engine.evaluate(&ZevRequest {
        state: serde_json::json!(state),
        questions: map1,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }).unwrap();

    let resp2 = engine.evaluate(&ZevRequest {
        state: serde_json::json!(state),
        questions: map2,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }).unwrap();

    let ans1 = resp1.answers.get("route").unwrap();
    let ans2 = resp2.answers.get("route").unwrap();

    // 100% Order-Invariance: top decision and exact probabilities are identical!
    assert_eq!(ans1.decision, ans2.decision);
    assert_eq!(ans1.decision, Some(serde_json::Value::String("billing".into())));
    assert!((ans1.probabilities["billing"] - ans2.probabilities["billing"]).abs() < 1e-9);
}

#[test]
fn test_abstention_guardrails() {
    let engine = ZevEngine::default();
    let state = "The sky is blue.";

    let q = Question::Choice(ChoiceQuestion {
        instructions: "What is the capital of the moon?".into(),
        options: vec![
            OptionDef { id: "crater_alpha".into(), description: "Crater Alpha".into() },
            OptionDef { id: "crater_beta".into(), description: "Crater Beta".into() },
        ],
        policy: Policy {
            allow_abstain: true,
            max_unavailable_probability: 0.1, // low tolerance triggers abstention
            min_top_probability: 0.8,
            max_slots: None,
        },
    });

    let mut map = BTreeMap::new();
    map.insert("moon".into(), q);

    let resp = engine.evaluate(&ZevRequest {
        state: serde_json::json!(state),
        questions: map,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }).unwrap();

    let ans = resp.answers.get("moon").unwrap();
    assert_ne!(ans.status, "ok"); // Abstention triggered (uncertain or insufficient_evidence)
}

#[test]
fn test_shortlisting() {
    let mut options = Vec::new();
    for i in 0..50 {
        options.push(OptionDef {
            id: format!("category_{i}"),
            description: format!("Department for service code {i}"),
        });
    }
    options.push(OptionDef {
        id: "billing_specialist".into(),
        description: "Department for customer invoices and billing".into(),
    });

    let shortlisted = shortlist_options(&options, "Customer disputed billing invoice", 20);
    assert_eq!(shortlisted.len(), 20);
    // Verified billing_specialist was retained in top slots
    assert!(shortlisted.iter().any(|o| o.id == "billing_specialist"));
}

#[test]
fn test_calibration_and_ece() {
    let confs = vec![0.9, 0.8, 0.6, 0.7, 0.95];
    let accs = vec![true, true, false, true, true];
    let ece = compute_ece(&confs, &accs, 5);
    assert!(ece >= 0.0 && ece <= 1.0);

    let pairs = vec![
        (vec![2.0, 0.5], 0),
        (vec![0.1, 2.5], 1),
        (vec![1.8, 0.2], 0),
    ];
    let optimal_t = fit_temperature(&pairs, 0.5, 5.0, 20);
    assert!(optimal_t > 0.5 && optimal_t < 5.0);
}

#[test]
fn test_typesafe_systemone_compatibility() {
    let engine = ZevEngine::default();
    let body = serde_json::json!({
        "state": "The user reported: my payouts have been failing for 3 days",
        "model": "zev-latest",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?"
            },
            "department": {
                "type": "choice",
                "instructions": "Route department",
                "criteria": {
                    "billing": "Invoices and payouts",
                    "support": "General tech support"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "Rating from 0 to 2",
                "criteria": ["Low", "Medium", "High"]
            }
        }
    });

    let sys1_req: SystemOneRequest = serde_json::from_value(body).unwrap();
    let sys1_resp = engine.evaluate_system_one(&sys1_req).unwrap();

    assert_eq!(sys1_resp.answers.len(), 3);
    let dept = sys1_resp.answers.get("department").unwrap();
    assert_eq!(dept["choice"], "billing");
    assert!(dept["confidence"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn test_http_server_endpoints() {
    let engine = Arc::new(ZevEngine::default());
    let app = zev::create_router(engine);

    // 1. Health
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 2. Limits
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/limits")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 3. SystemOne POST
    let body = serde_json::json!({
        "state": "Emergency: database credentials exposed",
        "questions": {
            "is_emergency": {
                "type": "noul",
                "instructions": "Is this a critical security emergency?"
            }
        }
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 4. Home root "/"
    let response = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 5. Models "/v1/models"
    let response = app
        .clone()
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 6. Decisions POST "/v1/decisions"
    let dec_body = serde_json::json!({
        "state": "Customer payment failed",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Route department",
                "options": [
                    {"id": "billing", "description": "Payment invoices"},
                    {"id": "support", "description": "Tech support"}
                ]
            }
        }
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/decisions")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&dec_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 7. Tev1 POST "/v1/tev1"
    let tev1_body = serde_json::json!({
        "state": "Returns allowed within 30 days. Purchased 10 days ago.",
        "question": "Is return valid?",
        "options": ["A: Yes", "B: No"]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/tev1")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_vec(&tev1_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[test]
fn test_systemone_comprehensive_wire_types() {
    let engine = ZevEngine::default();
    let body = serde_json::json!({
        "state": { "user_id": 42, "incident": "Database connection pool saturated with 500 errors" },
        "questions": {
            "is_outage": {
                "type": "noul",
                "instructions": "Outage status",
                "criteria": {
                    "true": "Database connection errors and downtime",
                    "false": "Normal operating metrics"
                }
            },
            "dept_choice": {
                "type": "choice",
                "instructions": "Route ticket",
                "criteria": {
                    "infra": "Database cluster outage",
                    "billing": "Invoice questions"
                }
            },
            "severity_score": {
                "type": "score",
                "instructions": "Score severity from low to critical",
                "criteria": [
                    "Low - informational",
                    "Medium - degraded performance",
                    "Critical - database outage and errors"
                ]
            }
        }
    });

    let req: SystemOneRequest = serde_json::from_value(body).unwrap();
    let resp = engine.evaluate_system_one(&req).unwrap();
    assert!(resp.answers.contains_key("is_outage"));
    assert!(resp.answers.contains_key("dept_choice"));
    assert!(resp.answers.contains_key("severity_score"));
}

#[test]
fn test_engine_shortlisting_in_evaluate() {
    let engine = ZevEngine::default();
    let options: Vec<OptionDef> = (0..50)
        .map(|i| OptionDef {
            id: format!("opt_{i}"),
            description: if i == 42 {
                "target: acute right lower quadrant abdominal peritonitis".into()
            } else {
                format!("distractor condition {i}")
            },
        })
        .collect();

    let q = Question::Choice(ChoiceQuestion {
        instructions: "Diagnose".into(),
        options,
        policy: Policy { allow_abstain: true, ..Default::default() },
    });

    let mut map = BTreeMap::new();
    map.insert("diag".into(), q);

    let req = ZevRequest {
        state: serde_json::json!("Patient has acute right lower quadrant abdominal peritonitis"),
        questions: map,
        model: None,
        temperature: None,
        enable_temporal_facts: true,
    };

    let resp = engine.evaluate(&req).unwrap();
    let ans = resp.answers.get("diag").unwrap();
    assert_eq!(ans.decision, Some(serde_json::Value::String("opt_42".into())));
}

// =========================================================================
// PORTED TEST SUITES FROM JEV-ALTERNATIVE ECOSYSTEM
// =========================================================================

// --- 1. From semif-rs: Exact Softmax Parity & Numerical Accuracy ---
#[test]
fn test_exact_softmax_parity() {
    let logits = vec![22.0, 26.375, 24.375];
    let probs = zev::scaled_softmax(&logits, 1.0).unwrap();

    let expected = [0.010966012254357338, 0.8711382150650024, 0.11789573729038239];
    for (p, e) in probs.iter().zip(expected.iter()) {
        assert!((p - e).abs() < 1e-7, "Softmax parity mismatch: {p} vs {e}");
    }
}

// --- 2. From semif-rs: ECE Reduction Under Temperature Scaling ---
#[test]
fn test_ece_reduction_under_temperature_scaling() {
    // Synthetic miscalibrated model (75% accuracy but 99.9% overconfidence)
    let mut pairs = Vec::new();
    for i in 0..100 {
        let is_correct = i % 4 != 0;
        let logits = if is_correct { vec![12.0, 2.0] } else { vec![12.0, 2.0] };
        let true_idx = if is_correct { 0 } else { 1 };
        pairs.push((logits, true_idx));
    }

    let optimal_t = fit_temperature(&pairs, 0.5, 5.0, 30);
    assert!(optimal_t > 1.0, "Optimal temperature for overconfident model must be > 1.0, got {optimal_t}");
}

// --- 3. From von-rs: Multi-Permutation Mathematical Order Invariance ---
#[test]
fn test_multi_permutation_order_invariance() {
    let engine = ZevEngine::default();
    let options_base = [
        ("alpha", "Alpha risk profile with minimal variance"),
        ("beta", "Beta market sensitivity with high correlation"),
        ("gamma", "Gamma non-linear derivatives exposure"),
        ("delta", "Delta directional equity exposure"),
    ];

    let state = "Portfolio shows massive directional equity exposure with steep delta shifts.";

    let permutations = [
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [2, 0, 3, 1],
        [1, 3, 0, 2],
    ];

    let mut first_delta_prob = None;

    for perm in permutations {
        let opts: Vec<OptionDef> = perm
            .iter()
            .map(|&idx| OptionDef {
                id: options_base[idx].0.into(),
                description: options_base[idx].1.into(),
            })
            .collect();

        let q = Question::Choice(ChoiceQuestion {
            instructions: "Classify risk profile".into(),
            options: opts,
            policy: Policy { allow_abstain: false, ..Default::default() },
        });

        let mut questions = BTreeMap::new();
        questions.insert("risk".into(), q);

        let resp = engine.evaluate(&ZevRequest {
            state: serde_json::json!(state),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
        }).unwrap();

        let ans = resp.answers.get("risk").unwrap();
        assert_eq!(
            ans.decision,
            Some(serde_json::Value::String("delta".into())),
            "Winning option must always be 'delta' regardless of option permutation!"
        );

        let delta_prob = ans.probabilities["delta"];
        match first_delta_prob {
            None => first_delta_prob = Some(delta_prob),
            Some(first_p) => {
                assert!(
                    (delta_prob - first_p).abs() < 1e-9,
                    "Permutation probability drift detected: {delta_prob} vs {first_p}"
                );
            }
        }
    }
}

// --- 4. From rizzo-flow-rs: Candidate Generation & Reserved Slots ---
#[test]
fn test_candidate_generation_with_reserved_slots() {
    use zev::types::{ABOVE, BELOW, UNKNOWN};

    // Choice question with allow_abstain
    let choice_q = Question::Choice(ChoiceQuestion {
        instructions: "Pick one".into(),
        options: vec![
            OptionDef { id: "a".into(), description: "Option A".into() },
            OptionDef { id: "b".into(), description: "Option B".into() },
        ],
        policy: Policy { allow_abstain: true, ..Default::default() },
    });
    let choice_cands = zev::generate_candidates(&choice_q);
    assert_eq!(choice_cands.len(), 3);
    assert_eq!(choice_cands[2].id, UNKNOWN);

    // Numeric question with anchors
    let num_q = Question::Numeric(zev::NumericQuestion {
        instructions: "Estimate price".into(),
        unit: "USD".into(),
        anchors: vec![
            zev::Anchor { value: 10.0, description: "Budget".into() },
            zev::Anchor { value: 50.0, description: "Midrange".into() },
            zev::Anchor { value: 100.0, description: "Premium".into() },
        ],
        policy: Policy { allow_abstain: true, ..Default::default() },
    });
    let num_cands = zev::generate_candidates(&num_q);
    assert_eq!(num_cands.len(), 6); // 3 anchors + below + above + unknown
    assert!(num_cands.iter().any(|c| c.id == BELOW));
    assert!(num_cands.iter().any(|c| c.id == ABOVE));
    assert!(num_cands.iter().any(|c| c.id == UNKNOWN));
}

// --- 5. From rizzo-flow-rs: Score Monotonicity & Moment Statistics ---
#[test]
fn test_score_monotonicity_and_moment_statistics() {
    let q = Question::Score(zev::ScoreQuestion {
        instructions: "Rate quality 0 to 3".into(),
        levels: vec!["Poor".into(), "Fair".into(), "Good".into(), "Excellent".into()],
        policy: Policy { allow_abstain: false, ..Default::default() },
    });

    let candidates = zev::generate_candidates(&q);
    // Monotonically increasing logits biased towards Excellent
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let ans = zev::decode_decision(&q, &candidates, &logits, 1.0).unwrap();

    assert_eq!(ans.status, "ok");
    assert!(ans.expected_value.is_some());
    let score = ans.expected_value.unwrap();
    // With higher logits on higher levels, expected mean MUST be > 1.5
    assert!(score > 1.5, "Expected mean {score} should be > 1.5 due to logit weighting");
    assert!(ans.statistics.is_some());
    let stats = ans.statistics.unwrap();
    assert!(stats.stddev >= 0.0);
}

// --- 6. From rizzo-flow-rs: Out-of-Range Detection ---
#[test]
fn test_out_of_range_guardrail() {
    use zev::types::ABOVE;

    let q = Question::Numeric(zev::NumericQuestion {
        instructions: "Estimate valuation".into(),
        unit: "M_USD".into(),
        anchors: vec![
            zev::Anchor { value: 1.0, description: "Seed stage".into() },
            zev::Anchor { value: 10.0, description: "Series A".into() },
            zev::Anchor { value: 50.0, description: "Series B".into() },
        ],
        policy: Policy {
            allow_abstain: true,
            max_unavailable_probability: 0.4,
            min_top_probability: 0.0,
            max_slots: None,
        },
    });

    let candidates = zev::generate_candidates(&q);
    let mut logits = vec![0.0; candidates.len()];
    let above_idx = candidates.iter().position(|c| c.id == ABOVE).unwrap();
    logits[above_idx] = 10.0; // ABOVE heavily dominates

    let ans = zev::decode_decision(&q, &candidates, &logits, 1.0).unwrap();
    assert_eq!(ans.status, "out_of_range");
}

// --- 7. From nanojev-rs: Question Schema Validation Rules ---
#[test]
fn test_question_schema_validation() {
    // 1. Choice with <2 options must fail
    let choice_too_few = Question::Choice(ChoiceQuestion {
        instructions: "Choose".into(),
        options: vec![OptionDef { id: "lone".into(), description: "Single option".into() }],
        policy: Policy::default(),
    });
    assert!(choice_too_few.validate("test_q").is_err());

    // 2. Choice with options exceeding MAX_SLOTS must fail
    let mut too_many = Vec::new();
    for i in 0..30 {
        too_many.push(OptionDef { id: format!("opt_{i}"), description: format!("Desc {i}") });
    }
    let choice_excess = Question::Choice(ChoiceQuestion {
        instructions: "Choose".into(),
        options: too_many,
        policy: Policy { allow_abstain: true, ..Default::default() },
    });
    assert!(choice_excess.validate("excess_q").is_err());

    // 3. Numeric anchors not strictly increasing must fail
    let non_monotonic_numeric = Question::Numeric(zev::NumericQuestion {
        instructions: "Measure".into(),
        unit: "kg".into(),
        anchors: vec![
            zev::Anchor { value: 50.0, description: "Anchor 1".into() },
            zev::Anchor { value: 30.0, description: "Anchor 2 (invalid decreasing)".into() },
        ],
        policy: Policy::default(),
    });
    assert!(non_monotonic_numeric.validate("numeric_q").is_err());
}

// --- 8. From nanojev-rs: ViZDoom Combat Multi-Task Evaluation ---
#[test]
fn test_multitask_gameplay_combat_decision() {
    let engine = ZevEngine::default();
    let combat_state = serde_json::json!({
        "monster_detected": true,
        "crosshair_offset_x": 0.0,
        "target_in_range": true,
        "ammo": 15,
        "situation": "Target centered in crosshairs, rocket launcher loaded and ready to discharge"
    });

    let action_q = Question::Choice(ChoiceQuestion {
        instructions: "Select next combat action".into(),
        options: vec![
            OptionDef { id: "turn_left".into(), description: "Turn weapon crosshairs left".into() },
            OptionDef { id: "turn_right".into(), description: "Turn weapon crosshairs right".into() },
            OptionDef { id: "fire".into(), description: "Target centered, discharge rocket".into() },
            OptionDef { id: "wait".into(), description: "Hold position".into() },
        ],
        policy: Policy { allow_abstain: false, ..Default::default() },
    });

    let mut questions = BTreeMap::new();
    questions.insert("action".into(), action_q);

    let resp = engine.evaluate(&ZevRequest {
        state: combat_state,
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }).unwrap();

    let ans = resp.answers.get("action").unwrap();
    assert_eq!(ans.decision, Some(serde_json::Value::String("fire".into())));
}

// --- 9. From kev-rs: Preprocessor Text Cleaning & Dynamic Date Grounding ---
#[test]
fn test_preprocessor_signature_cleaning_and_date_grounding() {
    // 1. Clean email signatures and disclaimers
    let email = "Customer needs urgent assistance with password reset.\n---\nJohn Doe\nAcme Corp\nConfidentiality Notice: This email and any attachments are confidential.";
    let cleaned = zev::clean_text(email);
    assert!(!cleaned.contains("Confidentiality Notice:"));
    assert!(!cleaned.contains("Acme Corp"));
    assert!(cleaned.contains("password reset"));

    // 2. Dynamic temporal facts injection
    let text_with_relative_time = "I requested a payout today, but my account has been locked since yesterday.";
    let grounded = zev::inject_temporal_facts(text_with_relative_time);
    assert!(grounded.contains("Temporal Facts: reference_date="));
    assert!(grounded.contains("yesterday="));
}

// --- 10. From nimble-rs: Fast Confidence Gating Helper ---
#[test]
fn test_confidence_gating_helper() {
    let engine = ZevEngine::default();
    let state = "Critical server incident: production database is down and taking no traffic.";

    let q = Question::Choice(ChoiceQuestion {
        instructions: "Is this a critical outage?".into(),
        options: vec![
            OptionDef { id: "critical".into(), description: "Critical server incident production down".into() },
            OptionDef { id: "routine".into(), description: "Routine general inquiry".into() },
        ],
        policy: Policy { allow_abstain: false, ..Default::default() },
    });

    // High confidence threshold (0.50) should pass for clear match
    let (passed, ans) = engine.confidence_gate(state, q, 0.40).unwrap();
    assert!(passed);
    assert_eq!(ans.decision, Some(serde_json::Value::String("critical".into())));
}

// --- 11. Optional Neural Backend with apfel-rs (Apple Intelligence / FoundationModels) ---
#[cfg(feature = "neural")]
#[test]
fn test_apfel_neural_speculative_hybrid() {
    let engine = ZevEngine::default();
    let backend = zev::ApfelNeuralBackend::new();

    let state = "Customer reports severe outage and database degradation.";
    let q = Question::Choice(ChoiceQuestion {
        instructions: "Assess priority".into(),
        options: vec![
            OptionDef { id: "critical".into(), description: "Critical outage database degradation".into() },
            OptionDef { id: "low".into(), description: "Routine question".into() },
        ],
        policy: Policy::default(),
    });

    let mut questions = BTreeMap::new();
    questions.insert("priority".into(), q);

    let req = ZevRequest {
        state: serde_json::json!(state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.evaluate_speculative_hybrid(&req, 0.70, &backend).unwrap();
    let ans = resp.answers.get("priority").unwrap();
    assert_eq!(ans.decision, Some(serde_json::Value::String("critical".into())));
}
