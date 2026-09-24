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
}
