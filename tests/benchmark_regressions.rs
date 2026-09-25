//! Benchmark Regressions Test Suite
//!
//! Regression test suite based on failed test cases from JevBench public datasets
//! (/tmp/jevbench/datasets/public/*.jsonl):
//!
//! 1. Short token / single letter collisions:
//!    - easy-extraction-06: Size 'M' vs 'L' collision in "blue shirt in size M, please"
//!
//! 2. Morphological negations:
//!    - easy-fact-03: "unpaid" -> is invoice paid? Expected: "no" (false)
//!    - easy-fact-09: "disabled" -> is 2FA enabled? Expected: "no" (false)
//!
//! 3. Boolean policy precondition failures:
//!    - original-policy-01-0: Missing receipt -> refund permitted? Expected: "no" (false)
//!    - original-policy-04-1: Open dispute -> reminder permitted? Expected: "no" (false)
//!    - original-policy-05-0: Suspension -> file access permitted? Expected: "no" (false)
//!    - original-adequacy-05-0: Constraint "exactly two words" -> "All done now" (3 words) -> Expected: "no" (false)
//!
//! 4. Mention vs request intent:
//!    - original-intent-05-0: "Your refund policy is clearer now. Thanks for explaining it." -> Expected: "other"
//!    - original-intent-05-1: "Thank you; I understand the policy on refunds now." -> Expected: "other"
//!
//! 5. Ordinal severity ladder:
//!    - original-ordinal-01-0: Misaligned icon, every function works -> Expected: 0 ("cosmetic only")
//!    - original-ordinal-03-0: All customers cannot sign in, no records lost -> Expected: 2 ("core function blocked")
//!    - original-ordinal-04-0: Backups and records irreversibly deleted -> Expected: 3 ("irreversible data loss")

use std::collections::BTreeMap;
use zev::types::{
    BooleanQuestion, ChoiceQuestion, OptionDef, Policy, Question, ScoreQuestion, SystemOneRequest,
    WireNoulCriteria, WireNoulQuestion, WireQuestion, ZevRequest,
};
use zev::DecisionEngine;

// ============================================================================
// Helper extraction functions
// ============================================================================

fn get_choice_decision(resp: &zev::types::ZevResponse, key: &str) -> Option<String> {
    resp.answers.get(key).and_then(|ans| match &ans.decision {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        _ => None,
    })
}

fn get_boolean_decision(resp: &zev::types::ZevResponse, key: &str) -> Option<bool> {
    resp.answers.get(key).and_then(|ans| match &ans.decision {
        Some(serde_json::Value::Bool(b)) => Some(*b),
        _ => {
            let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);
            let p_false = ans.probabilities.get("false").copied().unwrap_or(0.0);
            Some(p_true > p_false)
        }
    })
}

fn get_score_argmax(resp: &zev::types::ZevResponse, key: &str) -> Option<usize> {
    resp.answers.get(key).and_then(|ans| {
        let mut best_idx = None;
        let mut best_p = -1.0;
        for (k, &p) in &ans.probabilities {
            if let Ok(idx) = k.parse::<usize>() {
                if p > best_p {
                    best_p = p;
                    best_idx = Some(idx);
                }
            }
        }
        best_idx
    })
}

// ============================================================================
// 1. Short token / single letter collisions
// ============================================================================

/// easy-extraction-06:
/// State: "Could I get the blue shirt in size M, please?"
/// Options: S ("Small"), M ("Medium"), L ("Large"), XL ("Extra large")
/// Expected: "M"
///
/// Current failure mode: Single letter token "M" collides with substring/token boundaries
/// or defaults to uniform probability across short labels, incorrectly picking "L" or another size.
#[test]
fn test_easy_extraction_06_short_token_size() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "easy-extraction-06".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Which shirt size does the customer ask for?".into(),
            options: vec![
                OptionDef {
                    id: "S".into(),
                    description: "Small".into(),
                },
                OptionDef {
                    id: "M".into(),
                    description: "Medium".into(),
                },
                OptionDef {
                    id: "L".into(),
                    description: "Large".into(),
                },
                OptionDef {
                    id: "XL".into(),
                    description: "Extra large".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Could I get the blue shirt in size M, please?"),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "easy-extraction-06");
    assert_eq!(
        decision.as_deref(),
        Some("M"),
        "Failed easy-extraction-06: expected size 'M', got {:?}",
        decision
    );
}

// ============================================================================
// 2. Morphological negations
// ============================================================================

/// easy-fact-03:
/// State: "Invoice 2026-045. Amount: 80 EUR. Payment status: unpaid, overdue since 1 August."
/// Question: "Is the invoice paid? Answer strictly from the facts stated in the text."
/// Expected: "no" (false)
///
/// Current failure mode: The presence of root word "paid" inside "unpaid" triggers
/// strong affirmative lexical match, missing the morphological negation prefix "un-".
#[test]
fn test_easy_fact_03_morphological_negation_unpaid() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "easy-fact-03".to_string(),
        Question::Boolean(BooleanQuestion {
            instructions: "Is the invoice paid? Answer strictly from the facts stated in the text."
                .into(),
            true_description: "The text states that this is so".into(),
            false_description: "The text states that this is not so".into(),
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "Invoice 2026-045. Amount: 80 EUR. Payment status: unpaid, overdue since 1 August."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_boolean_decision(&resp, "easy-fact-03");
    assert_eq!(
        decision,
        Some(false),
        "Failed easy-fact-03: unpaid invoice should evaluate to false ('no')"
    );
}

/// easy-fact-09:
/// State: "Account settings: two-factor authentication is disabled."
/// Question: "Is two-factor authentication enabled? Answer strictly from the facts stated in the text."
/// Expected: "no" (false)
///
/// Current failure mode: The stem "enabled" inside "disabled" causes affirmative false-positive,
/// ignoring prefix "dis-".
#[test]
fn test_easy_fact_09_morphological_negation_disabled() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "easy-fact-09".to_string(),
        Question::Boolean(BooleanQuestion {
            instructions: "Is two-factor authentication enabled? Answer strictly from the facts stated in the text.".into(),
            true_description: "The text states that this is so".into(),
            false_description: "The text states that this is not so".into(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Account settings: two-factor authentication is disabled."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_boolean_decision(&resp, "easy-fact-09");
    assert_eq!(
        decision,
        Some(false),
        "Failed easy-fact-09: disabled 2FA should evaluate to false ('no')"
    );
}

// ============================================================================
// 3. Boolean policy precondition failures
// ============================================================================

/// original-policy-01-0:
/// State: "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt. Issue a refund."
/// Instructions: "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied."
/// Expected: "no" (false)
///
/// Current failure mode: In wire/JevBench protocol evaluation, positive policy tokens dominate,
/// predicting permitted ("yes", ~91.3% noul probability) instead of rejecting for missing receipt.
#[test]
fn test_original_policy_01_0_no_receipt_refund() {
    let engine = DecisionEngine::new();
    let sys1_req = SystemOneRequest {
        state: serde_json::json!("Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago but has no receipt. Issue a refund."),
        model: "openjev-v2".into(),
        questions: {
            let mut map = BTreeMap::new();
            map.insert(
                "original-policy-01-0".to_string(),
                WireQuestion::Noul(WireNoulQuestion {
                    instructions: serde_json::json!("Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied."),
                    criteria: Some(WireNoulCriteria {
                        false_criterion: Some(serde_json::json!("A condition is missing or a prohibition applies.")),
                        true_criterion: Some(serde_json::json!("Every required condition is established and no prohibition applies.")),
                    }),
                }),
            );
            map
        },
    };

    let resp = engine.eval(&sys1_req).expect("evaluation failed");
    let ans = resp.answers.get("original-policy-01-0").unwrap();
    let p_true = ans["noul"].as_f64().unwrap();
    assert!(
        p_true < 0.5,
        "Failed original-policy-01-0: expected 'no' (p_true < 0.5), got p_true = {p_true}"
    );
}

/// original-policy-04-1:
/// State: "There is an open dispute on an overdue invoice. Reminders require overdue status AND absence of disputes. Is a reminder allowed?"
/// Instructions: "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied."
/// Expected: "no" (false)
///
/// Current failure mode: Fails to enforce negative precondition ("absence of disputes")
/// when "overdue" condition is satisfied, predicting permitted (true ~97.7%).
#[test]
fn test_original_policy_04_1_open_dispute_reminder() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-policy-04-1".to_string(),
        Question::Boolean(BooleanQuestion {
            instructions: "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied.".into(),
            true_description: "Every required condition is established and no prohibition applies.".into(),
            false_description: "A condition is missing or a prohibition applies.".into(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("There is an open dispute on an overdue invoice. Reminders require overdue status AND absence of disputes. Is a reminder allowed?"),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_boolean_decision(&resp, "original-policy-04-1");
    assert_eq!(
        decision,
        Some(false),
        "Failed original-policy-04-1: open dispute prohibits reminder, expected false ('no')"
    );
}

/// original-policy-05-0:
/// State: "Policy: staff may access files in their own team. A temporary suspension overrides all access. The staff member owns the team but is suspended. Open its file."
/// Instructions: "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied."
/// Expected: "no" (false)
///
/// Current failure mode: Fails to apply override ("temporary suspension overrides all access"),
/// allowing file access (true ~85.7%).
#[test]
fn test_original_policy_05_0_suspension_file_access() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-policy-05-0".to_string(),
        Question::Boolean(BooleanQuestion {
            instructions: "Under the stated policy, is the requested action permitted? Treat unproved required conditions as not satisfied.".into(),
            true_description: "Every required condition is established and no prohibition applies.".into(),
            false_description: "A condition is missing or a prohibition applies.".into(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Policy: staff may access files in their own team. A temporary suspension overrides all access. The staff member owns the team but is suspended. Open its file."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_boolean_decision(&resp, "original-policy-05-0");
    assert_eq!(
        decision,
        Some(false),
        "Failed original-policy-05-0: suspension override should block access, expected false ('no')"
    );
}

/// original-adequacy-05-0:
/// State: "Request: Say exactly two words. Response:All done now"
/// Instructions: "Does the response fully satisfy the request, using the supplied reference when present?"
/// Expected: "no" (false)
///
/// Current failure mode: "All done now" is 3 words, violating the "exactly two words" constraint;
/// engine marks constraint as satisfied (true ~95.3%).
#[test]
fn test_original_adequacy_05_0_two_words_constraint() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-adequacy-05-0".to_string(),
        Question::Boolean(BooleanQuestion {
            instructions: "Does the response fully satisfy the request, using the supplied reference when present?".into(),
            true_description: "Correct, complete, and follows all explicit constraints".into(),
            false_description: "Wrong, incomplete, unsupported or violates a constraint".into(),
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Request: Say exactly two words. Response:All done now"),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_boolean_decision(&resp, "original-adequacy-05-0");
    assert_eq!(
        decision,
        Some(false),
        "Failed original-adequacy-05-0: 3 words violates 'exactly two words' constraint, expected false ('no')"
    );
}

// ============================================================================
// 4. Mention vs request intent
// ============================================================================

/// original-intent-05-0:
/// State: "Your refund policy is clearer now. Thanks for explaining it."
/// Instructions: "Select the primary requested action. A mention without a request does not establish intent."
/// Options: cancel, refund, status, change_address, other
/// Expected: "other"
///
/// Current failure mode: Mention of "refund" in compliment triggers false-positive "refund" intent
/// (~60.7% prob) despite explicit rubric "A mention without a request does not establish intent."
#[test]
fn test_original_intent_05_0_mention_vs_request_clarity() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-intent-05-0".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the primary requested action. A mention without a request does not establish intent.".into(),
            options: vec![
                OptionDef { id: "cancel".into(), description: "End an existing subscription".into() },
                OptionDef { id: "refund".into(), description: "Return money already charged".into() },
                OptionDef { id: "status".into(), description: "Learn delivery progress".into() },
                OptionDef { id: "change_address".into(), description: "Modify a delivery address".into() },
                OptionDef { id: "other".into(), description: "None of these actions is requested".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Your refund policy is clearer now. Thanks for explaining it."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "original-intent-05-0");
    assert_eq!(
        decision.as_deref(),
        Some("other"),
        "Failed original-intent-05-0: mention of refund without request should be 'other'"
    );
}

/// original-intent-05-1:
/// State: "Thank you; I understand the policy on refunds now."
/// Instructions: "Select the primary requested action. A mention without a request does not establish intent."
/// Options: cancel, refund, status, change_address, other
/// Expected: "other"
///
/// Current failure mode: Mention of refunds in gratitude triggers action classification (arbitrarily "cancel"
/// due to lexical tie-break) instead of "other".
#[test]
fn test_original_intent_05_1_mention_vs_request_gratitude() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-intent-05-1".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the primary requested action. A mention without a request does not establish intent.".into(),
            options: vec![
                OptionDef { id: "cancel".into(), description: "End an existing subscription".into() },
                OptionDef { id: "refund".into(), description: "Return money already charged".into() },
                OptionDef { id: "status".into(), description: "Learn delivery progress".into() },
                OptionDef { id: "change_address".into(), description: "Modify a delivery address".into() },
                OptionDef { id: "other".into(), description: "None of these actions is requested".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Thank you; I understand the policy on refunds now."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "original-intent-05-1");
    assert_eq!(
        decision.as_deref(),
        Some("other"),
        "Failed original-intent-05-1: understanding policy without request should be 'other'"
    );
}

// ============================================================================
// 5. Ordinal severity ladder
// ============================================================================

/// original-ordinal-01-0:
/// State: "The icon is misaligned. Every function works."
/// Levels: 0 (cosmetic), 1 (one user/nonessential), 2 (many users blocked), 3 (irreversible loss)
/// Expected: 0
///
/// Current failure mode: "Every function works" triggers positive function matches at levels 1 and 2,
/// causing argmax to select level 1 instead of 0.
#[test]
fn test_original_ordinal_01_0_severity_cosmetic() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-ordinal-01-0".to_string(),
        Question::Score(ScoreQuestion {
            instructions: "Rate incident impact using only reported facts. Use the highest fully supported level.".into(),
            levels: vec![
                "No function impaired; cosmetic only".into(),
                "One user or a nonessential function impaired, with a workaround".into(),
                "Many users blocked from a core function, no data loss".into(),
                "Confirmed irreversible data loss or physical harm".into(),
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("The icon is misaligned. Every function works."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let argmax = get_score_argmax(&resp, "original-ordinal-01-0");
    assert_eq!(
        argmax,
        Some(0),
        "Failed original-ordinal-01-0: purely cosmetic impact should be level 0"
    );
}

/// original-ordinal-03-0:
/// State: "All customers cannot sign in. No records are lost."
/// Levels: 0 (cosmetic), 1 (one user/nonessential), 2 (many users blocked), 3 (irreversible loss)
/// Expected: 2
///
/// Current failure mode: Negative phrase "No records are lost" over-weights level 0 ("No function impaired"),
/// causing argmax to pick 0 instead of 2 ("Many users blocked").
#[test]
fn test_original_ordinal_03_0_severity_blocked_core_function() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-ordinal-03-0".to_string(),
        Question::Score(ScoreQuestion {
            instructions: "Rate incident impact using only reported facts. Use the highest fully supported level.".into(),
            levels: vec![
                "No function impaired; cosmetic only".into(),
                "One user or a nonessential function impaired, with a workaround".into(),
                "Many users blocked from a core function, no data loss".into(),
                "Confirmed irreversible data loss or physical harm".into(),
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("All customers cannot sign in. No records are lost."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let argmax = get_score_argmax(&resp, "original-ordinal-03-0");
    assert_eq!(
        argmax,
        Some(2),
        "Failed original-ordinal-03-0: all customers unable to sign in should be level 2"
    );
}

/// original-ordinal-04-0:
/// State: "Backups and original customer records have been irreversibly deleted."
/// Levels: 0 (cosmetic), 1 (one user/nonessential), 2 (many users blocked), 3 (irreversible loss)
/// Expected: 3
///
/// Current failure mode: Fails to ladder to the top severity level 3, distributing probability
/// across levels 1-3 and settling on level 1.
#[test]
fn test_original_ordinal_04_0_severity_irreversible_loss() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "original-ordinal-04-0".to_string(),
        Question::Score(ScoreQuestion {
            instructions: "Rate incident impact using only reported facts. Use the highest fully supported level.".into(),
            levels: vec![
                "No function impaired; cosmetic only".into(),
                "One user or a nonessential function impaired, with a workaround".into(),
                "Many users blocked from a core function, no data loss".into(),
                "Confirmed irreversible data loss or physical harm".into(),
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "Backups and original customer records have been irreversibly deleted."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let argmax = get_score_argmax(&resp, "original-ordinal-04-0");
    assert_eq!(
        argmax,
        Some(3),
        "Failed original-ordinal-04-0: irreversible deletion should be level 3"
    );
}

#[test]
fn test_easy_intent_04_billing_question() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Which intent does the user's message express?".into(),
            options: vec![
                OptionDef {
                    id: "track_order".into(),
                    description: "Track order".into(),
                },
                OptionDef {
                    id: "cancel_order".into(),
                    description: "Cancel order".into(),
                },
                OptionDef {
                    id: "change_address".into(),
                    description: "Change address".into(),
                },
                OptionDef {
                    id: "report_damage".into(),
                    description: "Report damage".into(),
                },
                OptionDef {
                    id: "billing_question".into(),
                    description: "Billing question".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "Why was I charged twice on my credit card statement for one order?"
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("billing_question".to_string()));
}

#[test]
fn test_easy_intent_08_weather() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Which intent does the user's message express?".into(),
            options: vec![
                OptionDef {
                    id: "set_alarm".into(),
                    description: "Set alarm".into(),
                },
                OptionDef {
                    id: "play_music".into(),
                    description: "Play music".into(),
                },
                OptionDef {
                    id: "weather".into(),
                    description: "Weather".into(),
                },
                OptionDef {
                    id: "send_message".into(),
                    description: "Send message".into(),
                },
                OptionDef {
                    id: "turn_off_lights".into(),
                    description: "Turn off lights".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("Will it rain in Berlin tomorrow?"),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("weather".to_string()));
}

#[test]
fn test_original_intent_04_0_cancel_imperative() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the primary requested action. A mention without a request does not establish intent.".into(),
            options: vec![
                OptionDef { id: "cancel".into(), description: "cancel".into() },
                OptionDef { id: "refund".into(), description: "refund".into() },
                OptionDef { id: "status".into(), description: "status".into() },
                OptionDef { id: "change_address".into(), description: "change_address".into() },
                OptionDef { id: "other".into(), description: "other".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "Stop renewing my subscription after this month; I am not asking for money back."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("cancel".to_string()));
}

#[test]
fn test_original_intent_06_0_status_with_past_cancellation() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the primary requested action. A mention without a request does not establish intent.".into(),
            options: vec![
                OptionDef { id: "cancel".into(), description: "cancel".into() },
                OptionDef { id: "refund".into(), description: "refund".into() },
                OptionDef { id: "status".into(), description: "status".into() },
                OptionDef { id: "change_address".into(), description: "change_address".into() },
                OptionDef { id: "other".into(), description: "other".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("I cancelled yesterday. Was the parcel delivered yet?"),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("status".to_string()));
}

#[test]
fn test_original_extraction_01_1_depot_pickup_replacing_courier() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Extract the final confirmed delivery method. Ignore cancelled plans and hypothetical alternatives. Choose unknown if no final method is confirmed.".into(),
            options: vec![
                OptionDef { id: "courier".into(), description: "courier".into() },
                OptionDef { id: "pickup".into(), description: "pickup".into() },
                OptionDef { id: "post".into(), description: "post".into() },
                OptionDef { id: "unknown".into(), description: "unknown".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "The final arrangement is depot pickup, replacing the earlier courier idea."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("pickup".to_string()));
}

#[test]
fn test_original_extraction_02_0_unknown_delivery_hypothetical() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "decision".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Extract the final confirmed delivery method. Ignore cancelled plans and hypothetical alternatives. Choose unknown if no final method is confirmed.".into(),
            options: vec![
                OptionDef { id: "courier".into(), description: "courier".into() },
                OptionDef { id: "pickup".into(), description: "pickup".into() },
                OptionDef { id: "post".into(), description: "post".into() },
                OptionDef { id: "unknown".into(), description: "unknown".into() },
            ],
            policy: Policy { allow_abstain: false, ..Default::default() },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!("If approved, we might use post. No method is booked yet."),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "decision");
    assert_eq!(decision, Some("unknown".to_string()));
}

#[test]
fn test_laya_377_cancellation_negation_inversion() {
    let engine = DecisionEngine::new();
    let mut questions = BTreeMap::new();
    questions.insert(
        "q".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "What does the user want?".into(),
            options: vec![
                OptionDef {
                    id: "no_action".into(),
                    description: "keep the account as it is".into(),
                },
                OptionDef {
                    id: "cancel_account".into(),
                    description: "close the account".into(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "I do not want to cancel my subscription. Please keep the account as it is."
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.eval(&req).expect("evaluation failed");
    let decision = get_choice_decision(&resp, "q");
    assert_eq!(decision, Some("no_action".to_string()));
}
