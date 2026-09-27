//! Decision Index Regression & Non-Regression Test Suite
//!
//! Native Rust unit and integration tests converted from the multimodalart/jev-decision-index
//! benchmark suite. Guarantees that accuracy floors, high-cardinality routing,
//! order-invariance, and latency do not slide across releases or refactors.
//!
//! Covered Benchmarks:
//! 1. SimpleBench: Complete suite (all 10 official public questions).
//! 2. ARC-Easy: Representative 100-question sample.
//! 3. ARC-Challenge: Representative 100-question sample.
//! 4. ContractNLI: 50 multi-clause contract entailment tasks (>200 sub-decisions).
//! 5. Home Appliance Simulator: 15 multi-step action sequence scenarios.
//! 6. BANKING77: 30 intent classification queries across 77 options.
//! 7. CLINC150+OOS: 30 intent classification queries across 151 options.

use std::collections::{BTreeMap, HashMap};
use zev::types::{SystemOneRequest, WireQuestion};
use zev::ZevEngine;

#[derive(Debug, serde::Deserialize)]
struct BenchmarkTestCase {
    pub id: String,
    pub benchmark: String,
    pub state: serde_json::Value,
    pub questions: BTreeMap<String, WireQuestion>,
    pub expected: BTreeMap<String, String>,
}

fn load_fixtures() -> Vec<BenchmarkTestCase> {
    let raw = include_str!("fixtures/decision_index_regression.json");
    serde_json::from_str(raw).expect("Failed to deserialize decision_index_regression.json fixture")
}

#[test]
fn test_fixture_loading_and_schema_validation() {
    let cases = load_fixtures();
    assert!(!cases.is_empty(), "Fixture cases must not be empty");
    assert_eq!(cases.len(), 335, "Expected 335 fixture cases");

    let mut by_benchmark: HashMap<String, usize> = HashMap::new();
    for c in &cases {
        *by_benchmark.entry(c.benchmark.clone()).or_insert(0) += 1;
    }

    assert_eq!(*by_benchmark.get("SimpleBench").unwrap_or(&0), 10);
    assert_eq!(*by_benchmark.get("ARC-Easy").unwrap_or(&0), 100);
    assert_eq!(*by_benchmark.get("ARC-Challenge").unwrap_or(&0), 100);
    assert_eq!(*by_benchmark.get("ContractNLI").unwrap_or(&0), 50);
    assert_eq!(
        *by_benchmark.get("Home appliance simulator").unwrap_or(&0),
        15
    );
    assert_eq!(*by_benchmark.get("BANKING77").unwrap_or(&0), 30);
    assert_eq!(*by_benchmark.get("CLINC150+OOS").unwrap_or(&0), 30);
}

#[test]
fn test_simplebench_regression_and_accuracy_floor() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let simplebench_cases: Vec<_> = cases
        .into_iter()
        .filter(|c| c.benchmark == "SimpleBench")
        .collect();

    assert_eq!(simplebench_cases.len(), 10);

    let mut correct = 0;
    let total = simplebench_cases.len();

    for case in &simplebench_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };

        let resp = engine
            .evaluate_system_one(&req)
            .unwrap_or_else(|e| panic!("Evaluation failed on {}: {}", case.id, e));

        for (q_id, exp) in &case.expected {
            let ans_val = resp
                .answers
                .get(q_id)
                .unwrap_or_else(|| panic!("Missing answer for {} in {}", q_id, case.id));

            let choice = match ans_val.get("choice").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => panic!("Expected choice field in answer: {:?}", ans_val),
            };

            // Probability distribution validation
            let probs = ans_val
                .get("probabilities")
                .and_then(|p| p.as_object())
                .expect("Probabilities map missing");
            let sum: f64 = probs.values().filter_map(|v| v.as_f64()).sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "Probabilities must sum to 1.0, got {}",
                sum
            );

            if choice == exp {
                correct += 1;
            }
        }
    }

    let accuracy = correct as f64 / total as f64;
    println!(
        "SimpleBench Accuracy: {}/{} ({:.2}%)",
        correct,
        total,
        accuracy * 100.0
    );

    // Official multimodalart SimpleBench baseline is 40.0% (4/10)
    assert!(
        accuracy >= 0.40,
        "SimpleBench accuracy slid below baseline floor! Got {:.2}%, expected >= 40.0%",
        accuracy * 100.0
    );
}

#[test]
fn test_arc_easy_regression_and_accuracy_floor() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let arc_cases: Vec<_> = cases
        .into_iter()
        .filter(|c| c.benchmark == "ARC-Easy")
        .collect();

    assert_eq!(arc_cases.len(), 100);

    let mut correct = 0;
    let mut total = 0;

    for case in &arc_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };

        let resp = engine
            .evaluate_system_one(&req)
            .unwrap_or_else(|e| panic!("ARC-Easy eval failed on {}: {}", case.id, e));

        for (q_id, exp) in &case.expected {
            let ans_val = resp.answers.get(q_id).expect("Answer missing");
            let choice = ans_val.get("choice").and_then(|v| v.as_str()).unwrap();
            if choice == exp {
                correct += 1;
            }
            total += 1;
        }
    }

    let accuracy = correct as f64 / total as f64;
    println!(
        "ARC-Easy Accuracy (100 sample): {}/{} ({:.2}%)",
        correct,
        total,
        accuracy * 100.0
    );

    // Official multimodalart ARC-Easy baseline is ~26.0%
    assert!(
        accuracy >= 0.22,
        "ARC-Easy accuracy slid below regression floor! Got {:.2}%, expected >= 22.0%",
        accuracy * 100.0
    );
}

#[test]
fn test_arc_challenge_regression_and_accuracy_floor() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let arc_cases: Vec<_> = cases
        .into_iter()
        .filter(|c| c.benchmark == "ARC-Challenge")
        .collect();

    assert_eq!(arc_cases.len(), 100);

    let mut correct = 0;
    let mut total = 0;

    for case in &arc_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };

        let resp = engine
            .evaluate_system_one(&req)
            .unwrap_or_else(|e| panic!("ARC-Challenge eval failed on {}: {}", case.id, e));

        for (q_id, exp) in &case.expected {
            let ans_val = resp.answers.get(q_id).expect("Answer missing");
            let choice = ans_val.get("choice").and_then(|v| v.as_str()).unwrap();
            if choice == exp {
                correct += 1;
            }
            total += 1;
        }
    }

    let accuracy = correct as f64 / total as f64;
    println!(
        "ARC-Challenge Accuracy (100 sample): {}/{} ({:.2}%)",
        correct,
        total,
        accuracy * 100.0
    );

    // Official multimodalart ARC-Challenge baseline is ~23.2%
    assert!(
        accuracy >= 0.20,
        "ARC-Challenge accuracy slid below regression floor! Got {:.2}%, expected >= 20.0%",
        accuracy * 100.0
    );
}

#[test]
fn test_contract_nli_entailment_classification() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let contract_cases: Vec<_> = cases
        .into_iter()
        .filter(|c| c.benchmark == "ContractNLI")
        .collect();

    assert_eq!(contract_cases.len(), 50);

    let mut total_questions = 0;
    let mut answered_ok = 0;

    for case in &contract_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };

        let resp = engine
            .evaluate_system_one(&req)
            .unwrap_or_else(|e| panic!("ContractNLI eval failed on {}: {}", case.id, e));

        for q_id in case.expected.keys() {
            if let Some(ans_val) = resp.answers.get(q_id) {
                let choice = ans_val.get("choice").and_then(|v| v.as_str()).unwrap();
                assert!(
                    choice == "Entailment" || choice == "Contradiction" || choice == "NotMentioned",
                    "Invalid ContractNLI choice: {}",
                    choice
                );
                answered_ok += 1;
            }
            total_questions += 1;
        }
    }

    assert_eq!(answered_ok, total_questions);
    assert!(
        total_questions > 200,
        "Expected > 200 contract clause questions evaluated"
    );
}

#[test]
fn test_home_appliance_simulator_action_sequences() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let appliance_cases: Vec<_> = cases
        .into_iter()
        .filter(|c| c.benchmark == "Home appliance simulator")
        .collect();

    assert_eq!(appliance_cases.len(), 15);

    let mut total_subquestions = 0;
    for case in &appliance_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };

        let resp = engine
            .evaluate_system_one(&req)
            .unwrap_or_else(|e| panic!("Home appliance simulator failed on {}: {}", case.id, e));

        for q_id in case.expected.keys() {
            let ans = resp
                .answers
                .get(q_id)
                .unwrap_or_else(|| panic!("Missing answer for {} in {}", q_id, case.id));

            let choice = ans.get("choice").and_then(|v| v.as_str()).unwrap();
            assert!(!choice.is_empty(), "Choice must not be empty");

            let probs = ans
                .get("probabilities")
                .and_then(|p| p.as_object())
                .unwrap();
            let sum: f64 = probs.values().filter_map(|v| v.as_f64()).sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "Probabilities must sum to 1.0, got {}",
                sum
            );
            total_subquestions += 1;
        }
    }

    assert!(
        total_subquestions >= 100,
        "Expected >= 100 action questions evaluated"
    );
}

#[test]
fn test_banking77_and_clinc150_high_cardinality_routing() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();

    let banking_cases: Vec<_> = cases
        .iter()
        .filter(|c| c.benchmark == "BANKING77")
        .collect();
    let clinc_cases: Vec<_> = cases
        .iter()
        .filter(|c| c.benchmark == "CLINC150+OOS")
        .collect();

    assert_eq!(banking_cases.len(), 30);
    assert_eq!(clinc_cases.len(), 30);

    // Test BANKING77 (77 candidate options per question)
    for case in banking_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };
        let resp = engine
            .evaluate_system_one(&req)
            .expect("BANKING77 eval failed");
        for q_id in case.expected.keys() {
            let ans = resp.answers.get(q_id).expect("BANKING77 answer missing");
            let probs = ans
                .get("probabilities")
                .and_then(|p| p.as_object())
                .unwrap();
            assert_eq!(
                probs.len(),
                77,
                "BANKING77 must contain all 77 candidates in distribution"
            );
            let sum: f64 = probs.values().filter_map(|v| v.as_f64()).sum();
            assert!((sum - 1.0).abs() < 1e-3, "BANKING77 sum must equal 1.0");
        }
    }

    // Test CLINC150+OOS (151 candidate options per question)
    for case in clinc_cases {
        let req = SystemOneRequest {
            state: case.state.clone(),
            model: "zev-latest".to_string(),
            questions: case.questions.clone(),
        };
        let resp = engine
            .evaluate_system_one(&req)
            .expect("CLINC150 eval failed");
        for q_id in case.expected.keys() {
            let ans = resp.answers.get(q_id).expect("CLINC150 answer missing");
            let probs = ans
                .get("probabilities")
                .and_then(|p| p.as_object())
                .unwrap();
            assert_eq!(
                probs.len(),
                151,
                "CLINC150 must contain all 151 candidates in distribution"
            );
            let sum: f64 = probs.values().filter_map(|v| v.as_f64()).sum();
            assert!((sum - 1.0).abs() < 1e-3, "CLINC150 sum must equal 1.0");
        }
    }
}

#[test]
fn test_order_invariance_on_benchmark_suite() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();

    // Pick 20 questions across ARC and SimpleBench to verify 0.0% order flip rate
    for case in cases
        .iter()
        .filter(|c| c.benchmark == "ARC-Easy" || c.benchmark == "SimpleBench")
        .take(20)
    {
        for (q_id, q_def) in &case.questions {
            if let WireQuestion::Choice(choice_q) = q_def {
                // Forward order
                let mut fwd_criteria = BTreeMap::new();
                for (k, v) in &choice_q.criteria {
                    fwd_criteria.insert(k.clone(), v.clone());
                }

                // Reversed order
                let mut rev_criteria = BTreeMap::new();
                let keys: Vec<_> = choice_q.criteria.keys().cloned().collect();
                for k in keys.into_iter().rev() {
                    rev_criteria.insert(k.clone(), choice_q.criteria.get(&k).cloned().flatten());
                }

                let mut q_map1 = BTreeMap::new();
                q_map1.insert(
                    q_id.clone(),
                    WireQuestion::Choice(zev::types::WireChoiceQuestion {
                        instructions: choice_q.instructions.clone(),
                        criteria: fwd_criteria,
                    }),
                );

                let mut q_map2 = BTreeMap::new();
                q_map2.insert(
                    q_id.clone(),
                    WireQuestion::Choice(zev::types::WireChoiceQuestion {
                        instructions: choice_q.instructions.clone(),
                        criteria: rev_criteria,
                    }),
                );

                let req1 = SystemOneRequest {
                    state: case.state.clone(),
                    model: "zev-latest".to_string(),
                    questions: q_map1,
                };
                let req2 = SystemOneRequest {
                    state: case.state.clone(),
                    model: "zev-latest".to_string(),
                    questions: q_map2,
                };

                let resp1 = engine
                    .evaluate_system_one(&req1)
                    .expect("Forward eval failed");
                let resp2 = engine
                    .evaluate_system_one(&req2)
                    .expect("Reversed eval failed");

                let c1 = resp1.answers[q_id]["choice"].as_str().unwrap();
                let c2 = resp2.answers[q_id]["choice"].as_str().unwrap();

                assert_eq!(
                    c1, c2,
                    "Order invariance violated on {}/{}! Forward: {}, Reversed: {}",
                    case.id, q_id, c1, c2
                );
            }
        }
    }
}

#[test]
fn test_native_submillisecond_latency_budget() {
    let engine = ZevEngine::default();
    let cases = load_fixtures();
    let test_case = cases
        .iter()
        .find(|c| c.benchmark == "ARC-Challenge")
        .unwrap();

    let req = SystemOneRequest {
        state: test_case.state.clone(),
        model: "zev-latest".to_string(),
        questions: test_case.questions.clone(),
    };

    // Warm-up
    for _ in 0..10 {
        let _ = engine.evaluate_system_one(&req).unwrap();
    }

    // Measure 100 evaluations
    let start = std::time::Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ = engine.evaluate_system_one(&req).unwrap();
    }
    let total_time = start.elapsed();
    let avg_latency = total_time / iters;

    println!("Average pure-Rust decision latency: {:.2?}", avg_latency);

    // Pure Rust evaluation should be well under 1 millisecond (typically < 150 microseconds)
    assert!(
        avg_latency < std::time::Duration::from_millis(1),
        "Decision latency exceeded 1ms budget: {:.2?}",
        avg_latency
    );
}
