//! Multi-Method Benchmark Evaluation Suite for zev-rs
//!
//! Provides unit and integration tests to run all major benchmarks against EVERY
//! method supported by the Zev decision engine:
//!
//! Methods Tested:
//! 1. Pure SIMD Zero-Token (`zev-default` / `native`)
//! 2. Contrastive Language Model Shortlisting (`zev-clm`)
//! 3. Apple Intelligence / Apple Neural Engine (`zev-apfel`)
//! 4. Distilled Gemma 4 Speculative Fallback (`zev-gemma`)
//! 5. Dual Speculative Cascade (`zev-cascade`)
//! 6. Bayesian Product of Experts Ensemble (`zev-poe`)
//!
//! Benchmarks Covered:
//! 1. TypeSafe WorkflowEvals (`tests/fixtures/workflowevals_sample.json`)
//!    - customer_service, invoice_processing, security_incidents, agent_trace_observability
//! 2. Decision Index (`tests/fixtures/decision_index_regression.json`)
//!    - SimpleBench, ARC-Easy, ARC-Challenge, ContractNLI, HomeAppliance, Banking77, Clinc150
//! 3. JevBench 231 (`datasets/jevbench_public/`)
//!    - easy.jsonl, original.jsonl, hard.jsonl (18 task families, 231 frozen tasks)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;
use zev::types::SystemOneRequest;
use zev::wire::WireQuestion;
use zev::ZevEngine;

// -------------------------------------------------------------------------------------------------
// Method Definitions
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZevMethod {
    Simd,
    Clm,
    Apfel,
    Gemma,
    Cascade,
    Poe,
}

impl ZevMethod {
    pub const ALL: &'static [ZevMethod] = &[
        ZevMethod::Simd,
        ZevMethod::Clm,
        ZevMethod::Apfel,
        ZevMethod::Gemma,
        ZevMethod::Cascade,
        ZevMethod::Poe,
    ];

    pub fn model_name(&self) -> &'static str {
        match self {
            ZevMethod::Simd => "zev-default",
            ZevMethod::Clm => "zev-clm",
            ZevMethod::Apfel => "zev-apfel",
            ZevMethod::Gemma => "zev-gemma",
            ZevMethod::Cascade => "zev-cascade",
            ZevMethod::Poe => "zev-poe",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            ZevMethod::Simd => "Pure SIMD (Native)",
            ZevMethod::Clm => "Contrastive LM (CLM)",
            ZevMethod::Apfel => "Apple Intelligence (ANE)",
            ZevMethod::Gemma => "Distilled Gemma 4",
            ZevMethod::Cascade => "Dual Speculative Cascade",
            ZevMethod::Poe => "Product of Experts (PoE)",
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Helpers & Extractors
// -------------------------------------------------------------------------------------------------

fn extract_prediction_token(ans_val: &Value) -> String {
    if let Some(choice) = ans_val.get("choice").and_then(|v| v.as_str()) {
        choice.to_string()
    } else if let Some(noul_val) = ans_val.get("noul").and_then(|v| v.as_f64()) {
        if noul_val >= 0.5 {
            "true".to_string()
        } else {
            "false".to_string()
        }
    } else if let Some(probs) = ans_val.get("probabilities").and_then(|p| p.as_object()) {
        let best = probs
            .iter()
            .max_by(|a, b| {
                let pa = a.1.as_f64().unwrap_or(0.0);
                let pb = b.1.as_f64().unwrap_or(0.0);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.as_str())
            .unwrap_or("0");
        best.to_string()
    } else if let Some(score) = ans_val.get("score").and_then(|v| v.as_f64()) {
        format!("{}", score.round() as i64)
    } else {
        String::new()
    }
}

fn matches_expected(pred: &str, expected: &str) -> bool {
    let clean_pred = pred.trim().trim_matches('"').to_lowercase();
    let clean_exp = expected.trim().trim_matches('"').to_lowercase();
    if clean_pred == clean_exp {
        return true;
    }
    // Boolean synonyms (yes/true, no/false, 1/true, 0/false)
    let is_pred_yes = clean_pred == "true" || clean_pred == "yes" || clean_pred == "1";
    let is_exp_yes = clean_exp == "true" || clean_exp == "yes" || clean_exp == "1";
    if is_pred_yes && is_exp_yes {
        return true;
    }
    let is_pred_no = clean_pred == "false" || clean_pred == "no" || clean_pred == "0";
    let is_exp_no = clean_exp == "false" || clean_exp == "no" || clean_exp == "0";
    if is_pred_no && is_exp_no {
        return true;
    }
    false
}

// -------------------------------------------------------------------------------------------------
// 1. Smoke Test (Always Runs on `cargo test`)
// -------------------------------------------------------------------------------------------------

#[test]
fn test_smoke_all_methods() {
    let engine = ZevEngine::default();
    let mut criteria = BTreeMap::new();
    criteria.insert("refund".to_string(), Some(serde_json::json!("Customer seeks financial reimbursement")));
    criteria.insert("support".to_string(), Some(serde_json::json!("Technical application malfunction")));
    let q = WireQuestion::Choice(zev::wire::WireChoiceQuestion {
        instructions: serde_json::json!("Categorize ticket"),
        criteria,
    });

    for method in ZevMethod::ALL {
        let req = SystemOneRequest {
            state: serde_json::json!("Please refund transaction TX-100 immediately."),
            questions: [("cat".to_string(), q.clone())].into(),
            model: method.model_name().into(),
        };

        let resp = engine.evaluate_system_one(&req).unwrap_or_else(|e| {
            panic!("Method {:?} failed evaluate_system_one: {}", method, e);
        });

        assert_eq!(resp.model, method.model_name());
        let ans = resp.answers.get("cat").expect("Missing answer");
        let pred = extract_prediction_token(ans);
        assert_eq!(pred, "refund", "Method {:?} should classify refund", method);
    }
}

// -------------------------------------------------------------------------------------------------
// 2. TypeSafe WorkflowEvals Multi-Method Evaluation
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct WorkflowEvalSample {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    workflow: String,
    #[allow(dead_code)]
    kind: String,
    question: WireQuestion,
    state: Value,
    expected: String,
    #[allow(dead_code)]
    jev_ref: Option<String>,
}

fn load_workflowevals_samples() -> Vec<WorkflowEvalSample> {
    let raw = include_str!("fixtures/workflowevals_sample.json");
    serde_json::from_str(raw).expect("Deserialize workflowevals_sample.json")
}

#[test]
fn test_workflowevals_sample_across_methods() {
    let engine = ZevEngine::default();
    let samples = load_workflowevals_samples();
    assert!(!samples.is_empty(), "WorkflowEvals samples must not be empty");

    // Quick verification on first 5 samples across all methods
    for method in ZevMethod::ALL {
        let mut correct = 0;
        for s in samples.iter().take(5) {
            let req = SystemOneRequest {
                state: s.state.clone(),
                questions: [("q".to_string(), s.question.clone())].into(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                if let Some(ans) = resp.answers.get("q") {
                    let pred = extract_prediction_token(ans);
                    if matches_expected(&pred, &s.expected) {
                        correct += 1;
                    }
                }
            }
        }
        assert!(correct >= 1, "Method {:?} should get non-zero correct answers", method);
    }
}

#[test]
#[ignore = "Comprehensive multi-method evaluation on TypeSafe WorkflowEvals suite"]
fn test_workflowevals_full_all_methods() {
    let engine = ZevEngine::default();
    let samples = load_workflowevals_samples();
    let total = samples.len();

    println!("\n==============================================================================================");
    println!("             TYPESAFE WORKFLOWEVALS BENCHMARK ACROSS ALL ZEV EXECUTION METHODS                ");
    println!("==============================================================================================");
    println!("{:<28} | {:<16} | {:<12} | {:<12} | {:<14}", "Method", "Correct / Total", "Accuracy (%)", "Avg Latency", "Throughput");
    println!("─────────────────────────────+──────────────────+──────────────+──────────────+───────────────");

    for method in ZevMethod::ALL {
        let mut correct = 0;
        let t0 = Instant::now();
        for s in &samples {
            let req = SystemOneRequest {
                state: s.state.clone(),
                questions: [("q".to_string(), s.question.clone())].into(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                if let Some(ans) = resp.answers.get("q") {
                    let pred = extract_prediction_token(ans);
                    if matches_expected(&pred, &s.expected) {
                        correct += 1;
                    }
                }
            }
        }
        let elapsed = t0.elapsed();
        let acc = (correct as f64 / total as f64) * 100.0;
        let avg_lat_us = elapsed.as_micros() as f64 / total as f64;
        let throughput = total as f64 / elapsed.as_secs_f64();

        println!(
            "{:<28} | {:>6}/{:<6}    | {:>8.2}%   | {:>8.2} µs | {:>8.0} dec/s",
            method.display_name(),
            correct,
            total,
            acc,
            avg_lat_us,
            throughput
        );
    }
    println!("==============================================================================================\n");
}

// -------------------------------------------------------------------------------------------------
// 3. Decision Index Multi-Method Evaluation
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DecisionIndexTestCase {
    #[allow(dead_code)]
    id: String,
    benchmark: String,
    state: Value,
    questions: BTreeMap<String, WireQuestion>,
    expected: BTreeMap<String, String>,
}

fn load_decision_index_fixtures() -> Vec<DecisionIndexTestCase> {
    let raw = include_str!("fixtures/decision_index_regression.json");
    serde_json::from_str(raw).expect("Deserialize decision_index_regression.json")
}

#[test]
fn test_decision_index_sample_across_methods() {
    let engine = ZevEngine::default();
    let cases = load_decision_index_fixtures();
    let simplebench: Vec<_> = cases.into_iter().filter(|c| c.benchmark == "SimpleBench").collect();

    for method in ZevMethod::ALL {
        let mut correct = 0;
        for c in &simplebench {
            let req = SystemOneRequest {
                state: c.state.clone(),
                questions: c.questions.clone(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                for (qid, exp) in &c.expected {
                    if let Some(ans) = resp.answers.get(qid) {
                        let pred = extract_prediction_token(ans);
                        if matches_expected(&pred, exp) {
                            correct += 1;
                        }
                    }
                }
            }
        }
        assert!(correct >= 3, "Method {:?} should pass at least 3/10 on SimpleBench", method);
    }
}

#[test]
#[ignore = "Comprehensive multi-method evaluation on Decision Index regression suite (335 items)"]
fn test_decision_index_full_all_methods() {
    let engine = ZevEngine::default();
    let cases = load_decision_index_fixtures();

    println!("\n==============================================================================================");
    println!("              HUGGING FACE DECISION INDEX ACROSS ALL ZEV EXECUTION METHODS                    ");
    println!("==============================================================================================");
    println!("{:<28} | {:<16} | {:<12} | {:<12} | {:<14}", "Method", "Correct / Total", "Accuracy (%)", "Avg Latency", "Throughput");
    println!("─────────────────────────────+──────────────────+──────────────+──────────────+───────────────");

    for method in ZevMethod::ALL {
        let mut total_q = 0;
        let mut correct = 0;
        let mut by_bench: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let t0 = Instant::now();

        for c in &cases {
            let req = SystemOneRequest {
                state: c.state.clone(),
                questions: c.questions.clone(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                for (qid, exp) in &c.expected {
                    total_q += 1;
                    let stat = by_bench.entry(c.benchmark.clone()).or_insert((0, 0));
                    stat.1 += 1;
                    if let Some(ans) = resp.answers.get(qid) {
                        let pred = extract_prediction_token(ans);
                        if matches_expected(&pred, exp) {
                            correct += 1;
                            stat.0 += 1;
                        }
                    }
                }
            }
        }

        let elapsed = t0.elapsed();
        let acc = (correct as f64 / total_q as f64) * 100.0;
        let avg_lat_us = elapsed.as_micros() as f64 / total_q.max(1) as f64;
        let throughput = total_q as f64 / elapsed.as_secs_f64();

        println!(
            "{:<28} | {:>6}/{:<6}    | {:>8.2}%   | {:>8.2} µs | {:>8.0} dec/s",
            method.display_name(),
            correct,
            total_q,
            acc,
            avg_lat_us,
            throughput
        );
        for (bench, (b_corr, b_tot)) in &by_bench {
            let b_acc = (*b_corr as f64 / *b_tot as f64) * 100.0;
            println!("   ↳ {:<25}: {:>4}/{:<4} ({:>5.1}%)", bench, b_corr, b_tot, b_acc);
        }
    }
    println!("==============================================================================================\n");
}

// -------------------------------------------------------------------------------------------------
// 4. Real 231 JevBench Suite Multi-Method Evaluation
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct JevBenchItem {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    family: String,
    state: Value,
    question: WireQuestion,
    expected: Value,
}

fn load_jevbench_231() -> Vec<JevBenchItem> {
    let mut items = Vec::new();
    let paths = [
        "datasets/jevbench_public/easy.jsonl",
        "datasets/jevbench_public/original.jsonl",
        "datasets/jevbench_public/hard.jsonl",
    ];

    for path_str in paths {
        let path = std::path::Path::new(path_str);
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            for line_res in reader.lines() {
                if let Ok(line) = line_res {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(item) = serde_json::from_str::<JevBenchItem>(trimmed) {
                            items.push(item);
                        }
                    }
                }
            }
        }
    }
    items
}

#[test]
fn test_jevbench_sample_across_methods() {
    let engine = ZevEngine::default();
    let items = load_jevbench_231();
    if items.is_empty() {
        return;
    }

    for method in ZevMethod::ALL {
        let mut correct = 0;
        for item in items.iter().take(5) {
            let req = SystemOneRequest {
                state: item.state.clone(),
                questions: [("q".to_string(), item.question.clone())].into(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                if let Some(ans) = resp.answers.get("q") {
                    let pred = extract_prediction_token(ans);
                    let exp_str = match &item.expected {
                        Value::String(s) => s.clone(),
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    };
                    if matches_expected(&pred, &exp_str) {
                        correct += 1;
                    }
                }
            }
        }
        assert!(correct >= 1, "Method {:?} should get non-zero on JevBench sample", method);
    }
}

#[test]
#[ignore = "Comprehensive multi-method evaluation on real 231 JevBench suite"]
fn test_jevbench_231_full_all_methods() {
    let engine = ZevEngine::default();
    let items = load_jevbench_231();
    if items.is_empty() {
        println!("Note: datasets/jevbench_public/ not found, skipping.");
        return;
    }
    let total = items.len();

    println!("\n==============================================================================================");
    println!("                 REAL 231 JEVBENCH SUITE ACROSS ALL ZEV EXECUTION METHODS                     ");
    println!("==============================================================================================");
    println!("{:<28} | {:<16} | {:<12} | {:<12} | {:<14}", "Method", "Correct / Total", "Accuracy (%)", "Avg Latency", "Throughput");
    println!("─────────────────────────────+──────────────────+──────────────+──────────────+───────────────");

    for method in ZevMethod::ALL {
        let mut correct = 0;
        let t0 = Instant::now();

        for item in &items {
            let req = SystemOneRequest {
                state: item.state.clone(),
                questions: [("q".to_string(), item.question.clone())].into(),
                model: method.model_name().into(),
            };
            if let Ok(resp) = engine.evaluate_system_one(&req) {
                if let Some(ans) = resp.answers.get("q") {
                    let pred = extract_prediction_token(ans);
                    let exp_str = match &item.expected {
                        Value::String(s) => s.clone(),
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    };
                    if matches_expected(&pred, &exp_str) {
                        correct += 1;
                    }
                }
            }
        }

        let elapsed = t0.elapsed();
        let acc = (correct as f64 / total as f64) * 100.0;
        let avg_lat_us = elapsed.as_micros() as f64 / total as f64;
        let throughput = total as f64 / elapsed.as_secs_f64();

        println!(
            "{:<28} | {:>6}/{:<6}    | {:>8.2}%   | {:>8.2} µs | {:>8.0} dec/s",
            method.display_name(),
            correct,
            total,
            acc,
            avg_lat_us,
            throughput
        );
    }
    println!("==============================================================================================\n");
}
