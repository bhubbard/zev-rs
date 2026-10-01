//! Dynamic Live Evaluation on the Real 231 JevBench Public Tasks.
//!
//! Evaluates the 231 frozen tasks from `datasets/jevbench_public/`:
//! - `easy.jsonl` (48 items: intent, extraction, fact)
//! - `original.jsonl` (72 items: policy, ordinal, adequacy, etc.)
//! - `hard.jsonl` (111 items: routing_hard, judge_hard, multi_hop, etc.)
//!
//! Computes dynamically:
//! - Exact Accuracy per Family (18 task families)
//! - Overall Global Accuracy
//! - Latency Distribution (p50, p95, p99)
//! - Calibration ECE

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use serde_json::Value;
use zev::calibration::compute_ece;
use zev::wire::{wire_to_question, WireQuestion};
use zev::types::ZevRequest;
use zev::ZevEngine;

#[derive(Debug, Deserialize)]
struct JevBenchItem {
#[allow(dead_code)]
    id: String,
    family: String,
    state: Value,
    question: WireQuestion,
    expected: Value,
}

struct FamilyStat {
    total: usize,
    correct: usize,
}

struct EvaluationResult {
    method_name: &'static str,
    correct: usize,
    total: usize,
    accuracy: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    throughput: f64,
    ece: f64,
}

fn evaluate_items<F>(
    name: &'static str,
    tasks: &[JevBenchItem],
    engine: &ZevEngine,
    eval_fn: F,
) -> (EvaluationResult, BTreeMap<String, FamilyStat>)
where
    F: Fn(&ZevEngine, &ZevRequest) -> zev::error::Result<zev::types::ZevResponse>,
{
    let mut family_stats: BTreeMap<String, FamilyStat> = BTreeMap::new();
    let mut latencies_us = Vec::with_capacity(tasks.len());
    let mut confidences = Vec::with_capacity(tasks.len());
    let mut accuracies = Vec::with_capacity(tasks.len());
    let mut total_correct = 0;

    let t_start_all = Instant::now();

    for item in tasks {
        let question = match wire_to_question(&item.question) {
            Ok(q) => q,
            Err(_) => continue,
        };

        let mut questions = BTreeMap::new();
        questions.insert("q".to_string(), question);

        let req = ZevRequest {
            state: item.state.clone(),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: true,
            images: None,
        };

        let t0 = Instant::now();
        let resp = eval_fn(engine, &req).expect("Evaluation failed");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        latencies_us.push(elapsed_us);

        let is_correct = if let Some(ans) = resp.answers.get("q") {
            confidences.push(ans.confidence);

            match &item.expected {
                Value::String(exp_str) => {
                    let is_exp_yes = exp_str == "yes" || exp_str == "true";
                    let is_exp_no = exp_str == "no" || exp_str == "false";

                    if let Some(dec) = &ans.decision {
                        match dec {
                            Value::String(s) => {
                                s == exp_str || (is_exp_yes && (s == "true" || s == "yes")) || (is_exp_no && (s == "false" || s == "no"))
                            }
                            Value::Bool(b) => {
                                if *b { is_exp_yes } else { is_exp_no }
                            }
                            Value::Number(n) => {
                                if let Ok(exp_num) = exp_str.parse::<f64>() {
                                    (n.as_f64().unwrap_or(-999.0) - exp_num).abs() < 0.5
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        }
                    } else {
                        let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);
                        let p_false = ans.probabilities.get("false").copied().unwrap_or(0.0);
                        if is_exp_yes {
                            p_true > p_false
                        } else if is_exp_no {
                            p_false >= p_true
                        } else {
                            false
                        }
                    }
                }
                Value::Bool(exp_bool) => {
                    if let Some(dec) = &ans.decision {
                        match dec {
                            Value::Bool(b) => b == exp_bool,
                            Value::String(s) => {
                                if *exp_bool {
                                    s == "true" || s == "yes"
                                } else {
                                    s == "false" || s == "no"
                                }
                            }
                            _ => false,
                        }
                    } else {
                        let p_true = ans.probabilities.get("true").copied().unwrap_or(0.0);
                        let p_false = ans.probabilities.get("false").copied().unwrap_or(0.0);
                        if *exp_bool {
                            p_true > p_false
                        } else {
                            p_false >= p_true
                        }
                    }
                }
                Value::Number(exp_num) => {
                    if let Some(exp_f) = exp_num.as_f64() {
                        let pred_f = if let Some(Value::Number(n)) = &ans.decision {
                            n.as_f64().unwrap_or(-999.0)
                        } else if let Some(Value::String(s)) = &ans.decision {
                            s.parse::<f64>().unwrap_or(-999.0)
                        } else {
                            let mut best_score = -1.0;
                            let mut best_p = -1.0;
                            for (k, &p) in &ans.probabilities {
                                if let Ok(score_val) = k.parse::<f64>() {
                                    if p > best_p {
                                        best_p = p;
                                        best_score = score_val;
                                    }
                                }
                            }
                            best_score
                        };
                        (pred_f - exp_f).abs() < 0.5
                    } else {
                        false
                    }
                }
                _ => false,
            }
        } else {
            false
        };

        accuracies.push(is_correct);
        if is_correct {
            total_correct += 1;
        }

        let entry = family_stats.entry(item.family.clone()).or_insert(FamilyStat {
            total: 0,
            correct: 0,
        });
        entry.total += 1;
        if is_correct {
            entry.correct += 1;
        }
    }

    let total_wall = t_start_all.elapsed().as_secs_f64();
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50_us = latencies_us.get(latencies_us.len() / 2).copied().unwrap_or(0.0);
    let p95_us = latencies_us.get((latencies_us.len() as f64 * 0.95) as usize).copied().unwrap_or(0.0);
    let p99_us = latencies_us.get((latencies_us.len() as f64 * 0.99) as usize).copied().unwrap_or(0.0);
    let total_eval = tasks.len();
    let accuracy = if total_eval > 0 {
        (total_correct as f64 / total_eval as f64) * 100.0
    } else {
        0.0
    };
    let throughput = if total_wall > 0.0 {
        (total_eval as f64) / total_wall
    } else {
        0.0
    };
    let ece = compute_ece(&confidences, &accuracies, 10);

    (
        EvaluationResult {
            method_name: name,
            correct: total_correct,
            total: total_eval,
            accuracy,
            p50_us,
            p95_us,
            p99_us,
            throughput,
            ece,
        },
        family_stats,
    )
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════════════════");
    println!("             ZEV-RS LIVE DYNAMIC EVALUATION: REAL 231 JEVBENCH FROZEN PUBLIC SUITE            ");
    println!("══════════════════════════════════════════════════════════════════════════════════════════════");
    println!("Platform: Apple Silicon (macOS) | Pure Rust Release Binary");
    println!("Evaluation: 100% Dynamic at Runtime (Zero Hardcoded Scores)\n");

    let engine = ZevEngine::default();

    let files = [
        ("Easy Tier", "datasets/jevbench_public/easy.jsonl"),
        ("Original Tier", "datasets/jevbench_public/original.jsonl"),
        ("Hard Tier", "datasets/jevbench_public/hard.jsonl"),
    ];

    let mut all_tasks = Vec::new();

    for (tier_name, path) in &files {
        if !Path::new(path).exists() {
            eprintln!("Error: Dataset file '{}' not found. Please ensure datasets/jevbench_public is present.", path);
            return;
        }
        let file = File::open(path).expect("Open file");
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            let line = line.expect("Read line");
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(item) = serde_json::from_str::<JevBenchItem>(&line) {
                all_tasks.push(item);
                count += 1;
            }
        }
        println!("Loaded {:>3} items from {} ({})", count, tier_name, path);
    }

    println!("\nTotal JevBench Tasks Loaded: {}\n", all_tasks.len());

    // 1. Zev-Default (Pure SIMD)
    std::env::set_var("ZEV_FALLBACK", "none");
    let (simd_res, family_stats) = evaluate_items("Zev-Default (Pure SIMD)", &all_tasks, &engine, |eng, req| {
        eng.evaluate(req)
    });

    println!("==============================================================================================");
    println!("                   TASK FAMILY ACCURACY BREAKDOWN: ZEV-DEFAULT (PURE SIMD)                    ");
    println!("==============================================================================================");
    println!(
        "{:<24} | {:<8} | {:<16} | {:<12}",
        "Task Family", "Items", "Correct / Total", "Accuracy (%)"
    );
    println!("─────────────────────────+──────────+──────────────────+──────────────");

    for (family, stat) in &family_stats {
        let pct = (stat.correct as f64 / stat.total as f64) * 100.0;
        println!(
            "{:<24} | {:>6}   | {:>6}/{:<6}    | {:>8.2}%",
            family, stat.total, stat.correct, stat.total, pct
        );
    }

    println!("─────────────────────────+──────────+──────────────────+──────────────");
    println!(
        "{:<24} | {:>6}   | {:>6}/{:<6}    | {:>8.2}%",
        "TOTAL (All Families)", simd_res.total, simd_res.correct, simd_res.total, simd_res.accuracy
    );
    println!("==============================================================================================\n");

    println!("==============================================================================================");
    println!("                                   RUNTIME LATENCY & SPEED                                    ");
    println!("==============================================================================================");
    println!("  • Method Evaluated:         {}", simd_res.method_name);
    println!("  • Tasks Evaluated:          {}", simd_res.total);
    println!("  • Median Latency (p50):     {:.2} µs ({:.4} ms)", simd_res.p50_us, simd_res.p50_us / 1000.0);
    println!("  • 95th Percentile (p95):    {:.2} µs ({:.4} ms)", simd_res.p95_us, simd_res.p95_us / 1000.0);
    println!("  • 99th Percentile (p99):    {:.2} µs ({:.4} ms)", simd_res.p99_us, simd_res.p99_us / 1000.0);
    println!("  • Evaluation Throughput:    {:.0} decisions / second", simd_res.throughput);
    println!("  • Expected Calibration (ECE): {:.4} ({:.2}%)", simd_res.ece, simd_res.ece * 100.0);
    println!("==============================================================================================\n");
}
