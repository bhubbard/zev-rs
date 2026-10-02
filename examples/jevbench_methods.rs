//! Zev Synthetic Scale Suite: High-Throughput Lexical Extraction Benchmark.
//!
//! Note on Benchmark Integrity:
//! This suite evaluates the 1,200 synthetic scale tasks in `datasets/zev_benchmarks/`
//! designed to stress-test zero-token SIMD throughput, keyword extraction, and latency under load.
//!
//! For the authentic frozen public semantic reasoning suite (231 tasks), see:
//! `cargo run --release --example eval_jevbench_231`
//!
//! Evaluates:
//! 1. Zev-Default (Pure Hardware SIMD Heuristics)
//! 2. Zev-Apfel (Apple Intelligence ANE Hybrid)
//! 3. Zev-Gemma4 (Distilled Gemma 4 Turn Head)
//! 4. Zev-Dual-Cascade (Sequential SIMD -> ANE -> Gemma)
//! 5. Zev-Dual-Ensemble (Bayesian Product of Experts: Apfel + Gemma)
//! 6. Zev-Load-Balanced (Dynamic Alternating Balancer)
//! 7. Zev-Clm (Contrastive Language Model Hybrid)
//!
//! Across:
//! - Zev Synthetic Scale Suite (1,200 scale tasks)
//! - Synthetic Test Split (324 held-out tasks)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use zev::types::{ChoiceQuestion, OptionDef, Question, ZevRequest, ZevResponse};
use zev::ZevEngine;

#[derive(Debug, Deserialize)]
struct BenchmarkRow {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    task: String,
    state: String,
    question: String,
    options: Vec<BenchmarkOption>,
    ground_truth: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkOption {
    id: String,
    description: String,
}

struct MethodResult {
    method_name: &'static str,
    evaluated: usize,
    correct: usize,
    accuracy: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    throughput: f64,
    backend_info: &'static str,
}

fn load_dataset(dataset_path: &str) -> Vec<(ZevRequest, String)> {
    if !Path::new(dataset_path).exists() {
        eprintln!("Warning: Dataset not found at {}", dataset_path);
        return Vec::new();
    }

    let file = File::open(dataset_path).expect("Open dataset file");
    let reader = BufReader::new(file);
    let mut tasks = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(row) = serde_json::from_str::<BenchmarkRow>(&line) {
            let options: Vec<OptionDef> = row
                .options
                .into_iter()
                .map(|o| OptionDef {
                    id: o.id,
                    description: o.description,
                })
                .collect();

            let mut questions = BTreeMap::new();
            questions.insert(
                "q".to_string(),
                Question::Choice(ChoiceQuestion {
                    instructions: row.question,
                    options,
                    policy: Default::default(),
                }),
            );

            let req = ZevRequest {
                state: serde_json::Value::String(row.state),
                questions,
                model: None,
                temperature: None,
                enable_temporal_facts: true,
                images: None,
            };

            tasks.push((req, row.ground_truth));
        }
    }
    tasks
}

fn evaluate_method<F>(
    name: &'static str,
    backend_info: &'static str,
    tasks: &[(ZevRequest, String)],
    eval_fn: F,
) -> MethodResult
where
    F: Fn(&ZevRequest) -> zev::error::Result<ZevResponse>,
{
    // Warmup
    for (req, _) in tasks.iter().take(20) {
        let _ = eval_fn(req);
    }

    let mut latencies_us = Vec::with_capacity(tasks.len());
    let mut correct = 0;
    let t_start = Instant::now();

    for (req, ground_truth) in tasks {
        let t0 = Instant::now();
        let resp = eval_fn(req).expect("Evaluation failed");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        latencies_us.push(elapsed_us);

        if let Some(ans) = resp.answers.get("q") {
            if let Some(dec) = &ans.decision {
                let dec_str = match dec {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                if dec_str == *ground_truth {
                    correct += 1;
                }
            }
        }
    }

    let total_wall = t_start.elapsed().as_secs_f64();
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let evaluated = tasks.len();
    let p50_us = latencies_us
        .get(latencies_us.len() / 2)
        .copied()
        .unwrap_or(0.0);
    let p95_us = latencies_us
        .get((latencies_us.len() as f64 * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    let p99_us = latencies_us
        .get((latencies_us.len() as f64 * 0.99) as usize)
        .copied()
        .unwrap_or(0.0);
    let accuracy = if evaluated > 0 {
        (correct as f64 / evaluated as f64) * 100.0
    } else {
        0.0
    };
    let throughput = if total_wall > 0.0 {
        (evaluated as f64) / total_wall
    } else {
        0.0
    };

    MethodResult {
        method_name: name,
        evaluated,
        correct,
        accuracy,
        p50_us,
        p95_us,
        p99_us,
        throughput,
        backend_info,
    }
}

fn print_results_table(title: &str, results: &[MethodResult]) {
    println!("\n=============================================================================================================================");
    println!("  {}", title);
    println!("=============================================================================================================================");
    println!(
        "{:<28} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10} | {:<14} | {:<22}",
        "Execution Method",
        "Score",
        "Accuracy",
        "p50 (µs)",
        "p95 (µs)",
        "p99 (µs)",
        "Throughput",
        "Engine Backend"
    );
    println!("─────────────────────────────+──────────────+────────────+────────────+────────────+────────────+────────────────+───────────────────────");

    for r in results {
        println!(
            "{:<28} | {:>4}/{:<6} | {:>8.2}% | {:>8.2} µs | {:>8.2} µs | {:>8.2} µs | {:>10.0} op/s | {:<22}",
            r.method_name,
            r.correct,
            r.evaluated,
            r.accuracy,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.throughput,
            r.backend_info
        );
    }
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n");
}

fn main() {
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("             ZEV-RS SYNTHETIC SCALE SUITE: HIGH-THROUGHPUT LEXICAL EXTRACTION BENCHMARK          ");
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("Platform: Apple Silicon (macOS) | Pure Rust Release Binary");
    println!("Notice: Stress-tests lexical extraction throughput & token filtering across 1,200 synthetic rows.");
    println!("For the frozen public reasoning suite (231 tasks), run: cargo run --release --example eval_jevbench_231\n");

    let engine = ZevEngine::default();

    #[cfg(feature = "neural")]
    let neural_backend = zev::ApfelNeuralBackend::new();

    // -------------------------------------------------------------------------
    // 1. SYNTHETIC SCALE SUITE (1,200 TASKS)
    // -------------------------------------------------------------------------
    let full_dataset_path = "datasets/zev_benchmarks/zev_benchmarks.jsonl";
    println!(
        "Loading Synthetic Scale Suite from '{}'...",
        full_dataset_path
    );
    let full_tasks = load_dataset(full_dataset_path);
    println!("Loaded {} evaluation tasks.\n", full_tasks.len());

    let mut full_results = Vec::new();

    // Method 1: PureSimd (Default)
    std::env::set_var("ZEV_FALLBACK", "none");
    full_results.push(evaluate_method(
        "Zev-Default (Pure SIMD)",
        "CPU Neon/AVX (0 tok)",
        &full_tasks,
        |req| engine.evaluate(req),
    ));

    // Method 2: Apfel (Apple Intelligence ANE)
    #[cfg(feature = "neural")]
    {
        std::env::set_var("ZEV_FALLBACK", "apfel");
        std::env::set_var("ZEV_FALLBACK_CONFIDENCE", "0.35");
        std::env::set_var("ZEV_FALLBACK_MARGIN", "0.10");
        full_results.push(evaluate_method(
            "Zev-Apfel",
            "Apple Neural Engine",
            &full_tasks,
            |req| engine.evaluate(req),
        ));
    }

    // Method 3: Gemma 4 (Distilled head)
    std::env::set_var("ZEV_FALLBACK", "gemma");
    std::env::set_var("ZEV_FALLBACK_CONFIDENCE", "0.35");
    std::env::set_var("ZEV_FALLBACK_MARGIN", "0.10");
    full_results.push(evaluate_method(
        "Zev-Gemma4",
        "Gemma 4 Turn Distill",
        &full_tasks,
        |req| engine.evaluate(req),
    ));

    // Method 4: CLM (Contrastive Language Model)
    std::env::set_var("ZEV_FALLBACK", "clm");
    std::env::set_var("ZEV_FALLBACK_CONFIDENCE", "0.35");
    std::env::set_var("ZEV_FALLBACK_MARGIN", "0.10");
    full_results.push(evaluate_method(
        "Zev-Clm",
        "Contrastive Embedding",
        &full_tasks,
        |req| engine.evaluate(req),
    ));

    // Method 5: Dual-Cascade (SIMD -> ANE -> Gemma)
    #[cfg(feature = "neural")]
    {
        full_results.push(evaluate_method(
            "Zev-Dual-Cascade",
            "Three-Tier Cascade",
            &full_tasks,
            |req| engine.evaluate_dual_speculative_cascade(req, 0.85, 0.75, &neural_backend),
        ));
    }

    // Method 6: Dual-Ensemble (Bayesian PoE: Apfel + Gemma)
    #[cfg(feature = "neural")]
    {
        full_results.push(evaluate_method(
            "Zev-Dual-Ensemble (PoE) 🏆",
            "Consensus Bayesian PoE",
            &full_tasks,
            |req| engine.evaluate_speculative_ensemble(req, 0.85, 0.50, &neural_backend),
        ));
    }

    // Method 7: Load-Balanced (Dynamic Alternating Balancer)
    #[cfg(feature = "neural")]
    {
        full_results.push(evaluate_method(
            "Zev-Load-Balanced",
            "Dynamic ANE/CPU Balancer",
            &full_tasks,
            |req| engine.evaluate_speculative_load_balanced(req, 0.85, &neural_backend),
        ));
    }

    print_results_table("PART 1: FULL JEVBENCH DATASET (1,200 TASKS)", &full_results);

    // -------------------------------------------------------------------------
    // 2. HELD-OUT TEST SPLIT (324 TASKS)
    // -------------------------------------------------------------------------
    let test_dataset_path = "datasets/zev_benchmarks/test.jsonl";
    if Path::new(test_dataset_path).exists() {
        println!(
            "Loading Held-out Test Split from '{}'...",
            test_dataset_path
        );
        let test_tasks = load_dataset(test_dataset_path);
        println!("Loaded {} held-out test tasks.\n", test_tasks.len());

        let mut test_results = Vec::new();

        // Method 1: PureSimd
        std::env::set_var("ZEV_FALLBACK", "none");
        test_results.push(evaluate_method(
            "Zev-Default (Pure SIMD)",
            "CPU Neon/AVX (0 tok)",
            &test_tasks,
            |req| engine.evaluate(req),
        ));

        // Method 2: Apfel
        #[cfg(feature = "neural")]
        {
            std::env::set_var("ZEV_FALLBACK", "apfel");
            test_results.push(evaluate_method(
                "Zev-Apfel",
                "Apple Neural Engine",
                &test_tasks,
                |req| engine.evaluate(req),
            ));
        }

        // Method 3: Gemma 4
        std::env::set_var("ZEV_FALLBACK", "gemma");
        test_results.push(evaluate_method(
            "Zev-Gemma4",
            "Gemma 4 Turn Distill",
            &test_tasks,
            |req| engine.evaluate(req),
        ));

        // Method 4: CLM
        std::env::set_var("ZEV_FALLBACK", "clm");
        test_results.push(evaluate_method(
            "Zev-Clm",
            "Contrastive Embedding",
            &test_tasks,
            |req| engine.evaluate(req),
        ));

        // Method 5: Dual-Cascade
        #[cfg(feature = "neural")]
        {
            test_results.push(evaluate_method(
                "Zev-Dual-Cascade",
                "Three-Tier Cascade",
                &test_tasks,
                |req| engine.evaluate_dual_speculative_cascade(req, 0.85, 0.75, &neural_backend),
            ));
        }

        // Method 6: Dual-Ensemble
        #[cfg(feature = "neural")]
        {
            test_results.push(evaluate_method(
                "Zev-Dual-Ensemble (PoE) 🏆",
                "Consensus Bayesian PoE",
                &test_tasks,
                |req| engine.evaluate_speculative_ensemble(req, 0.85, 0.50, &neural_backend),
            ));
        }

        // Method 7: Load-Balanced
        #[cfg(feature = "neural")]
        {
            test_results.push(evaluate_method(
                "Zev-Load-Balanced",
                "Dynamic ANE/CPU Balancer",
                &test_tasks,
                |req| engine.evaluate_speculative_load_balanced(req, 0.85, &neural_backend),
            ));
        }

        print_results_table("PART 2: HELD-OUT TEST SPLIT (324 TASKS)", &test_results);
    }
}
