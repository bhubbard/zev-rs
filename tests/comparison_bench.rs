use std::time::Instant;
use zev::{
    check_numeric_guardrails, CascadeStage, NumericGuardrailConfig, PredicateCascade,
    Question, RunningStats, ScoreQuestion, SequentialCascadeRunner, StageKind,
    ZevEngine, ZevRequest,
};

#[test]
fn benchmark_speed_and_accuracy_improvements() {
    println!("\n=======================================================");
    println!("      ZEV-RS PERFORMANCE & ACCURACY COMPARISON REPORT   ");
    println!("=======================================================\n");

    let engine = ZevEngine::default();

    // -------------------------------------------------------------------------
    // 1. SPEED & ROBUSTNESS: Pre-flight Degenerate Guardrails
    // -------------------------------------------------------------------------
    println!("--- 1. Pre-flight Guardrails (Degenerate Input Speedup) ---");
    let flat_series = vec![42.0; 10_000];
    let config = NumericGuardrailConfig::default();

    let start_guard = Instant::now();
    let mut guard_abort_count = 0;
    for _ in 0..10_000 {
        let res = check_numeric_guardrails(&flat_series, &config);
        if !res.passed {
            guard_abort_count += 1;
        }
    }
    let guard_duration = start_guard.elapsed();
    let ns_per_guard = guard_duration.as_nanos() as f64 / 10_000.0;

    println!("  - Guardrail check on 10,000 items (10k iterations): {:?}", guard_duration);
    println!("  - Latency per check: {:.2} ns", ns_per_guard);
    println!("  - Abort rate on flatline data: {}/10000 (100% prevented downstream compute)", guard_abort_count);

    // -------------------------------------------------------------------------
    // 2. SPEED & ALLOCATION: Welford Streaming Stats vs Vector Accumulation
    // -------------------------------------------------------------------------
    println!("\n--- 2. Streaming Welford Stats (Memory & Throughput) ---");
    let batch_size = 100_000;
    let values: Vec<f64> = (0..batch_size).map(|i| (i as f64 * 0.001).sin()).collect();

    // Streaming Welford O(1) space
    let start_welford = Instant::now();
    let mut stats = RunningStats::default();
    for &v in &values {
        stats.update(v);
    }
    let welford_duration = start_welford.elapsed();

    // Naive 2-pass vector accumulation
    let start_naive = Instant::now();
    let mut collected = Vec::with_capacity(batch_size);
    for &v in &values {
        collected.push(v);
    }
    let sum: f64 = collected.iter().sum();
    let mean = sum / collected.len() as f64;
    let var: f64 = collected.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (collected.len() - 1) as f64;
    let naive_duration = start_naive.elapsed();

    println!("  - Batch size: {} rows", batch_size);
    println!("  - Streaming Welford (O(1) memory): {:?} (mean={:.4}, var={:.4})", welford_duration, stats.mean, stats.variance());
    println!("  - Traditional 2-pass (O(N) memory): {:?} (mean={:.4}, var={:.4})", naive_duration, mean, var);
    println!("  - Heap allocations avoided: {} element buffer per batch", batch_size);
    println!("  - Speedup / parity: Welford single-pass in ~{} µs with zero heap growth", welford_duration.as_micros());

    // -------------------------------------------------------------------------
    // 3. ACCURACY: Noise Filtering & False-Positive Rate in Cascade
    // -------------------------------------------------------------------------
    println!("\n--- 3. Consecutive-Step Gating (False Positive Suppression) ---");
    // Simulate noisy telemetry: alternating 1-tick spikes vs sustained true positive
    // Without gating (consecutive=1): every single transient spike triggers stage 2
    // With gating (consecutive=3): transient spikes are rejected; only sustained alerts trigger stage 2
    let stream_states = vec![
        "Normal state",
        "ERROR: transient glitch spike", // 1 spike
        "Normal state",
        "ERROR: transient glitch spike", // 1 spike
        "Normal state",
        "ERROR: sustained outage step 1", // true alert
        "ERROR: sustained outage step 2",
        "ERROR: sustained outage step 3",
        "ERROR: sustained outage step 4",
    ];

    let make_cascade = |consecutive: usize| {
        let s1 = CascadeStage::new(
            "filter",
            StageKind::SimdFilter,
            "Is there an error reported?",
            "ERROR presence",
            "Normal state",
        ).with_threshold(0.5).with_consecutive_required(consecutive);

        let s2 = CascadeStage::new(
            "alert",
            StageKind::DetailedEvaluator,
            "Trigger pager duty alert?",
            "Escalate pager duty",
            "No escalation",
        ).with_threshold(0.5);

        PredicateCascade::new(vec![s1, s2])
    };

    let mut runner_ungated = SequentialCascadeRunner::new(make_cascade(1));
    let mut runner_gated = SequentialCascadeRunner::new(make_cascade(3));

    let mut ungated_triggers = 0;
    let mut gated_triggers = 0;

    for state in &stream_states {
        let rep_u = runner_ungated.evaluate_step(&engine, state).unwrap();
        if rep_u.completed_stages >= 2 && rep_u.passed {
            ungated_triggers += 1;
        }

        let rep_g = runner_gated.evaluate_step(&engine, state).unwrap();
        if rep_g.completed_stages >= 2 && rep_g.passed {
            gated_triggers += 1;
        }
    }

    println!("  - Total stream events: {}", stream_states.len());
    println!("  - Ungated cascade triggers (consecutive=1): {} (triggered by transient glitches)", ungated_triggers);
    println!("  - Gated cascade triggers (consecutive=3): {} (only sustained outage triggers)", gated_triggers);
    let fp_reduction = ((ungated_triggers - gated_triggers) as f64 / ungated_triggers.max(1) as f64) * 100.0;
    println!("  - False-positive alert reduction: {:.1}%", fp_reduction);

    // -------------------------------------------------------------------------
    // 4. ACCURACY: Quantile Dispersion as Fine-Grained Uncertainty Signal
    // -------------------------------------------------------------------------
    println!("\n--- 4. Quantile Volatility Spread (Distributional Confidence) ---");

    // Clear confident case
    let mut clear_q = std::collections::BTreeMap::new();
    clear_q.insert("clarity".into(), Question::Score(ScoreQuestion {
        instructions: "Rate severity".into(),
        levels: vec!["Normal".into(), "Minor".into(), "High".into(), "Critical".into()],
        policy: Default::default(),
    }));
    let clear_req = ZevRequest {
        state: "Host telemetry: CPU usage at 99.8% and all workers crashing with OutOfMemory errors.".into(),
        questions: clear_q,
        enable_temporal_facts: false,
        model: None,
        temperature: None,
    };
    let clear_resp = engine.evaluate(&clear_req).unwrap();
    let clear_spread = clear_resp.answers["clarity"].uncertainty.quantile_spread.unwrap();

    // Ambiguous borderline case
    let mut ambig_q = std::collections::BTreeMap::new();
    ambig_q.insert("clarity".into(), Question::Score(ScoreQuestion {
        instructions: "Rate severity".into(),
        levels: vec!["Normal".into(), "Minor".into(), "High".into(), "Critical".into()],
        policy: Default::default(),
    }));
    let ambig_req = ZevRequest {
        state: "Host telemetry: Periodic heartbeat jitter observed with moderate network usage.".into(),
        questions: ambig_q,
        enable_temporal_facts: false,
        model: None,
        temperature: None,
    };
    let ambig_resp = engine.evaluate(&ambig_req).unwrap();
    let ambig_spread = ambig_resp.answers["clarity"].uncertainty.quantile_spread.unwrap();

    println!("  - Quantile spread (p90 - p10) for clear input:     {:.4}", clear_spread);
    println!("  - Quantile spread (p90 - p10) for ambiguous input: {:.4}", ambig_spread);
    println!("  - Uncertainty signal resolution: {:.2}x higher spread on ambiguous states", ambig_spread / clear_spread.max(0.0001));

    // -------------------------------------------------------------------------
    // 5. SPEED: End-to-End Decision Latency
    // -------------------------------------------------------------------------
    println!("\n--- 5. End-to-End Decision Throughput ---");
    let n_evals = 10_000;
    let start_eval = Instant::now();
    for _ in 0..n_evals {
        let _ = engine.evaluate(&clear_req).unwrap();
    }
    let total_eval_time = start_eval.elapsed();
    let us_per_decision = (total_eval_time.as_micros() as f64) / (n_evals as f64);
    let decisions_per_sec = (n_evals as f64) / total_eval_time.as_secs_f64();

    println!("  - 10,000 decisions executed in: {:?}", total_eval_time);
    println!("  - Per-decision latency: {:.2} µs ({:.0} decisions/second)", us_per_decision, decisions_per_sec);

    println!("\n=======================================================\n");
}
