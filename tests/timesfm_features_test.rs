use std::collections::BTreeMap;
use zev::{
    check_numeric_guardrails, CascadeStage, NumericGuardrailConfig,
    PredicateCascade, Question, RunningStats, ScoreQuestion, SequentialCascadeRunner, StageKind,
    TabularBatch, TabularEngine, TabularFilterPredicate, TabularRow, ZevEngine, ZevRequest,
};

#[test]
fn test_feature_1_numeric_guardrails() {
    let config = NumericGuardrailConfig::default();

    // 1. Empty input
    let res_empty = check_numeric_guardrails(&[], &config);
    assert!(!res_empty.passed);
    assert!(res_empty.should_abstain);
    assert!(res_empty.is_flatline);

    // 2. High NaN ratio
    let nan_data = vec![1.0, f64::NAN, f64::NAN, 2.0]; // 50% NaN > 30%
    let res_nan = check_numeric_guardrails(&nan_data, &config);
    assert!(!res_nan.passed);
    assert!(res_nan.should_abstain);
    assert!(res_nan.nan_ratio > 0.3);

    // 3. Degenerate flatline (zero variance)
    let flat_data = vec![42.0, 42.0, 42.0, 42.0];
    let res_flat = check_numeric_guardrails(&flat_data, &config);
    assert!(!res_flat.passed);
    assert!(res_flat.should_abstain);
    assert!(res_flat.is_flatline);
    assert_eq!(res_flat.variance, 0.0);

    // 4. Valid non-degenerate input
    let valid_data = vec![10.0, 20.0, 30.0, 40.0];
    let res_valid = check_numeric_guardrails(&valid_data, &config);
    assert!(res_valid.passed);
    assert!(!res_valid.should_abstain);
    assert!(!res_valid.is_flatline);
    assert!(res_valid.variance > 0.0);
}

#[test]
fn test_feature_2_quantile_volatility_spread() {
    let engine = ZevEngine::default();
    let state = "The application response time has degraded significantly under peak load.";

    let q = Question::Score(ScoreQuestion {
        instructions: "Rate degradation severity".into(),
        levels: vec![
            "Severity 0: Normal performance".into(),
            "Severity 1: Minor latency increase".into(),
            "Severity 2: Elevated latency and packet drops".into(),
            "Severity 3: Critical service outage".into(),
        ],
        policy: Default::default(),
    });

    let mut questions = BTreeMap::new();
    questions.insert("severity".into(), q);

    let req = ZevRequest {
        state: serde_json::json!(state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    let resp = engine.evaluate(&req).unwrap();
    let ans = resp.answers.get("severity").unwrap();

    // Verify quantile_spread (p90 - p10) is populated
    assert!(ans.uncertainty.quantile_spread.is_some());
    let spread = ans.uncertainty.quantile_spread.unwrap();
    assert!(spread >= 0.0, "Spread should be non-negative");

    if let Some(ref stats) = ans.statistics {
        let expected_spread = stats.quantiles["p90"] - stats.quantiles["p10"];
        assert!((spread - expected_spread).abs() < 1e-6);
    }
}

#[test]
fn test_feature_3_streaming_welford_statistics() {
    // 1. Standalone RunningStats tests
    let mut stats = RunningStats::new();
    let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    for &x in &data {
        stats.update(x);
    }
    // Mean of data = 40 / 8 = 5.0
    assert_eq!(stats.count, 8);
    assert!((stats.mean - 5.0).abs() < 1e-9);
    // Sample variance of [2, 4, 4, 4, 5, 5, 7, 9] = 32 / 7 = 4.571428...
    assert!((stats.variance() - 32.0 / 7.0).abs() < 1e-9);
    assert_eq!(stats.min, 2.0);
    assert_eq!(stats.max, 9.0);

    // 2. Integration with TabularEngine
    let engine = TabularEngine::default();
    let batch = TabularBatch::new(vec![
        TabularRow::new("row1", "Database cluster connection timeout on replica 2"),
        TabularRow::new("row2", "Customer asked for billing receipt email"),
        TabularRow::new("row3", "CRITICAL: Out of memory crash on web server"),
    ]);

    let predicate = TabularFilterPredicate::new(
        "Is this an infrastructure outage incident?",
        "Yes, database or server crash / outage",
        "No, user question or billing request",
    )
    .with_threshold(0.5);

    let (filtered, report) = engine.filter_batch(&batch, &predicate).unwrap();
    assert_eq!(filtered.len(), 2);
    assert!(report.confidence_stats.is_some());

    let conf_stats = report.confidence_stats.unwrap();
    assert_eq!(conf_stats.count, 3);
    assert!(conf_stats.mean > 0.0 && conf_stats.mean <= 1.0);
    assert!(conf_stats.min >= 0.0 && conf_stats.max <= 1.0);
}

#[test]
fn test_feature_4_consecutive_step_gating() {
    let engine = ZevEngine::default();

    // Stage 1 requires 3 consecutive matches before triggering an alert/pass
    let stage1 = CascadeStage::new(
        "high_cpu_stage",
        StageKind::SimdFilter,
        "Is CPU utilization dangerously elevated?",
        "High CPU utilization above 90%",
        "Normal CPU utilization",
    )
    .with_cost(1.0)
    .with_selectivity(0.3)
    .with_threshold(0.5)
    .with_consecutive_required(3);

    let stage2 = CascadeStage::new(
        "secondary_verification",
        StageKind::DetailedEvaluator,
        "Is process worker pool saturated?",
        "Worker pool saturation",
        "Normal worker pool",
    )
    .with_cost(50.0)
    .with_selectivity(0.9)
    .with_threshold(0.5);

    let cascade = PredicateCascade::new(vec![stage1, stage2]);
    let mut runner = SequentialCascadeRunner::new(cascade);

    let high_cpu_state = "Host telemetry: CPU usage at 98% for process worker_pool saturation";
    let normal_cpu_state = "Host telemetry: CPU usage normal at 22%";

    // Step 1: High CPU -> streak 1 < 3 -> should NOT pass (short-circuit)
    let rep1 = runner.evaluate_step(&engine, high_cpu_state).unwrap();
    assert!(!rep1.passed, "Step 1 (streak 1/3) should not pass yet");
    assert!(rep1.short_circuited);
    assert_eq!(runner.streaks["high_cpu_stage"], 1);

    // Step 2: High CPU -> streak 2 < 3 -> should NOT pass yet
    let rep2 = runner.evaluate_step(&engine, high_cpu_state).unwrap();
    assert!(!rep2.passed, "Step 2 (streak 2/3) should not pass yet");
    assert_eq!(runner.streaks["high_cpu_stage"], 2);

    // Step 3: High CPU -> streak 3 >= 3 -> PASS!
    let rep3 = runner.evaluate_step(&engine, high_cpu_state).unwrap();
    assert!(rep3.passed, "Step 3 (streak 3/3) should successfully pass!");
    assert!(!rep3.short_circuited);
    assert_eq!(runner.streaks["high_cpu_stage"], 3);

    // Step 4: Normal CPU -> streak broken -> reset to 0 and fail
    let rep4 = runner.evaluate_step(&engine, normal_cpu_state).unwrap();
    assert!(!rep4.passed);
    assert_eq!(runner.streaks["high_cpu_stage"], 0);

    // Step 5: Test reset_streaks()
    runner.streaks.insert("high_cpu_stage".into(), 5);
    runner.reset_streaks();
    assert!(runner.streaks.is_empty());
}

#[test]
fn test_running_stats_edge_cases_and_non_finite() {
    let mut stats = RunningStats::new();
    assert_eq!(stats.count, 0);
    assert_eq!(stats.variance(), 0.0);
    assert_eq!(stats.stddev(), 0.0);

    // Single item
    stats.update(10.0);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.mean, 10.0);
    assert_eq!(stats.variance(), 0.0);
    assert_eq!(stats.stddev(), 0.0);
    assert_eq!(stats.min, 10.0);
    assert_eq!(stats.max, 10.0);

    // Non-finite values must be ignored without updating count or state
    stats.update(f64::NAN);
    stats.update(f64::INFINITY);
    stats.update(f64::NEG_INFINITY);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.mean, 10.0);

    // Second finite item
    stats.update(20.0);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.mean, 15.0);
    assert_eq!(stats.variance(), 50.0);
    assert!((stats.stddev() - 50.0_f64.sqrt()).abs() < 1e-9);
    assert_eq!(stats.min, 10.0);
    assert_eq!(stats.max, 20.0);
}

#[test]
fn test_numeric_guardrails_edge_cases() {
    let config = NumericGuardrailConfig {
        min_length: 2,
        min_variance: 1e-3,
        max_nan_ratio: 0.1, // Strict: only 10% allowed
    };

    // Single item: cannot compute variance
    let res_single = check_numeric_guardrails(&[5.0], &config);
    assert!(!res_single.passed);
    assert!(res_single.should_abstain);
    assert!(res_single.is_flatline);

    // All NaNs
    let res_all_nan = check_numeric_guardrails(&[f64::NAN, f64::NAN], &config);
    assert!(!res_all_nan.passed);
    assert!(res_all_nan.should_abstain);
    assert_eq!(res_all_nan.nan_ratio, 1.0);

    // Infinities count towards nan_ratio
    let res_inf = check_numeric_guardrails(&[1.0, 2.0, f64::INFINITY], &config);
    assert!(!res_inf.passed);
    assert!(res_inf.nan_ratio > 0.1);

    // Low variance below min_variance threshold
    let res_low_var = check_numeric_guardrails(&[1.00001, 1.00002, 1.00001], &config);
    assert!(!res_low_var.passed);
    assert!(res_low_var.is_flatline);
}

#[test]
fn test_tabular_engine_batch_operations_with_stats() {
    let engine = TabularEngine::default();
    let batch = TabularBatch::new(vec![
        TabularRow::new("row1", "Production payment gateway is responding with HTTP 500"),
        TabularRow::new("row2", "User changed billing address"),
    ]);

    // 1. score_batch
    let (scores, rep_score) = engine.score_batch(
        &batch,
        "Is this an urgent incident?",
        "Urgent incident",
        "Routine event",
    ).unwrap();
    assert_eq!(scores.len(), 2);
    assert!(rep_score.confidence_stats.is_some());
    let score_stats = rep_score.confidence_stats.unwrap();
    assert_eq!(score_stats.count, 2);

    // 2. route_batch
    let mut routes = BTreeMap::new();
    routes.insert("devops".to_string(), "Infrastructure errors and outages".to_string());
    routes.insert("support".to_string(), "Customer account and billing inquiries".to_string());
    let (routed, rep_route) = engine.route_batch(&batch, &routes).unwrap();
    assert_eq!(routed.len(), 2);
    assert!(rep_route.confidence_stats.is_some());
    let route_stats = rep_route.confidence_stats.unwrap();
    assert_eq!(route_stats.count, 2);

    // 3. join_batch
    let partners = TabularBatch::new(vec![
        TabularRow::new("p1", "HTTP 500 error on payment gateway"),
        TabularRow::new("p2", "Account address modification"),
    ]);
    let (matches, rep_join) = engine.join_batch(
        &batch,
        &partners,
        "Determine if the partner event matches the state incident",
        0.5,
    ).unwrap();
    assert!(!matches.is_empty());
    assert!(rep_join.confidence_stats.is_some());
    let join_stats = rep_join.confidence_stats.unwrap();
    assert_eq!(join_stats.count, 4); // 2x2 cross product
}

#[test]
fn test_cascade_optimization_and_serialization() {
    let s_expensive = CascadeStage::new(
        "neural_deep",
        StageKind::DetailedEvaluator,
        "Is there a critical infrastructure failure?",
        "Critical failure",
        "Normal operation",
    )
    .with_cost(100.0)
    .with_selectivity(0.8)
    .with_threshold(0.5);

    let s_cheap = CascadeStage::new(
        "keyword_filter",
        StageKind::RuleCheck,
        "Is there an error or outage reported?",
        "Error or outage failure",
        "Normal operation",
    )
    .with_cost(0.5)
    .with_selectivity(0.1)
    .with_threshold(0.5);

    let cascade = PredicateCascade::new(vec![s_expensive, s_cheap]);
    // optimize_order should have sorted cheap filter first
    assert_eq!(cascade.stages[0].name, "keyword_filter");
    assert_eq!(cascade.stages[1].name, "neural_deep");

    // Serde roundtrip
    let json_str = serde_json::to_string(&cascade).unwrap();
    let deserialized: PredicateCascade = serde_json::from_str(&json_str).unwrap();
    assert_eq!(deserialized.stages.len(), 2);
    assert_eq!(deserialized.stages[0].name, "keyword_filter");

    // Test direct cascade.evaluate
    let engine = ZevEngine::default();
    // 1. Passing state: has error
    let rep_pass = cascade.evaluate(&engine, "Critical failure: database error connection refused").unwrap();
    assert!(rep_pass.passed);
    assert_eq!(rep_pass.completed_stages, 2);
    assert!(!rep_pass.short_circuited);

    // 2. Early rejection short-circuit: normal operation
    let rep_fail = cascade.evaluate(&engine, "Normal operation: user updated profile picture").unwrap();
    assert!(!rep_fail.passed);
    assert_eq!(rep_fail.completed_stages, 1);
    assert!(rep_fail.short_circuited);
}

#[test]
fn test_tabular_helpers_and_metadata() {
    let mut batch = TabularBatch::default();
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);

    let row = TabularRow::new("row_1", "Some test text")
        .with_metadata("tenant_id", "acme")
        .with_metadata("env", "prod");
    assert_eq!(row.metadata.get("tenant_id").unwrap(), "acme");
    assert_eq!(row.metadata.get("env").unwrap(), "prod");

    batch.push(row);
    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 1);
}
