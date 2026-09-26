use std::collections::BTreeMap;
use zev::{
    AnchorPartnerEvaluator, BinaryReadout, CascadeStage, PagedContextArena, PartnerMatrix,
    PredicateCascade, PrunedHead, StageKind, TabularBatch, TabularEngine, TabularFilterPredicate,
    TabularRow, ZevEngine, DEFAULT_PAGE_SIZE,
};

#[test]
fn test_feature_1_tabular_streaming_engine() {
    let engine = TabularEngine::default();

    let mut batch = TabularBatch::default();
    batch.push(TabularRow::new(
        "row_1",
        "Production PostgreSQL database connection refused 500 error",
    ));
    batch.push(TabularRow::new(
        "row_2",
        "I love the new UI dashboard, great design!",
    ));
    batch.push(TabularRow::new(
        "row_3",
        "General inquiry: how to update account profile settings",
    ));

    // 1. Filter: Find incident / outage rows
    let filter = TabularFilterPredicate::new(
        "Is this text reporting an urgent technical outage or system failure?",
        "Production database, outage, downtime, system failure, 500 connection refused",
        "Compliment, praise, UI dashboard, great design, account login, password reset, or question",
    );
    let (survivors, report) = engine.filter_batch(&batch, &filter).unwrap();
    for row in &batch.rows {
        let (s, _) = engine
            .filter_batch(&TabularBatch::new(vec![row.clone()]), &filter)
            .unwrap();
        println!("row {}: survived? {}", row.id, !s.is_empty());
    }
    println!(
        "test_feature_1 survivors: {:?}",
        survivors.rows.iter().map(|r| &r.id).collect::<Vec<_>>()
    );
    assert_eq!(report.input_rows, 3);
    assert_eq!(survivors.len(), 1);
    assert_eq!(survivors.rows[0].id, "row_1");

    // 2. Route: Route across departments
    let mut routes = BTreeMap::new();
    routes.insert(
        "infra".to_string(),
        "Database, server outages, infrastructure errors".to_string(),
    );
    routes.insert(
        "feedback".to_string(),
        "UI dashboard design, compliments, UX feedback, general appreciation".to_string(),
    );
    routes.insert(
        "auth".to_string(),
        "Login, passwords, account authentication".to_string(),
    );

    let (routed, route_report) = engine.route_batch(&batch, &routes).unwrap();
    println!("routed: {:?}", routed);
    assert_eq!(route_report.input_rows, 3);
    assert_eq!(routed[0].1, "infra");
    assert_eq!(routed[1].1, "feedback");
    assert_eq!(routed[2].1, "auth");

    // 3. Score: Continuous probability
    let (scores, score_report) = engine
        .score_batch(
            &batch,
            "Assess customer sentiment",
            "Positive, pleased, compliment, love, great design",
            "Frustrated, issue, complaint, error, refused, outage",
        )
        .unwrap();
    assert_eq!(score_report.input_rows, 3);
    let row_2_score = scores.iter().find(|(id, _)| id == "row_2").unwrap().1;
    let row_1_score = scores.iter().find(|(id, _)| id == "row_1").unwrap().1;
    assert!(
        row_2_score > row_1_score,
        "Row 2 (compliment) should have higher positive sentiment"
    );
}

#[test]
fn test_feature_2_asymmetric_anchor_partner_matrix() {
    let partners = [
        (
            "billing",
            "Invoices, payments, chargebacks, subscriptions, refunds",
        ),
        (
            "security",
            "Vulnerability report, credentials compromised, malware, breach",
        ),
        (
            "hardware",
            "Broken laptop screen, printer jam, keyboard replacement",
        ),
    ];

    let matrix = PartnerMatrix::from_options(&partners, 512);
    assert_eq!(matrix.num_partners(), 3);
    assert!(matrix.dim() > 0);

    let evaluator = AnchorPartnerEvaluator::new(matrix, 2.179);

    let anchors = [
        (
            "ticket_101",
            "Our office printer is smoking and paper is jammed",
        ),
        (
            "ticket_102",
            "We need a refund for duplicate charges on invoice #8892",
        ),
        (
            "ticket_103",
            "Someone leaked the admin API credentials on pastebin",
        ),
    ];

    let results = evaluator.evaluate_anchors(&anchors).unwrap();
    println!("test_feature_2 results: {:?}", results);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].1, "hardware");
    assert_eq!(results[1].1, "billing");
    assert_eq!(results[2].1, "security");
    assert!(results[0].2 > 0.4);
    assert!(results[1].2 > 0.4);
    assert!(results[2].2 > 0.4);
}

#[test]
fn test_feature_3_kv_rewind_paged_arena() {
    let mut arena = PagedContextArena::new(100, DEFAULT_PAGE_SIZE);
    assert_eq!(arena.free_pages_count(), 100);

    // 1. Allocate prefix for a document of 40 tokens (needs 3 pages of size 16)
    let doc_key = "doc_alpha";
    let table = arena.allocate_prefix(doc_key, 40).unwrap();
    assert_eq!(table.token_count, 40);
    assert_eq!(table.page_ids.len(), 3);
    assert_eq!(arena.free_pages_count(), 97);

    // 2. Append question 1 (10 tokens -> total 50 tokens, needs 4 pages)
    arena.append_suffix(doc_key, 10).unwrap();
    assert_eq!(arena.free_pages_count(), 96);

    // 3. Rewind back to document prefix
    arena.rewind(doc_key).unwrap();
    assert_eq!(arena.free_pages_count(), 97);

    // 4. Append question 2 (5 tokens -> total 45 tokens, fits in existing 3 pages)
    let evaluated = arena
        .with_rewind(doc_key, 5, |t| {
            assert_eq!(t.token_count, 45);
            Ok("result_ok")
        })
        .unwrap();

    assert_eq!(evaluated, "result_ok");
    assert_eq!(arena.free_pages_count(), 97);

    // 5. Free document
    arena.free_key(doc_key).unwrap();
    assert_eq!(arena.free_pages_count(), 100);
}

#[test]
fn test_feature_4_output_head_pruning_and_token_pooling() {
    // Hidden dimension 8, 4 retained token variants:
    // Index 0: "true", Index 1: "yes"
    // Index 2: "false", Index 3: "no"
    let hidden_dim = 8;
    let retained_tokens = vec![101, 102, 201, 202];

    // Weights: positive features on indices 0 & 1, negative on 2 & 3
    let mut weights = vec![0.0f32; 4 * hidden_dim];
    // Row 0 ("true"): positive weight on feature 0
    weights[0] = 1.0;
    // Row 1 ("yes"): positive weight on feature 1
    weights[hidden_dim + 1] = 1.2;
    // Row 2 ("false"): positive weight on feature 2
    weights[2 * hidden_dim + 2] = 1.1;
    // Row 3 ("no"): positive weight on feature 3
    weights[3 * hidden_dim + 3] = 1.3;

    let head = PrunedHead::new(hidden_dim, retained_tokens, weights);
    let readout = BinaryReadout::new(vec![0, 1], vec![2, 3]);

    // Test with affirmative hidden state (activation on feature 1)
    let affirmative_state = vec![0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let logits = head.forward_logits(&affirmative_state).unwrap();
    assert!(readout.decision(&logits));
    let prob_yes = readout.score(&logits);
    assert!(
        prob_yes > 0.8,
        "Affirmative score should be high, got {prob_yes}"
    );

    // Test with negative hidden state (activation on feature 3)
    let negative_state = vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0];
    let logits_neg = head.forward_logits(&negative_state).unwrap();
    assert!(!readout.decision(&logits_neg));
    let prob_no = readout.score(&logits_neg);
    assert!(prob_no < 0.2, "Negative score should be low, got {prob_no}");
}

#[test]
fn test_feature_5_predicate_cascade_early_rejection() {
    let engine = ZevEngine::default();

    // Set up a 3-stage cascade:
    // Stage 1: Broad gating check (cheapest, high rejection rate)
    // Stage 2: Specific domain criteria
    // Stage 3: Fine-grained clinical differential
    let stages = vec![
        CascadeStage::new(
            "medical_context",
            StageKind::RuleCheck,
            "Is this state discussing a patient or clinical health scenario?",
            "Patient, symptoms, clinical condition, medical triage",
            "Software coding, nginx web server, programming, sales, or general conversation",
        )
        .with_cost(0.5)
        .with_selectivity(0.1),
        CascadeStage::new(
            "acute_abdomen",
            StageKind::SimdFilter,
            "Is the patient experiencing severe abdominal symptoms or peritonitis?",
            "Right lower quadrant pain, tenderness, acute abdomen",
            "Mild headache, dermatological rash, or limb injury",
        )
        .with_cost(5.0)
        .with_selectivity(0.3),
        CascadeStage::new(
            "appendicitis_differential",
            StageKind::DetailedEvaluator,
            "Does the patient show signs consistent with acute appendicitis?",
            "McBurney tenderness, peritonitis, localized right lower quadrant pain",
            "Diffuse gastroenteritis with diarrhea",
        )
        .with_cost(20.0)
        .with_selectivity(0.5),
    ];

    let cascade = PredicateCascade::new(stages);

    // Ensure stages are ordered by cost-benefit ratio (lowest cost / reject rate first)
    assert_eq!(cascade.stages[0].name, "medical_context");

    // Case A: Completely non-medical state -> should short-circuit at Stage 1
    let non_medical = "How do I configure nginx SSL reverse proxy with Let's Encrypt?";
    let report_a = cascade.evaluate(&engine, non_medical).unwrap();
    assert!(!report_a.passed);
    assert!(report_a.short_circuited);
    assert_eq!(
        report_a.completed_stages, 1,
        "Should short-circuit after Stage 1"
    );

    // Case B: Clinical acute appendicitis scenario -> passes all 3 stages
    let medical_state = "Patient reports acute right lower quadrant pain with McBurney tenderness, fever, and nausea.";
    let report_b = cascade.evaluate(&engine, medical_state).unwrap();
    assert!(report_b.passed);
    assert_eq!(report_b.completed_stages, 3);
    assert!(!report_b.short_circuited);
}
