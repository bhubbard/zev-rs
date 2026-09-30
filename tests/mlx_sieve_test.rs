#![cfg(feature = "mlx")]

use zev::{MlxDlqClusterer, MlxProjectionHead, SemanticSieve};

#[test]
fn test_mlx_sieve_single_query() {
    let mut cpu_sieve = SemanticSieve::new(0.20, 0.40);

    let billing_vec = SemanticSieve::hash_embed(
        "invoice billing payment charge refund payment subscription",
        64,
    );
    let tech_vec =
        SemanticSieve::hash_embed("database server outage cluster connection error crash", 64);
    let sales_vec = SemanticSieve::hash_embed(
        "enterprise contract discount pricing custom quote sales",
        64,
    );

    cpu_sieve.add_candidate("billing", billing_vec);
    cpu_sieve.add_candidate("tech", tech_vec);
    cpu_sieve.add_candidate("sales", sales_vec);

    // Compile CPU sieve into Metal GPU MLX sieve
    let mlx_sieve = cpu_sieve.to_mlx().expect("Failed to compile MLX sieve");

    assert_eq!(mlx_sieve.candidate_count(), 3);
    assert_eq!(mlx_sieve.dim(), 64);

    // Test Query: Billing
    let query_billing =
        SemanticSieve::hash_embed("I have an unexpected charge on my subscription invoice", 64);

    let cpu_res = cpu_sieve
        .evaluate_vector(&query_billing)
        .expect("CPU eval failed");
    let mlx_res = mlx_sieve
        .evaluate_vector(&query_billing)
        .expect("MLX eval failed")
        .expect("No result returned");

    assert_eq!(mlx_res.candidate_id, "billing");
    assert_eq!(mlx_res.candidate_id, cpu_res.candidate_id);
    assert!(mlx_res.decisive);
    assert!(mlx_res.margin >= 0.20);
    // Score difference between CPU and Metal GPU float matmul should be < 1e-4
    assert!(
        (mlx_res.top_score - cpu_res.top_score).abs() < 1e-4,
        "Top score discrepancy: MLX {} vs CPU {}",
        mlx_res.top_score,
        cpu_res.top_score
    );
}

#[test]
fn test_mlx_sieve_batch_dispatch() {
    let mut cpu_sieve = SemanticSieve::new(0.15, 0.35);

    let billing_vec = SemanticSieve::hash_embed(
        "billing credit invoice card payment charge refund subscription",
        64,
    );
    let tech_vec = SemanticSieve::hash_embed(
        "production database server fatal panic crash 500 error outage",
        64,
    );
    let auth_vec = SemanticSieve::hash_embed(
        "login authentication password 2fa token jwt session access",
        64,
    );

    cpu_sieve.add_candidate("billing", billing_vec);
    cpu_sieve.add_candidate("tech", tech_vec);
    cpu_sieve.add_candidate("auth", auth_vec);

    let mlx_sieve = cpu_sieve.to_mlx().expect("MLX compilation failed");

    let query1 = SemanticSieve::hash_embed(
        "Customer has unexpected payment charge on subscription invoice",
        64,
    );
    let query2 =
        SemanticSieve::hash_embed("Production database server panic 500 error and crashed", 64);
    let query3 = SemanticSieve::hash_embed(
        "User forgot login password and authentication token 2fa failed",
        64,
    );

    let batch = vec![query1.clone(), query2.clone(), query3.clone()];

    // Evaluate entire batch on Metal in a single matrix-matrix multiplication
    let results = mlx_sieve
        .evaluate_batch(&batch)
        .expect("Batch evaluation failed");

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].candidate_id, "billing");
    assert_eq!(results[1].candidate_id, "tech");
    assert_eq!(results[2].candidate_id, "auth");

    // Verify batch dispatch matches individual vector evaluations
    let single1 = mlx_sieve.evaluate_vector(&query1).unwrap().unwrap();
    let single2 = mlx_sieve.evaluate_vector(&query2).unwrap().unwrap();
    let single3 = mlx_sieve.evaluate_vector(&query3).unwrap().unwrap();

    assert_eq!(results[0].candidate_id, single1.candidate_id);
    assert_eq!(results[1].candidate_id, single2.candidate_id);
    assert_eq!(results[2].candidate_id, single3.candidate_id);

    assert!((results[0].top_score - single1.top_score).abs() < 1e-4);
    assert!((results[1].top_score - single2.top_score).abs() < 1e-4);
    assert!((results[2].top_score - single3.top_score).abs() < 1e-4);

    for res in results {
        assert!(res.decisive);
        assert!(res.margin >= 0.15);
    }
}

#[test]
fn test_mlx_projection_head() {
    let head = MlxProjectionHead::new(128, 32).expect("Projection head initialization failed");
    assert_eq!(head.input_dim(), 128);
    assert_eq!(head.output_dim(), 32);

    let input1 = vec![0.5f32; 128];
    let input2 = vec![-0.25f32; 128];

    let projected = head
        .project_batch(&[input1, input2])
        .expect("Batch projection failed");

    assert_eq!(projected.len(), 2);
    assert_eq!(projected[0].len(), 32);
    assert_eq!(projected[1].len(), 32);

    // Verify L2 normalization
    for p in &projected {
        let norm_sq: f32 = p.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq.sqrt() - 1.0).abs() < 1e-5,
            "Projected vector must be unit norm"
        );
    }
}

#[test]
fn test_mlx_dlq_clustering() {
    let dim = 64;

    // Cluster 1: PostgreSQL database timeout errors
    let db_err1 = SemanticSieve::hash_embed(
        "postgresql database connection error timeout pool exhausted",
        dim,
    );
    let db_err2 = SemanticSieve::hash_embed(
        "postgresql database connection error timeout after 30s connection reset",
        dim,
    );
    let db_err3 = SemanticSieve::hash_embed(
        "postgresql database connection error timeout server terminated pool",
        dim,
    );

    // Cluster 2: JSON payload schema validation errors
    let val_err1 = SemanticSieve::hash_embed(
        "json payload schema validation error missing required field user_id",
        dim,
    );
    let val_err2 = SemanticSieve::hash_embed(
        "json payload schema validation error unexpected field type string instead of integer",
        dim,
    );

    // Cluster 3: Unrelated outlier error
    let outlier = SemanticSieve::hash_embed("external smtp mailer failed to send invoice", dim);

    let batch = vec![db_err1, db_err2, db_err3, val_err1, val_err2, outlier];

    let clusters =
        MlxDlqClusterer::cluster_failures(&batch, 0.60).expect("DLQ failure clustering failed");

    // Expect the largest cluster (database errors) first
    assert!(!clusters.is_empty());
    assert_eq!(
        clusters[0].size, 3,
        "Database cluster should contain 3 items"
    );
    assert!(
        clusters[0].item_indices.contains(&0)
            && clusters[0].item_indices.contains(&1)
            && clusters[0].item_indices.contains(&2)
    );

    // Second cluster should be validation errors
    assert_eq!(
        clusters[1].size, 2,
        "Validation cluster should contain 2 items"
    );
    assert!(clusters[1].item_indices.contains(&3) && clusters[1].item_indices.contains(&4));
}

#[test]
fn test_mlx_gemma_triage_classifier() {
    use zev::{
        Candidate, ChoiceQuestion, MlxTriageClassifier, OptionDef, Question, SlmModelFamily,
    };

    let classifier = MlxTriageClassifier::new(SlmModelFamily::Gemma, 64, 32)
        .expect("Failed to initialize Gemma triage classifier");

    let question = Question::Choice(ChoiceQuestion {
        instructions: "Categorize the user inquiry".to_string(),
        options: vec![
            OptionDef {
                id: "billing".to_string(),
                description: "Customer invoice or payment issue".to_string(),
            },
            OptionDef {
                id: "tech".to_string(),
                description: "Database server outage or software crash".to_string(),
            },
        ],
        policy: Default::default(),
    });

    let candidates = vec![
        Candidate {
            id: "billing".to_string(),
            description: "Customer invoice or payment issue".to_string(),
            value: None,
        },
        Candidate {
            id: "tech".to_string(),
            description: "Database server outage or software crash".to_string(),
            value: None,
        },
    ];

    let state = "Customer received unexpected bill for monthly cloud subscription";
    let answer = classifier
        .evaluate_candidates(state, &question, &candidates)
        .expect("Gemma evaluation failed");

    assert_eq!(answer.source, Some("mlx-gemma".to_string()));
    assert_eq!(answer.decision, Some(serde_json::json!("billing")));
    assert!(answer.confidence > 0.5);
    assert!(answer.temperature > 1.5);
}

#[test]
fn test_mlx_qwen_triage_classifier() {
    use zev::{
        Candidate, ChoiceQuestion, MlxTriageClassifier, OptionDef, Question, SlmModelFamily,
    };

    let classifier = MlxTriageClassifier::new(SlmModelFamily::Qwen, 64, 32)
        .expect("Failed to initialize Qwen triage classifier");

    let question = Question::Choice(ChoiceQuestion {
        instructions: "Triage incident severity".to_string(),
        options: vec![
            OptionDef {
                id: "sev1".to_string(),
                description: "Critical production database server down with catastrophic data loss"
                    .to_string(),
            },
            OptionDef {
                id: "sev3".to_string(),
                description: "Minor cosmetic misalignment in settings navigation bar".to_string(),
            },
        ],
        policy: Default::default(),
    });

    let candidates = vec![
        Candidate {
            id: "sev1".to_string(),
            description: "Critical production database server down with catastrophic data loss"
                .to_string(),
            value: None,
        },
        Candidate {
            id: "sev3".to_string(),
            description: "Minor cosmetic misalignment in settings navigation bar".to_string(),
            value: None,
        },
    ];

    let state = "Production database server is down and customer records were lost in outage";
    let answer = classifier
        .evaluate_candidates(state, &question, &candidates)
        .expect("Qwen evaluation failed");

    assert_eq!(answer.source, Some("mlx-qwen".to_string()));
    assert_eq!(answer.decision, Some(serde_json::json!("sev1")));
    assert!(answer.confidence > 0.5);
}

#[test]
fn test_engine_speculative_fallback_mlx() {
    use zev::{ChoiceQuestion, OptionDef, Question, ZevEngine, ZevRequest};

    std::env::set_var("ZEV_FALLBACK", "mlx");
    std::env::set_var("ZEV_FALLBACK_CONFIDENCE", "0.999"); // Force fallback to trigger
    std::env::set_var("MLX_MODEL_FAMILY", "gemma");

    let engine = ZevEngine::default();
    let mut questions = std::collections::BTreeMap::new();
    questions.insert(
        "route".to_string(),
        Question::Choice(ChoiceQuestion {
            instructions: "Route this request".to_string(),
            options: vec![
                OptionDef {
                    id: "billing".to_string(),
                    description: "Refunds and payment invoices".to_string(),
                },
                OptionDef {
                    id: "tech".to_string(),
                    description: "Server errors and bug reports".to_string(),
                },
            ],
            policy: Default::default(),
        }),
    );

    let req = ZevRequest {
        state: serde_json::json!(
            "Client requested immediate wire refund for incorrect subscription fee"
        ),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
        images: None,
    };

    let resp = engine.evaluate(&req).expect("Engine evaluate failed");
    assert_eq!(resp.answers.len(), 1);
    let ans = resp
        .answers
        .get("route")
        .expect("Answer for 'route' missing");
    assert_eq!(ans.source, Some("mlx-gemma".to_string()));
    assert_eq!(ans.decision, Some(serde_json::json!("billing")));
}

#[tokio::test]
async fn test_server_dlq_triage_endpoint() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use zev::{create_router, DlqTriageRequest, DlqTriageResponse, ZevEngine};

    let router = create_router(Arc::new(ZevEngine::default()));

    let dlq_req = DlqTriageRequest {
        messages: vec![
            "postgres connection timed out after 30000ms".to_string(),
            "postgres connection error connection pool exhausted".to_string(),
            "schema error json payload missing user_id field".to_string(),
            "schema validation error invalid string for age".to_string(),
            "unrelated email delivery failure smtp timeout".to_string(),
        ],
        similarity_threshold: 0.50,
        max_clusters: Some(5),
    };

    let body_bytes = serde_json::to_vec(&dlq_req).unwrap();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/dlq/triage")
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let triage_resp: DlqTriageResponse = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(triage_resp.total_messages, 5);
    assert!(triage_resp.cluster_count >= 2);
    assert_eq!(triage_resp.execution_device, "metal-gpu");
}

#[test]
fn test_mlx_sieve_edge_cases() {
    // 1. Empty candidates should return error
    let empty_res = zev::MlxSemanticSieve::from_candidates(&[], 0.20, 0.40);
    assert!(empty_res.is_err());

    // 2. Single candidate
    let single_cand = vec![("only_one".to_string(), vec![1.0f32, 0.0f32])];
    let sieve = zev::MlxSemanticSieve::from_candidates(&single_cand, 0.20, 0.40).unwrap();
    assert_eq!(sieve.candidate_count(), 1);
    assert_eq!(sieve.dim(), 2);

    let res = sieve
        .evaluate_vector(&[1.0f32, 0.0f32])
        .unwrap()
        .expect("Result should be present");
    assert_eq!(res.candidate_id, "only_one");
    assert_eq!(res.margin, res.top_score);
    assert!(res.decisive);

    // 3. Batch evaluation with empty batch
    let batch_res = sieve.evaluate_batch(&[]).unwrap();
    assert!(batch_res.is_empty());
}

#[test]
fn test_mlx_dlq_clusterer_empty_and_single() {
    // 1. Empty messages
    let summary_empty = MlxDlqClusterer::cluster_failures(&[], 0.50).unwrap();
    assert_eq!(summary_empty.len(), 0);

    // 2. Single message vector
    let single_msg = vec![vec![1.0f32, 0.0f32, 0.0f32]];
    let summary_single = MlxDlqClusterer::cluster_failures(&single_msg, 0.50).unwrap();
    assert_eq!(summary_single.len(), 1);
    assert_eq!(summary_single[0].size, 1);
    assert_eq!(summary_single[0].item_indices, vec![0]);

    // 3. Dimension mismatch error
    let mismatched = vec![vec![1.0f32, 0.0f32], vec![1.0f32, 0.0f32, 0.0f32]];
    assert!(MlxDlqClusterer::cluster_failures(&mismatched, 0.50).is_err());
}

#[test]
fn test_mlx_projection_head_edge_cases() {
    let head = MlxProjectionHead::new(64, 16).unwrap();

    // 1. Empty batch
    let empty_proj = head.project_batch(&[]).unwrap();
    assert!(empty_proj.is_empty());

    // 2. Dimension mismatch error
    let bad_dim_vec = vec![0.5f32; 10]; // 10 instead of 64
    let err_proj = head.project_batch(&[bad_dim_vec]);
    assert!(err_proj.is_err());
}

#[test]
fn test_mlx_triage_classifier_validation_and_temperature() {
    use zev::{Candidate, ChoiceQuestion, OptionDef, Question, SlmModelFamily};

    // 1. Family debug formatting
    assert_eq!(format!("{:?}", SlmModelFamily::Gemma), "Gemma");
    assert_eq!(format!("{:?}", SlmModelFamily::Qwen), "Qwen");

    let classifier = zev::MlxTriageClassifier::new(SlmModelFamily::Gemma, 64, 32).unwrap();

    let question = Question::Choice(ChoiceQuestion {
        instructions: "Categorize query".into(),
        options: vec![
            OptionDef {
                id: "c1".into(),
                description: "Invoice issue".into(),
            },
            OptionDef {
                id: "c2".into(),
                description: "Hardware crash".into(),
            },
        ],
        policy: Default::default(),
    });

    let candidates = vec![
        Candidate {
            id: "c1".into(),
            description: "Invoice issue".into(),
            value: None,
        },
        Candidate {
            id: "c2".into(),
            description: "Hardware crash".into(),
            value: None,
        },
    ];

    // 2. Empty candidates error
    let valid_state = "Customer has billing complaint";
    let empty_eval = classifier.evaluate_candidates(valid_state, &question, &[]);
    assert!(empty_eval.is_err());

    // 3. Successful evaluation
    let answer = classifier
        .evaluate_candidates(valid_state, &question, &candidates)
        .expect("Evaluation must succeed");
    assert_eq!(answer.question_type, "choice");
    assert!(answer.probabilities.contains_key("c1"));
}
