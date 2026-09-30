// ============================================================================
// tests/gemma4_test.rs — Comprehensive Test Suite for Gemma 4 in zev-rs
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use zev::gemma::{evaluate_gemma, format_gemma_prompt, GemmaConfig};
use zev::types::{Candidate, ChoiceQuestion, OptionDef, Question};

static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_gemma4_live_mock_http_roundtrip() {
    let _lock = ENV_MUTEX.lock().unwrap();
    use axum::extract::Json;
    use axum::routing::post;
    use axum::Router;
    use serde_json::Value;

    let request_count = Arc::new(AtomicUsize::new(0));
    let req_counter = request_count.clone();

    // Mock OpenAI/vLLM/Ollama Gemma-4-31B chat completions handler
    let app = Router::new().route(
        "/chat/completions",
        post(move |Json(payload): Json<Value>| {
            let req_counter = req_counter.clone();
            async move {
                req_counter.fetch_add(1, Ordering::SeqCst);

                // 1. Verify model is Gemma-4-31B
                assert_eq!(
                    payload.get("model").and_then(|m| m.as_str()),
                    Some("gemma-4-31b"),
                    "Expected Gemma-4-31B model identifier"
                );

                // 2. Verify messages structure and Gemma turn format
                let messages = payload
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .expect("messages array");
                assert_eq!(messages.len(), 2);
                assert_eq!(messages[0]["role"], "system");
                assert_eq!(messages[1]["role"], "user");

                let user_content = messages[1]["content"].as_str().expect("user prompt");
                assert!(user_content.contains("<start_of_turn>user"));
                assert!(user_content.contains("<start_of_turn>model"));
                assert!(user_content.contains("[security]"));

                // 3. Return Gemma 4 decision: [security]
                Json(serde_json::json!({
                    "id": "chatcmpl-gemma4-9981",
                    "object": "chat.completion",
                    "created": 1727710000,
                    "model": "gemma-4-31b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Decision: [security]"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 64,
                        "completion_tokens": 4,
                        "total_tokens": 68
                    }
                }))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Point GemmaConfig to mock Gemma 4 server
    std::env::set_var("GEMMA_URL", format!("http://{}", addr));
    std::env::set_var("GEMMA_MODEL", "gemma-4-31b");
    std::env::set_var("GEMMA_TIMEOUT_MS", "2000");
    std::env::remove_var("GEMMA_MODE");

    let question = Question::Choice(ChoiceQuestion {
        instructions: "Route security incident report".into(),
        options: vec![
            OptionDef {
                id: "billing".into(),
                description: "Invoice and card charge issues".into(),
            },
            OptionDef {
                id: "security".into(),
                description: "Unauthorized account login and credential compromise".into(),
            },
            OptionDef {
                id: "sales".into(),
                description: "Commercial pricing and contract renewals".into(),
            },
        ],
        policy: Default::default(),
    });

    let candidates = vec![
        Candidate {
            id: "billing".into(),
            description: "Invoice and card charge issues".into(),
            value: None,
        },
        Candidate {
            id: "security".into(),
            description: "Unauthorized account login and credential compromise".into(),
            value: None,
        },
        Candidate {
            id: "sales".into(),
            description: "Commercial pricing and contract renewals".into(),
            value: None,
        },
    ];

    let state =
        "Suspicious login detected from unauthorized foreign IP address trying password spray.";
    let answer =
        evaluate_gemma(state, &question, &candidates).expect("Gemma 4 evaluation should succeed");

    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        answer.decision,
        Some(serde_json::Value::String("security".into()))
    );
    assert_eq!(answer.source, Some("gemma".to_string()));
    assert_eq!(answer.confidence, 0.88);
    assert!(answer.probabilities.contains_key("security"));
    assert_eq!(answer.probabilities["security"], 0.88);

    // Clean up env
    std::env::remove_var("GEMMA_URL");
    std::env::remove_var("GEMMA_MODEL");
    std::env::remove_var("GEMMA_TIMEOUT_MS");
}

#[test]
fn test_gemma4_prompt_token_formatting() {
    let candidates = vec![
        Candidate {
            id: "alpha".into(),
            description: "First option".into(),
            value: None,
        },
        Candidate {
            id: "beta".into(),
            description: "Second option".into(),
            value: None,
        },
    ];

    let prompt = format_gemma_prompt("Contextual state", "Select option", &candidates);

    assert!(prompt.starts_with("<start_of_turn>user\n"));
    assert!(prompt.contains("Instructions:\nSelect option\n"));
    assert!(prompt.contains("1. [alpha] First option\n"));
    assert!(prompt.contains("2. [beta] Second option\n"));
    assert!(prompt.contains("Context:\nContextual state\n"));
    assert!(prompt
        .ends_with("Decision (respond strictly with [id]):<end_of_turn>\n<start_of_turn>model\n"));
}

#[test]
fn test_gemma4_config_defaults() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::remove_var("GEMMA_URL");
    std::env::remove_var("GEMMA_MODEL");
    std::env::remove_var("GEMMA_TIMEOUT_MS");

    let cfg = GemmaConfig::default();
    assert_eq!(cfg.endpoint, "http://127.0.0.1:8000/v1");
    assert_eq!(cfg.model, "gemma-4-31b");
    assert_eq!(cfg.timeout, std::time::Duration::from_millis(2000));
    assert!(cfg.fallback_to_heuristic);
}

#[test]
fn test_gemma4_distillation_and_shortlisting() {
    let _lock = ENV_MUTEX.lock().unwrap();
    std::env::set_var("GEMMA_MODE", "distill");

    let mut options = Vec::new();
    let mut candidates = Vec::new();
    for i in 0..12 {
        let id = format!("queue_{i}");
        let desc = format!("Service queue category number {i}");
        options.push(OptionDef {
            id: id.clone(),
            description: desc.clone(),
        });
        candidates.push(Candidate {
            id,
            description: desc,
            value: None,
        });
    }

    let q = Question::Choice(ChoiceQuestion {
        instructions: "Route to appropriate queue".into(),
        options,
        policy: Default::default(),
    });

    let ans = evaluate_gemma(
        "Work item intended for service queue category number 4",
        &q,
        &candidates,
    )
    .expect("Distilled Gemma evaluation should succeed");

    assert_eq!(ans.source, Some("gemma-distill".to_string()));
    assert_eq!(ans.confidence, 0.82);
    assert_eq!(
        ans.probabilities.len(),
        12,
        "Should preserve probability distribution for all 12 candidates"
    );

    std::env::remove_var("GEMMA_MODE");
}

#[test]
fn test_gemma4_unreachable_endpoint_graceful_fallback() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // Port 1 is unassigned / refused
    std::env::set_var("GEMMA_URL", "http://127.0.0.1:1/v1");
    std::env::set_var("GEMMA_TIMEOUT_MS", "50");
    std::env::remove_var("GEMMA_MODE");

    let q = Question::Choice(ChoiceQuestion {
        instructions: "Categorize query".into(),
        options: vec![
            OptionDef {
                id: "yes".into(),
                description: "Affirmative".into(),
            },
            OptionDef {
                id: "no".into(),
                description: "Negative".into(),
            },
        ],
        policy: Default::default(),
    });

    let candidates = vec![
        Candidate {
            id: "yes".into(),
            description: "Affirmative".into(),
            value: None,
        },
        Candidate {
            id: "no".into(),
            description: "Negative".into(),
            value: None,
        },
    ];

    // Must not panic or return Err; must fallback gracefully to distilled knowledge
    let ans = evaluate_gemma("Affirmative confirmation required", &q, &candidates);
    assert!(
        ans.is_ok(),
        "Unreachable Gemma endpoint must trigger graceful fallback"
    );
    let unwrapped = ans.unwrap();
    assert_eq!(unwrapped.source, Some("gemma-distill".to_string()));

    std::env::remove_var("GEMMA_URL");
    std::env::remove_var("GEMMA_TIMEOUT_MS");
}
