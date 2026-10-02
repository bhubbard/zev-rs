use crate::decision_cache::{hash_decision_request, DecisionCache};
use crate::engine::ZevEngine;
use crate::types::{
    DlqClusterReport, DlqTriageRequest, DlqTriageResponse, SystemOneRequest, SystemOneResponse,
    ZevRequest, ZevResponse, DEFAULT_MODEL, MAX_QUESTIONS, MAX_SLOTS, MODEL_ALIAS,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;

const INDEX_HTML: &str = include_str!("../assets/index.html");

#[derive(Clone, Debug)]
pub struct ServerState {
    pub engine: Arc<ZevEngine>,
    pub cache: Arc<DecisionCache<u64, serde_json::Value>>,
}

impl ServerState {
    pub fn new(engine: Arc<ZevEngine>) -> Self {
        Self {
            engine,
            cache: Arc::new(DecisionCache::new(
                10_000,
                Some(std::time::Duration::from_secs(300)),
            )),
        }
    }
}

pub fn create_router(engine: Arc<ZevEngine>) -> Router {
    let state = ServerState::new(engine);

    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/", get(home_handler))
        .route("/models", get(models_handler))
        .route("/v1/models", get(models_handler))
        .route("/limits", get(limits_handler))
        .route("/v1/limits", get(limits_handler))
        .route("/decisions", post(decisions_handler))
        .route("/v1/decisions", post(decisions_handler))
        .route("/systemone", post(systemone_handler))
        .route("/v1/systemone", post(systemone_handler))
        .route("/evaluate", post(systemone_handler))
        .route("/v1/evaluate", post(systemone_handler))
        .route("/tev1", post(tev1_handler))
        .route("/v1/tev1", post(tev1_handler))
        .route("/dlq/triage", post(dlq_triage_handler))
        .route("/v1/dlq/triage", post(dlq_triage_handler))
        .with_state(state)
}

async fn home_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "ready": true,
        "engine": "zev",
        "model": DEFAULT_MODEL,
        "order_invariant": true,
        "calibrated": true,
    }))
}

async fn ready_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "ready": true,
        "engine": "zev",
        "checkpoints_resident": true,
    }))
}

async fn models_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [
            {
                "name": MODEL_ALIAS,
                "description": "Zev Apex: 100% order-invariant, calibrated zero-token decision engine",
                "release_date": "2026-09-24"
            },
            {
                "name": DEFAULT_MODEL,
                "description": "Zev Apex v1 release",
                "release_date": "2026-09-24"
            },
            {
                "name": "jev-latest",
                "description": "TypeSafe compatibility endpoint answered by Zev",
                "release_date": "2026-09-24"
            },
            {
                "name": "zev-gemma",
                "description": "Zev with local distilled Gemma 4 speculative fallback",
                "release_date": "2026-10-01"
            },
            {
                "name": "zev-apfel",
                "description": "Zev with Apple Intelligence ANE speculative hybrid",
                "release_date": "2026-10-01"
            },
            {
                "name": "zev-clm",
                "description": "Zev with Contrastive Language Model speculative fallback",
                "release_date": "2026-10-01"
            },
            {
                "name": "zev-cascade",
                "description": "Zev three-tier speculative cascade (SIMD -> ANE -> Gemma)",
                "release_date": "2026-10-01"
            },
            {
                "name": "zev-poe",
                "description": "Zev dual speculative ensemble with Bayesian Product of Experts",
                "release_date": "2026-10-01"
            }
        ]
    }))
}

async fn limits_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "max_answers_per_question": MAX_SLOTS,
        "max_questions": MAX_QUESTIONS,
        "order_invariance_guarantee": "0.0% permutation flip rate",
        "default_calibrated_temperature": 2.179078721266035,
        "output_tokens_generated": 0,
    }))
}

async fn decisions_handler(
    State(state): State<ServerState>,
    Json(req): Json<ZevRequest>,
) -> Result<([(axum::http::HeaderName, String); 3], Json<ZevResponse>), (StatusCode, String)> {
    let start = std::time::Instant::now();
    let cache_key = hash_decision_request(&req).ok();

    if let Some(key) = cache_key {
        if let Some(cached_val) = state.cache.get(&key) {
            if let Ok(resp) = serde_json::from_value::<ZevResponse>(cached_val) {
                let eval_ms = start.elapsed().as_secs_f64() * 1000.0;
                let headers = [
                    (
                        axum::http::HeaderName::from_static("server-timing"),
                        format!("eval;dur={eval_ms:.3}"),
                    ),
                    (
                        axum::http::HeaderName::from_static("x-inference-time-ms"),
                        format!("{eval_ms:.3}"),
                    ),
                    (
                        axum::http::HeaderName::from_static("x-cache"),
                        "HIT".to_string(),
                    ),
                ];
                return Ok((headers, Json(resp)));
            }
        }
    }

    let resp = state
        .engine
        .evaluate(&req)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let eval_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Some(key) = cache_key {
        if let Ok(val) = serde_json::to_value(&resp) {
            state.cache.insert(key, val);
        }
    }

    let headers = [
        (
            axum::http::HeaderName::from_static("server-timing"),
            format!("eval;dur={eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-inference-time-ms"),
            format!("{eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-cache"),
            "MISS".to_string(),
        ),
    ];
    Ok((headers, Json(resp)))
}

async fn systemone_handler(
    State(state): State<ServerState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<
    (
        [(axum::http::HeaderName, String); 3],
        Json<SystemOneResponse>,
    ),
    (StatusCode, String),
> {
    let start = std::time::Instant::now();
    let cache_key = hash_decision_request(&req).ok();

    if let Some(key) = cache_key {
        if let Some(cached_val) = state.cache.get(&key) {
            if let Ok(resp) = serde_json::from_value::<SystemOneResponse>(cached_val) {
                let eval_ms = start.elapsed().as_secs_f64() * 1000.0;
                let headers = [
                    (
                        axum::http::HeaderName::from_static("server-timing"),
                        format!("eval;dur={eval_ms:.3}"),
                    ),
                    (
                        axum::http::HeaderName::from_static("x-inference-time-ms"),
                        format!("{eval_ms:.3}"),
                    ),
                    (
                        axum::http::HeaderName::from_static("x-cache"),
                        "HIT".to_string(),
                    ),
                ];
                return Ok((headers, Json(resp)));
            }
        }
    }

    let resp = state
        .engine
        .evaluate_system_one(&req)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let eval_ms = start.elapsed().as_secs_f64() * 1000.0;

    if let Some(key) = cache_key {
        if let Ok(val) = serde_json::to_value(&resp) {
            state.cache.insert(key, val);
        }
    }

    let headers = [
        (
            axum::http::HeaderName::from_static("server-timing"),
            format!("eval;dur={eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-inference-time-ms"),
            format!("{eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-cache"),
            "MISS".to_string(),
        ),
    ];
    Ok((headers, Json(resp)))
}

async fn tev1_handler(
    State(state): State<ServerState>,
    Json(req): Json<crate::tev1::Tev1Request>,
) -> Result<
    (
        [(axum::http::HeaderName, String); 2],
        Json<crate::tev1::Tev1Response>,
    ),
    (StatusCode, String),
> {
    let start = std::time::Instant::now();
    let resp = state
        .engine
        .evaluate_tev1(&req)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let eval_ms = start.elapsed().as_secs_f64() * 1000.0;
    let headers = [
        (
            axum::http::HeaderName::from_static("server-timing"),
            format!("eval;dur={eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-inference-time-ms"),
            format!("{eval_ms:.3}"),
        ),
    ];
    Ok((headers, Json(resp)))
}

async fn dlq_triage_handler(
    Json(req): Json<DlqTriageRequest>,
) -> Result<
    (
        [(axum::http::HeaderName, String); 2],
        Json<DlqTriageResponse>,
    ),
    (StatusCode, String),
> {
    let start = std::time::Instant::now();

    if req.messages.is_empty() {
        let headers = [
            (
                axum::http::HeaderName::from_static("server-timing"),
                "eval;dur=0.000".to_string(),
            ),
            (
                axum::http::HeaderName::from_static("x-inference-time-ms"),
                "0.000".to_string(),
            ),
        ];
        return Ok((
            headers,
            Json(DlqTriageResponse {
                total_messages: 0,
                cluster_count: 0,
                clusters: Vec::new(),
                execution_device: "none".to_string(),
            }),
        ));
    }

    let b = req.messages.len();
    let dim = 64;
    let embeddings: Vec<Vec<f32>> = req
        .messages
        .iter()
        .map(|msg| crate::semantic_sieve::SemanticSieve::hash_embed(msg, dim))
        .collect();

    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    let (clusters_raw, device) = {
        let clusters =
            crate::mlx::MlxDlqClusterer::cluster_failures(&embeddings, req.similarity_threshold)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mapped = clusters
            .into_iter()
            .map(|c| (c.representative_index, c.item_indices))
            .collect::<Vec<_>>();
        (mapped, "metal-gpu")
    };

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
    let (clusters_raw, device) = {
        let clusters = crate::semantic_sieve::SemanticSieve::cluster_vectors_cpu(
            &embeddings,
            req.similarity_threshold,
        );
        (clusters, "simd-cpu")
    };

    let mut clusters = Vec::new();
    for (cluster_id, (rep_idx, members)) in clusters_raw.into_iter().enumerate() {
        if let Some(max_c) = req.max_clusters {
            if cluster_id >= max_c {
                break;
            }
        }
        let size = members.len();
        let pct = (size as f64 / b as f64) * 100.0;
        let rep_msg = req.messages[rep_idx].clone();
        clusters.push(DlqClusterReport {
            cluster_id,
            size,
            percentage: pct,
            representative_message: rep_msg,
            message_indices: members,
        });
    }

    let eval_ms = start.elapsed().as_secs_f64() * 1000.0;
    let headers = [
        (
            axum::http::HeaderName::from_static("server-timing"),
            format!("eval;dur={eval_ms:.3}"),
        ),
        (
            axum::http::HeaderName::from_static("x-inference-time-ms"),
            format!("{eval_ms:.3}"),
        ),
    ];

    Ok((
        headers,
        Json(DlqTriageResponse {
            total_messages: b,
            cluster_count: clusters.len(),
            clusters,
            execution_device: device.to_string(),
        }),
    ))
}
