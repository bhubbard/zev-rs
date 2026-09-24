use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use crate::engine::ZevEngine;
use crate::types::{
    SystemOneRequest, SystemOneResponse, ZevRequest, ZevResponse,
    DEFAULT_MODEL, MAX_QUESTIONS, MAX_SLOTS, MODEL_ALIAS,
};

#[derive(Clone)]
pub struct ServerState {
    pub engine: Arc<ZevEngine>,
}

pub fn create_router(engine: Arc<ZevEngine>) -> Router {
    let state = ServerState { engine };

    Router::new()
        .route("/health", get(health_handler))
        .route("/", get(home_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/limits", get(limits_handler))
        .route("/v1/decisions", post(decisions_handler))
        .route("/v1/systemone", post(systemone_handler))
        .route("/v1/tev1", post(tev1_handler))
        .with_state(state)
}

async fn home_handler() -> &'static str {
    "Zev: High-performance, 100% order-invariant, calibrated zero-token LLM decision engine."
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
) -> Result<Json<ZevResponse>, (StatusCode, String)> {
    state
        .engine
        .evaluate(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

async fn systemone_handler(
    State(state): State<ServerState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, (StatusCode, String)> {
    state
        .engine
        .evaluate_system_one(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

async fn tev1_handler(
    State(state): State<ServerState>,
    Json(req): Json<crate::tev1::Tev1Request>,
) -> Result<Json<crate::tev1::Tev1Response>, (StatusCode, String)> {
    state
        .engine
        .evaluate_tev1(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

