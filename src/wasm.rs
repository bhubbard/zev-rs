//! WebAssembly (WASM) Edge bindings for zev-rs.
//!
//! Enables zero-token, sub-millisecond execution inside Cloudflare Workers,
//! V8 isolates, Node.js, Deno, and modern browser environments.

use crate::engine::ZevEngine;
use crate::types::ZevRequest;
use crate::wire::{SystemOneRequest, SystemOneResponse};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

static PANIC_HOOK_SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn init_panic_hook() {
    if !PANIC_HOOK_SET.swap(true, std::sync::atomic::Ordering::Relaxed) {
        std::panic::set_hook(Box::new(|info| {
            error(&format!("ZEV_WASM_PANIC: {}", info));
        }));
    }
}

/// Evaluates a Zev decision request formatted as JSON.
///
/// Returns serialized `ZevResponse` JSON or a string error.
#[wasm_bindgen]
pub fn zev_evaluate_json(request_json: &str) -> Result<String, JsValue> {
    init_panic_hook();
    let req: ZevRequest = serde_json::from_str(request_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid ZevRequest JSON: {e}")))?;

    let engine = ZevEngine::new(None);
    let resp = engine
        .evaluate(&req)
        .map_err(|e| JsValue::from_str(&format!("Zev evaluation error: {e}")))?;

    serde_json::to_string(&resp)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

/// Evaluates a state string and questions JSON map directly.
///
/// `state`: String representation of context/state.
/// `questions_json`: JSON string representing `{ "q1": { "type": "choice", ... } }`.
#[wasm_bindgen]
pub fn zev_evaluate_state_and_questions(
    state: &str,
    questions_json: &str,
) -> Result<String, JsValue> {
    let questions: std::collections::BTreeMap<String, crate::types::Question> =
        serde_json::from_str(questions_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid Questions JSON: {e}")))?;

    let req = ZevRequest {
        state: serde_json::Value::String(state.to_string()),
        questions,
        model: Some(crate::types::DEFAULT_MODEL.to_string()),
        temperature: None,
        enable_temporal_facts: true,
        images: None,
    };

    let engine = ZevEngine::new(None);
    let resp = engine
        .evaluate(&req)
        .map_err(|e| JsValue::from_str(&format!("Zev evaluation error: {e}")))?;

    serde_json::to_string(&resp)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}

/// Evaluates a SystemOne/Jev wire request format (used by Clef and Decider) in WASM.
#[wasm_bindgen]
pub fn zev_evaluate_system_one_json(request_json: &str) -> Result<String, JsValue> {
    init_panic_hook();
    let wire_req: SystemOneRequest = serde_json::from_str(request_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid SystemOneRequest JSON: {e}")))?;

    let mut questions = std::collections::BTreeMap::new();
    for (id, wq) in &wire_req.questions {
        let q = crate::wire::wire_to_question(wq)
            .map_err(|e| JsValue::from_str(&format!("Wire question conversion error: {e}")))?;
        questions.insert(id.clone(), q);
    }

    let req = ZevRequest {
        state: wire_req.state,
        questions,
        model: Some(crate::types::DEFAULT_MODEL.to_string()),
        temperature: None,
        enable_temporal_facts: true,
        images: None,
    };

    let engine = ZevEngine::new(None);
    let resp = engine
        .evaluate(&req)
        .map_err(|e| JsValue::from_str(&format!("Evaluation error: {e}")))?;

    let mut answers = std::collections::BTreeMap::new();
    for (id, ans) in resp.answers {
        if let Some(wq) = wire_req.questions.get(&id) {
            let wire_val = crate::wire::wire_value_from_zev_answer(wq, &ans)
                .map_err(|e| JsValue::from_str(&format!("Wire answer conversion error: {e}")))?;
            answers.insert(id, wire_val);
        }
    }

    let wire_resp = SystemOneResponse {
        model: "zev-wasm-edge".into(),
        answers,
        usage: crate::wire::WireUsage {
            input_tokens: 0,
            output_tokens: 0,
        },
    };

    serde_json::to_string(&wire_resp)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
}
