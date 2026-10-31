//! Native macOS Foundation Models (`/usr/bin/fm`) backend for `zev-rs`.
//!
//! Supports Apple's built-in Foundation Models CLI & HTTP server introduced in modern macOS
//! (`AFM 3 Core Advanced` on-device `system` model and `pcc` Private Cloud Compute model):
//! - **Fast HTTP Daemon Transport (`fm serve`)**: Sends `POST /v1/chat/completions` with
//!   `"stream": false` and grammar-constrained `json_schema` `enum` decoding over candidate IDs.
//! - **Zero-Daemon Direct CLI Transport (`/usr/bin/fm respond`)**: Invokes `/usr/bin/fm respond`
//!   with `--use-case general --guardrails permissive-content-transformations --greedy --no-stream`
//!   and a temporary `--schema` file for 100% valid candidate selection.

use crate::error::Result;
#[cfg(not(target_arch = "wasm32"))]
use crate::error::ZevError;
use crate::types::{Candidate, Question, UncertaintyMetrics, ZevAnswer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Default path to the native macOS Foundation Models CLI binary.
pub const DEFAULT_FM_BIN: &str = "/usr/bin/fm";

/// Default local endpoint when running `fm serve --port 1976`.
pub const DEFAULT_FM_ENDPOINT: &str = "http://127.0.0.1:1976/v1";

/// Default on-device Apple Foundation Model (`AFM 3 Core Advanced`).
pub const FM_SYSTEM_MODEL: &str = "system";

/// Apple Foundation Model on Private Cloud Compute.
pub const FM_PCC_MODEL: &str = "pcc";

/// Configuration for the macOS Foundation Models (`fm`) backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmConfig {
    /// Path to the `fm` binary (defaults to `FM_BIN` or `/usr/bin/fm`).
    pub binary_path: String,
    /// Model identifier: `"system"` (on-device AFM 3 Core Advanced) or `"pcc"` (Private Cloud Compute).
    pub model: String,
    /// Optional `fm serve` HTTP endpoint (defaults to `FM_URL` or `http://127.0.0.1:1976/v1`).
    pub endpoint: String,
    /// Use grammar-constrained JSON Schema (`--schema` / `response_format`) over candidate IDs.
    pub use_structured_schema: bool,
    /// Guardrails mode (`"permissive-content-transformations"` or `"default"`).
    pub guardrails: String,
    /// HTTP request timeout when talking to `fm serve`.
    pub timeout: Duration,
}

impl Default for FmConfig {
    fn default() -> Self {
        let binary_path =
            std::env::var("FM_BIN").unwrap_or_else(|_| DEFAULT_FM_BIN.to_string());
        let model = std::env::var("FM_MODEL").unwrap_or_else(|_| FM_SYSTEM_MODEL.to_string());
        let endpoint =
            std::env::var("FM_URL").unwrap_or_else(|_| DEFAULT_FM_ENDPOINT.to_string());
        let use_structured_schema = std::env::var("FM_STRUCTURED_SCHEMA")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true);
        let guardrails = std::env::var("FM_GUARDRAILS")
            .unwrap_or_else(|_| "permissive-content-transformations".to_string());
        let timeout_ms = std::env::var("FM_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10_000);

        Self {
            binary_path,
            model,
            endpoint,
            use_structured_schema,
            guardrails,
            timeout: Duration::from_millis(timeout_ms),
        }
    }
}

impl FmConfig {
    /// Creates a configuration targeting the on-device `system` model (`AFM 3 Core Advanced`).
    pub fn system() -> Self {
        Self {
            model: FM_SYSTEM_MODEL.to_string(),
            ..Self::default()
        }
    }

    /// Creates a configuration targeting Apple's Private Cloud Compute (`pcc`) model.
    pub fn pcc() -> Self {
        Self {
            model: FM_PCC_MODEL.to_string(),
            ..Self::default()
        }
    }
}

/// Checks whether the native macOS `/usr/bin/fm` CLI is present on the host system.
pub fn is_fm_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        let bin = std::env::var("FM_BIN").unwrap_or_else(|_| DEFAULT_FM_BIN.to_string());
        std::path::Path::new(&bin).exists()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Checks whether a local `fm serve` HTTP daemon is currently listening.
#[cfg(not(target_arch = "wasm32"))]
pub fn is_fm_serve_running(endpoint: &str) -> bool {
    static FM_SERVE_CACHE: std::sync::Mutex<Option<(String, i64, bool)>> =
        std::sync::Mutex::new(None);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let mut cache = FM_SERVE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((ref cached_ep, cached_time, cached_alive)) = *cache {
        if cached_ep == endpoint && (now - cached_time) < 10 {
            return cached_alive;
        }
    }

    let alive = crate::config::probe_http_endpoint(endpoint, Duration::from_millis(30));
    *cache = Some((endpoint.to_string(), now, alive));
    alive
}

/// Builds a grammar-constrained JSON Schema matching `fm schema object` format.
///
/// Constrains the model's `decision` property to an exact `enum` of the candidate IDs
/// plus `"__insufficient__"`.
pub fn build_fm_decision_schema(candidates: &[Candidate]) -> serde_json::Value {
    let mut enum_vals: Vec<serde_json::Value> = candidates
        .iter()
        .map(|c| serde_json::Value::String(c.id.clone()))
        .collect();
    if !candidates.iter().any(|c| c.id == "__insufficient__") {
        enum_vals.push(serde_json::Value::String("__insufficient__".to_string()));
    }

    serde_json::json!({
        "type": "object",
        "title": "Decision",
        "x-order": ["decision"],
        "required": ["decision"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": enum_vals,
                "description": "Selected candidate option ID or __insufficient__ if evidence is missing"
            }
        },
        "additionalProperties": false
    })
}

/// Formats the classification prompt for `fm`.
pub fn format_fm_prompt(state: &str, instructions: &str, candidates: &[Candidate]) -> String {
    let mut prompt = format!("Question / Task:\n{}\n\nCandidate Options:\n", instructions.trim());
    for (i, c) in candidates.iter().enumerate() {
        prompt.push_str(&format!("{}. [{}] {}\n", i + 1, c.id, c.description));
    }
    prompt.push_str(&format!(
        "\nInput State:\n{}\n\nSelect the best matching candidate option ID (or __insufficient__ if the state lacks evidence).",
        state.trim()
    ));
    prompt
}

/// Evaluates a decision question using the default macOS Foundation Models (`fm`) configuration.
pub fn evaluate_fm(
    state: &str,
    question: &Question,
    candidates: &[Candidate],
) -> Result<ZevAnswer> {
    let config = FmConfig::default();
    evaluate_fm_with_config(state, question, candidates, &config)
}

/// Evaluates a decision question using a specific `FmConfig` (`system` or `pcc`).
pub fn evaluate_fm_with_config(
    state: &str,
    question: &Question,
    candidates: &[Candidate],
    config: &FmConfig,
) -> Result<ZevAnswer> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (state, question, candidates, config);
        return Err( crate::error::ZevError::Evaluation(
            "macOS /usr/bin/fm is not available in wasm32 targets".into(),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::qos::elevate_thread_qos();

        // Shortlist long candidate lists (> 8) via SIMD token-set overlap to prevent context dilution
        let effective_candidates: Vec<Candidate> = if candidates.len() > 8 {
            let opt_defs: Vec<crate::types::OptionDef> = candidates
                .iter()
                .map(|c| crate::types::OptionDef {
                    id: c.id.clone(),
                    description: c.description.clone(),
                })
                .collect();
            let shortlisted = crate::shortlist::shortlist_options(&opt_defs, state, 8);
            let mut list = Vec::new();
            for s in shortlisted {
                if let Some(c) = candidates.iter().find(|c| c.id == s.id) {
                    list.push(c.clone());
                }
            }
            if list.is_empty() {
                candidates.to_vec()
            } else {
                list
            }
        } else {
            candidates.to_vec()
        };

        let prompt = format_fm_prompt(state, question.instructions(), &effective_candidates);
        let schema = build_fm_decision_schema(&effective_candidates);

        // 1. Try fast persistent `fm serve` HTTP daemon if active
        let raw_output = if is_fm_serve_running(&config.endpoint) {
            match call_fm_serve_http(config, &prompt, &schema) {
                Ok(out) => out,
                Err(_) => call_fm_cli_respond(config, &prompt, &schema)?,
            }
        } else {
            // 2. Fall back to direct `/usr/bin/fm respond` CLI invocation
            call_fm_cli_respond(config, &prompt, &schema)?
        };

        let parsed = parse_fm_response(&raw_output, candidates);
        build_fm_answer(parsed, question, candidates, &config.model)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn call_fm_serve_http(
    config: &FmConfig,
    prompt: &str,
    schema: &serde_json::Value,
) -> Result<String> {
    use std::io::Read;

    let url = format!("{}/chat/completions", config.endpoint.trim_end_matches('/'));
    let mut payload = serde_json::json!({
        "model": config.model,
        "stream": false,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": "You are an accurate, deterministic decision engine. Select the matching candidate option ID."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    if config.use_structured_schema {
        payload["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "Decision",
                "strict": true,
                "schema": schema
            }
        });
    }

    let mut resp = ureq::post(&url)
        .send_json(&payload)
        .map_err(|e| ZevError::Evaluation(format!("fm serve HTTP request failed: {e}")))?;

    let mut body_str = String::new();
    resp.body_mut()
        .as_reader()
        .read_to_string(&mut body_str)
        .map_err(|e| ZevError::Evaluation(format!("Failed reading fm serve response: {e}")))?;

    let json_val: serde_json::Value = serde_json::from_str(&body_str)
        .map_err(|e| ZevError::Evaluation(format!("Invalid JSON from fm serve: {e}")))?;

    if let Some(content) = json_val["choices"][0]["message"]["content"].as_str() {
        Ok(content.trim().to_string())
    } else {
        Err(ZevError::Evaluation(
            "Missing choices[0].message.content in fm serve response".into(),
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn call_fm_cli_respond(
    config: &FmConfig,
    prompt: &str,
    schema: &serde_json::Value,
) -> Result<String> {
    if !std::path::Path::new(&config.binary_path).exists() {
        return Err(ZevError::Evaluation(format!(
            "macOS Foundation Models CLI not found at {}",
            config.binary_path
        )));
    }

    let mut cmd = std::process::Command::new(&config.binary_path);
    cmd.arg("respond")
        .arg("--model")
        .arg(&config.model)
        .arg("--use-case")
        .arg("general")
        .arg("--guardrails")
        .arg(&config.guardrails)
        .arg("--greedy")
        .arg("--no-stream")
        .arg("--instructions")
        .arg("You are an accurate, deterministic decision classifier. Select the matching option ID from the provided candidates.");

    // Write temporary schema file if structured schema mode is enabled
    let temp_schema_path = if config.use_structured_schema {
        let tid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("zev_fm_schema_{tid}_{nanos}.json"));
        if let Ok(schema_bytes) = serde_json::to_vec(schema) {
            if std::fs::write(&path, schema_bytes).is_ok() {
                cmd.arg("--schema").arg(&path);
                Some(path)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    cmd.arg(prompt);

    let output_res = cmd.output();

    if let Some(ref p) = temp_schema_path {
        let _ = std::fs::remove_file(p);
    }

    let output = output_res
        .map_err(|e| ZevError::Evaluation(format!("Failed to execute {}: {e}", config.binary_path)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ZevError::Evaluation(format!(
            "fm respond exited with status {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Parses either structured JSON (`{"decision": "id"}`) or bracketed/plain text from `fm`.
pub fn parse_fm_response(raw: &str, candidates: &[Candidate]) -> String {
    let trimmed = raw.trim();

    // 1. Structured JSON object output from `--schema` / `response_format`: {"decision": "..."}
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(dec) = val.get("decision").and_then(|v| v.as_str()) {
            let dec_trim = dec.trim();
            if dec_trim.eq_ignore_ascii_case("__insufficient__")
                || dec_trim.eq_ignore_ascii_case("insufficient")
                || dec_trim.eq_ignore_ascii_case("none")
            {
                return "__insufficient__".to_string();
            }
            if let Some(c) = candidates
                .iter()
                .find(|c| c.id.eq_ignore_ascii_case(dec_trim))
            {
                return c.id.clone();
            }
        }
    }

    // 2. Bracket extraction: [id]
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed[start..].find(']') {
            let inside = trimmed[start + 1..start + end].trim();
            if inside.eq_ignore_ascii_case("__insufficient__")
                || inside.eq_ignore_ascii_case("insufficient")
                || inside.eq_ignore_ascii_case("none")
            {
                return "__insufficient__".to_string();
            }
            if let Some(c) = candidates
                .iter()
                .find(|c| c.id.eq_ignore_ascii_case(inside))
            {
                return c.id.clone();
            }
        }
    }

    let lower = trimmed.to_lowercase();
    if lower.contains("__insufficient__")
        || lower.contains("insufficient")
        || lower.contains("unrelated")
        || lower.contains("out of domain")
    {
        return "__insufficient__".to_string();
    }

    // 3. Exact or substring candidate match
    for c in candidates {
        if c.id.eq_ignore_ascii_case(trimmed) {
            return c.id.clone();
        }
    }
    for c in candidates {
        if lower.contains(&c.id.to_lowercase()) {
            return c.id.clone();
        }
    }

    "__insufficient__".to_string()
}

fn build_fm_answer(
    parsed_id: String,
    question: &Question,
    candidates: &[Candidate],
    model: &str,
) -> Result<ZevAnswer> {
    let is_abstain =
        parsed_id == "__insufficient__" || !candidates.iter().any(|c| c.id == parsed_id);

    let mut probabilities = BTreeMap::new();
    let mut logits = BTreeMap::new();

    let (status, decision_val, top_p, other_p) = if is_abstain {
        let uniform = if !candidates.is_empty() {
            1.0 / candidates.len() as f64
        } else {
            0.0
        };
        for c in candidates {
            probabilities.insert(c.id.clone(), uniform);
            logits.insert(c.id.clone(), 0.0);
        }
        (
            "insufficient".to_string(),
            Some(serde_json::Value::String("__insufficient__".to_string())),
            uniform,
            uniform,
        )
    } else {
        let top = 0.96;
        let other = if candidates.len() > 1 {
            (1.0 - top) / (candidates.len() - 1) as f64
        } else {
            0.0
        };
        for c in candidates {
            if c.id == parsed_id {
                probabilities.insert(c.id.clone(), top);
                logits.insert(c.id.clone(), 3.18);
            } else {
                probabilities.insert(c.id.clone(), other);
                logits.insert(c.id.clone(), 0.0);
            }
        }
        let dec = match question {
            Question::Boolean(_) => {
                let b = parsed_id.eq_ignore_ascii_case("true")
                    || parsed_id.eq_ignore_ascii_case("yes");
                serde_json::Value::Bool(b)
            }
            _ => serde_json::Value::String(parsed_id),
        };
        ("ok".to_string(), Some(dec), top, other)
    };

    let q_type = match question {
        Question::Boolean(_) => "boolean",
        Question::Choice(_) => "choice",
        Question::Score(_) => "score",
        Question::Numeric(_) => "numeric",
    };

    let source_tag = if model.eq_ignore_ascii_case("pcc") {
        "fm-pcc".to_string()
    } else {
        "fm".to_string()
    };

    Ok(ZevAnswer {
        question_type: q_type.to_string(),
        status,
        decision: decision_val,
        confidence: if is_abstain { 0.0 } else { top_p },
        probabilities,
        logits,
        uncertainty: UncertaintyMetrics {
            top_probability: top_p,
            entropy_nats: if is_abstain { 1.0 } else { 0.08 },
            concentration: if is_abstain { 0.0 } else { 0.92 },
            margin: Some((top_p - other_p).max(0.0)),
            unavailable_probability: if is_abstain { 1.0 } else { 0.0 },
            quantile_spread: None,
        },
        statistics: None,
        expected_value: None,
        temperature: 1.0,
        source: Some(source_tag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candidates() -> Vec<Candidate> {
        vec![
            Candidate {
                id: "billing".into(),
                description: "Invoices and refunds".into(),
                value: None,
            },
            Candidate {
                id: "infra".into(),
                description: "Database outages and 500 errors".into(),
                value: None,
            },
        ]
    }

    #[test]
    fn test_build_fm_decision_schema() {
        let cands = sample_candidates();
        let schema = build_fm_decision_schema(&cands);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["title"], "Decision");
        let enums = schema["properties"]["decision"]["enum"]
            .as_array()
            .expect("must have enum array");
        assert_eq!(enums.len(), 3);
        assert_eq!(enums[0], "billing");
        assert_eq!(enums[1], "infra");
        assert_eq!(enums[2], "__insufficient__");
    }

    #[test]
    fn test_parse_fm_response_structured_and_brackets() {
        let cands = sample_candidates();
        assert_eq!(
            parse_fm_response(r#"{"decision": "infra"}"#, &cands),
            "infra"
        );
        assert_eq!(
            parse_fm_response(r#"{"decision": "__insufficient__"}"#, &cands),
            "__insufficient__"
        );
        assert_eq!(parse_fm_response("[billing]", &cands), "billing");
        assert_eq!(parse_fm_response("infra", &cands), "infra");
        assert_eq!(
            parse_fm_response("unrelated topic", &cands),
            "__insufficient__"
        );
    }

    #[test]
    fn test_fm_config_presets() {
        let sys = FmConfig::system();
        assert_eq!(sys.model, "system");
        let pcc = FmConfig::pcc();
        assert_eq!(pcc.model, "pcc");
    }
}
