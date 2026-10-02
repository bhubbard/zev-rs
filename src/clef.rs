//! Cloudflare Clef Decision Model provider for zev-rs.
//!
//! Connects to Cloudflare Workers AI hosting Clef (`@cf/typesafe/clef` and `@cf/typesafe/clef-flash`),
//! the specialized non-autoregressive decision models trained with 2-stage attention,
//! Brier loss calibration, and RLCD partial credit.
//!
//! Clef uses the Jev / SystemOne wire specification for input states and questions,
//! returning probability distributions across structured candidate slots.

use crate::error::{Result, ZevError};
use crate::types::{Question, UncertaintyMetrics, ZevAnswer};
use crate::wire::{question_to_wire, SystemOneRequest, SystemOneResponse, WireAnswer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_CLEF_MODEL: &str = "@cf/typesafe/clef";
pub const CLEF_FLASH_MODEL: &str = "@cf/typesafe/clef-flash";
pub const CLOUDFLARE_AI_BASE_URL: &str = "https://api.cloudflare.com/client/v4/accounts";

/// Configuration for connecting to Cloudflare Workers AI Clef endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudflareClefConfig {
    pub account_id: String,
    pub api_token: String,
    pub model: String,
    pub endpoint_override: Option<String>,
    pub timeout_ms: u64,
}

impl Default for CloudflareClefConfig {
    fn default() -> Self {
        Self {
            account_id: String::new(),
            api_token: String::new(),
            model: DEFAULT_CLEF_MODEL.to_string(),
            endpoint_override: None,
            timeout_ms: 5000,
        }
    }
}

/// Standard Cloudflare Workers AI response envelope:
/// `{ "result": { ... }, "success": true, "errors": [], "messages": [] }`
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CloudflareAiEnvelope {
    pub result: Option<SystemOneResponse>,
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub errors: Vec<CloudflareApiError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CloudflareApiError {
    #[serde(default)]
    pub code: Option<i64>,
    #[serde(default)]
    pub message: Option<String>,
}

/// Cloudflare Workers AI Clef Provider.
#[derive(Debug, Clone)]
pub struct CloudflareClefProvider {
    config: CloudflareClefConfig,
}

impl CloudflareClefProvider {
    /// Creates a new provider with the specified account ID and API token.
    pub fn new(account_id: impl Into<String>, api_token: impl Into<String>) -> Self {
        Self {
            config: CloudflareClefConfig {
                account_id: account_id.into(),
                api_token: api_token.into(),
                model: DEFAULT_CLEF_MODEL.to_string(),
                endpoint_override: None,
                timeout_ms: 5000,
            },
        }
    }

    /// Sets the target Clef model identifier (e.g. `@cf/typesafe/clef` or `@cf/typesafe/clef-flash`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    /// Configures the provider to use the lightweight, low-latency Clef Flash model.
    pub fn with_flash(mut self) -> Self {
        self.config.model = CLEF_FLASH_MODEL.to_string();
        self
    }

    /// Overrides the endpoint URL (useful for self-hosted proxies, Cloudflare Workers routes, or mock testing).
    pub fn with_endpoint_override(mut self, url: impl Into<String>) -> Self {
        self.config.endpoint_override = Some(url.into());
        self
    }

    /// Sets request timeout in milliseconds.
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = timeout_ms;
        self
    }

    /// Loads provider configuration from environment variables:
    /// - `CLOUDFLARE_ACCOUNT_ID` or `CF_ACCOUNT_ID`
    /// - `CLOUDFLARE_API_TOKEN` or `CF_API_TOKEN`
    /// - `CLOUDFLARE_CLEF_MODEL` (optional, defaults to `@cf/typesafe/clef`)
    /// - `CLOUDFLARE_CLEF_ENDPOINT` (optional URL override)
    /// - `CLOUDFLARE_CLEF_TIMEOUT_MS` (optional timeout in ms)
    pub fn from_env() -> Option<Self> {
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .or_else(|_| std::env::var("CF_ACCOUNT_ID"))
            .ok()?;
        let api_token = std::env::var("CLOUDFLARE_API_TOKEN")
            .or_else(|_| std::env::var("CF_API_TOKEN"))
            .ok()?;

        let model = std::env::var("CLOUDFLARE_CLEF_MODEL")
            .unwrap_or_else(|_| DEFAULT_CLEF_MODEL.to_string());
        let endpoint_override = std::env::var("CLOUDFLARE_CLEF_ENDPOINT").ok();
        let timeout_ms = std::env::var("CLOUDFLARE_CLEF_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5000);

        Some(Self {
            config: CloudflareClefConfig {
                account_id,
                api_token,
                model,
                endpoint_override,
                timeout_ms,
            },
        })
    }

    /// Returns the target URL for Cloudflare Workers AI execution.
    pub fn endpoint_url(&self) -> String {
        if let Some(ref override_url) = self.config.endpoint_override {
            return override_url.clone();
        }
        format!(
            "{}/{}/ai/run/{}",
            CLOUDFLARE_AI_BASE_URL, self.config.account_id, self.config.model
        )
    }

    /// Returns the active model identifier.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Parses a raw response body into a `SystemOneResponse`, supporting both
    /// the Cloudflare API envelope (`{ "result": { ... }, "success": true }`)
    /// and direct `SystemOneResponse` payloads.
    pub fn parse_response_body(body: &str) -> Result<SystemOneResponse> {
        // Try Cloudflare envelope format first
        if let Ok(envelope) = serde_json::from_str::<CloudflareAiEnvelope>(body) {
            if let Some(res) = envelope.result {
                return Ok(res);
            }
            if !envelope.success && !envelope.errors.is_empty() {
                let err_msg = envelope
                    .errors
                    .into_iter()
                    .filter_map(|e| e.message)
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(ZevError::Internal(format!(
                    "Cloudflare Workers AI error: {err_msg}"
                )));
            }
        }

        // Try direct deserialization
        serde_json::from_str::<SystemOneResponse>(body).map_err(|e| {
            ZevError::Internal(format!(
                "Failed to deserialize SystemOneResponse from Cloudflare Clef: {e} (body: {body})"
            ))
        })
    }

    /// Sends a `SystemOneRequest` to Cloudflare Workers AI and returns the evaluated `SystemOneResponse`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn evaluate_system_one(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let url = self.endpoint_url();

        let mut req = ureq::post(&url);

        if !self.config.api_token.is_empty() {
            req = req.header(
                "Authorization",
                &format!("Bearer {}", self.config.api_token),
            );
        }

        let mut resp = req.send_json(request).map_err(|e| {
            ZevError::Internal(format!("HTTP request to Cloudflare Clef failed: {e}"))
        })?;

        use std::io::Read;
        let mut body = String::new();
        resp.body_mut()
            .as_reader()
            .read_to_string(&mut body)
            .map_err(|e| ZevError::Internal(format!("Failed reading response body: {e}")))?;

        Self::parse_response_body(&body)
    }

    /// Sends a `SystemOneRequest` to Cloudflare Workers AI and returns the evaluated `SystemOneResponse`.
    #[cfg(target_arch = "wasm32")]
    pub fn evaluate_system_one(&self, _request: &SystemOneRequest) -> Result<SystemOneResponse> {
        Err(ZevError::Internal(
            "HTTP outbound requests via ureq are not supported in wasm32 isolate. Run local Zev engine inside isolate.".into(),
        ))
    }

    /// Evaluates a single question against state using Cloudflare Clef, returning a standard `ZevAnswer`.
    pub fn evaluate(
        &self,
        state: &str,
        question: &Question,
        question_id: &str,
    ) -> Result<ZevAnswer> {
        let wire_q = question_to_wire(question)?;
        let mut questions = BTreeMap::new();
        questions.insert(question_id.to_string(), wire_q.clone());

        let req = SystemOneRequest {
            state: serde_json::Value::String(state.to_string()),
            model: self.config.model.clone(),
            questions,
        };

        let response = self.evaluate_system_one(&req)?;

        let raw_answer = response.answers.get(question_id).ok_or_else(|| {
            ZevError::DecodingError(format!(
                "Response from Clef missing answer for question ID '{question_id}'"
            ))
        })?;

        let wire_ans: WireAnswer = serde_json::from_value(raw_answer.clone()).map_err(|e| {
            ZevError::DecodingError(format!(
                "Failed parsing WireAnswer for '{question_id}': {e} (raw: {raw_answer})"
            ))
        })?;

        // Convert WireAnswer back into ZevAnswer
        clef_wire_answer_to_zev(question, &wire_ans, &self.config.model)
    }
}

/// Converts a `WireAnswer` received from Clef into a strongly-typed `ZevAnswer`.
pub fn clef_wire_answer_to_zev(
    question: &Question,
    wire_ans: &WireAnswer,
    model_name: &str,
) -> Result<ZevAnswer> {
    let source_tag = format!("cloudflare:{}", model_name);

    match wire_ans {
        WireAnswer::Noul {
            noul,
            confidence,
            source,
        } => {
            let mut probs = BTreeMap::new();
            let p_true = noul.clamp(0.0, 1.0);
            let p_false = (1.0 - p_true).clamp(0.0, 1.0);
            probs.insert("true".to_string(), p_true);
            probs.insert("false".to_string(), p_false);

            let conf = confidence.unwrap_or_else(|| p_true.max(p_false));
            let winner_is_true = p_true >= p_false;

            Ok(ZevAnswer {
                question_type: "boolean".into(),
                status: "ok".into(),
                decision: Some(serde_json::Value::Bool(winner_is_true)),
                confidence: conf,
                probabilities: probs,
                logits: BTreeMap::new(),
                uncertainty: UncertaintyMetrics {
                    top_probability: conf,
                    entropy_nats: 0.0,
                    concentration: 1.0,
                    unavailable_probability: 0.0,
                    margin: Some((p_true - p_false).abs()),
                    quantile_spread: None,
                },
                statistics: None,
                expected_value: None,
                temperature: 1.0,
                source: source.clone().or(Some(source_tag)),
            })
        }
        WireAnswer::Choice {
            choice,
            probabilities,
            confidence,
            source,
        } => {
            let top_prob = *confidence;
            Ok(ZevAnswer {
                question_type: "choice".into(),
                status: "ok".into(),
                decision: Some(serde_json::Value::String(choice.clone())),
                confidence: top_prob,
                probabilities: probabilities.clone(),
                logits: BTreeMap::new(),
                uncertainty: UncertaintyMetrics {
                    top_probability: top_prob,
                    entropy_nats: 0.0,
                    concentration: 1.0,
                    unavailable_probability: 0.0,
                    margin: None,
                    quantile_spread: None,
                },
                statistics: None,
                expected_value: None,
                temperature: 1.0,
                source: source.clone().or(Some(source_tag)),
            })
        }
        WireAnswer::Score {
            score,
            legend: _,
            probabilities,
            confidence,
            source,
        } => {
            // Check if ordinal smoothing is configured on ScoreQuestion
            let smoothed_probs = if let Question::Score(s) = question {
                if let Some(alpha) = s.ordinal_smoothing {
                    let mut keys: Vec<usize> = probabilities
                        .keys()
                        .filter_map(|k| k.parse::<usize>().ok())
                        .collect();
                    keys.sort_unstable();
                    let vec_probs: Vec<f64> = keys
                        .iter()
                        .map(|k| probabilities.get(&k.to_string()).copied().unwrap_or(0.0))
                        .collect();
                    let smoothed_vec =
                        crate::decoding::smooth_ordinal_probabilities(&vec_probs, alpha);
                    let mut smoothed_map = BTreeMap::new();
                    for (k, p) in keys.iter().zip(smoothed_vec.iter()) {
                        smoothed_map.insert(k.to_string(), *p);
                    }
                    smoothed_map
                } else {
                    probabilities.clone()
                }
            } else {
                probabilities.clone()
            };

            Ok(ZevAnswer {
                question_type: "score".into(),
                status: "ok".into(),
                decision: Some(serde_json::json!(*score)),
                confidence: *confidence,
                probabilities: smoothed_probs,
                logits: BTreeMap::new(),
                uncertainty: UncertaintyMetrics {
                    top_probability: *confidence,
                    entropy_nats: 0.0,
                    concentration: 1.0,
                    unavailable_probability: 0.0,
                    margin: None,
                    quantile_spread: None,
                },
                statistics: None,
                expected_value: Some(*score),
                temperature: 1.0,
                source: source.clone().or(Some(source_tag)),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChoiceQuestion, OptionDef};

    #[test]
    fn test_clef_config_builder() {
        let provider = CloudflareClefProvider::new("acc123", "token456")
            .with_flash()
            .with_timeout_ms(2500);

        assert_eq!(provider.model(), CLEF_FLASH_MODEL);
        assert_eq!(provider.config.timeout_ms, 2500);
        assert_eq!(
            provider.endpoint_url(),
            "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/typesafe/clef-flash"
        );
    }

    #[test]
    fn test_clef_endpoint_override() {
        let provider = CloudflareClefProvider::new("acc", "tok")
            .with_endpoint_override("https://my-worker.example.workers.dev/clef");

        assert_eq!(
            provider.endpoint_url(),
            "https://my-worker.example.workers.dev/clef"
        );
    }

    #[test]
    fn test_clef_parse_wrapped_response() {
        let wrapped_json = r#"{
            "result": {
                "model": "@cf/typesafe/clef",
                "answers": {
                    "q1": {
                        "type": "choice",
                        "choice": "refund",
                        "probabilities": {"refund": 0.88, "exchange": 0.12},
                        "confidence": 0.88
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 1}
            },
            "success": true,
            "errors": [],
            "messages": []
        }"#;

        let parsed = CloudflareClefProvider::parse_response_body(wrapped_json).unwrap();
        assert_eq!(parsed.model, "@cf/typesafe/clef");
        assert!(parsed.answers.contains_key("q1"));
    }

    #[test]
    fn test_clef_parse_unwrapped_response() {
        let unwrapped_json = r#"{
            "model": "@cf/typesafe/clef-flash",
            "answers": {
                "b1": {
                    "type": "noul",
                    "noul": 0.94,
                    "confidence": 0.94
                }
            },
            "usage": {"input_tokens": 8, "output_tokens": 1}
        }"#;

        let parsed = CloudflareClefProvider::parse_response_body(unwrapped_json).unwrap();
        assert_eq!(parsed.model, "@cf/typesafe/clef-flash");
        assert!(parsed.answers.contains_key("b1"));
    }

    #[test]
    fn test_clef_wire_answer_to_zev_choice() {
        let q = Question::Choice(ChoiceQuestion {
            instructions: "Pick one".into(),
            options: vec![
                OptionDef {
                    id: "opt_a".into(),
                    description: "First option".into(),
                },
                OptionDef {
                    id: "opt_b".into(),
                    description: "Second option".into(),
                },
            ],
            policy: Default::default(),
        });

        let mut probs = BTreeMap::new();
        probs.insert("opt_a".into(), 0.85);
        probs.insert("opt_b".into(), 0.15);

        let wire_ans = WireAnswer::Choice {
            choice: "opt_a".into(),
            probabilities: probs,
            confidence: 0.85,
            source: None,
        };

        let zev_ans = clef_wire_answer_to_zev(&q, &wire_ans, DEFAULT_CLEF_MODEL).unwrap();
        assert_eq!(zev_ans.status, "ok");
        assert_eq!(zev_ans.confidence, 0.85);
        assert_eq!(
            zev_ans.decision,
            Some(serde_json::Value::String("opt_a".into()))
        );
        assert_eq!(zev_ans.source.unwrap(), "cloudflare:@cf/typesafe/clef");
    }

    #[test]
    fn test_clef_wire_answer_to_zev_score_with_smoothing() {
        let q = Question::Score(crate::types::ScoreQuestion {
            instructions: "Rate severity".into(),
            levels: vec!["Low".into(), "Medium".into(), "High".into()],
            ordinal_smoothing: Some(0.10),
            policy: Default::default(),
        });

        let mut probs = BTreeMap::new();
        probs.insert("0".into(), 1.0);
        probs.insert("1".into(), 0.0);
        probs.insert("2".into(), 0.0);

        let wire_ans = WireAnswer::Score {
            score: 0.0,
            legend: BTreeMap::new(),
            probabilities: probs,
            confidence: 1.0,
            source: None,
        };

        let zev_ans = clef_wire_answer_to_zev(&q, &wire_ans, CLEF_FLASH_MODEL).unwrap();
        assert_eq!(zev_ans.status, "ok");
        // Smoothed: level 0 should be 0.90, level 1 should be 0.10
        assert_eq!(zev_ans.probabilities.get("0").copied().unwrap(), 0.90);
        assert_eq!(zev_ans.probabilities.get("1").copied().unwrap(), 0.10);
    }

    #[test]
    fn test_clef_wire_answer_to_zev_noul() {
        use crate::types::BooleanQuestion;

        let q = Question::Boolean(BooleanQuestion {
            instructions: "Is this inquiry about refunds?".into(),
            true_description: "Refund related".into(),
            false_description: "Not refund related".into(),
            policy: Default::default(),
        });

        let wire_ans = WireAnswer::Noul {
            noul: 0.92,
            confidence: Some(0.92),
            source: None,
        };

        let zev_ans = clef_wire_answer_to_zev(&q, &wire_ans, DEFAULT_CLEF_MODEL).unwrap();
        assert_eq!(zev_ans.question_type, "boolean");
        assert_eq!(zev_ans.status, "ok");
        assert_eq!(zev_ans.decision, Some(serde_json::Value::Bool(true)));
        assert_eq!(zev_ans.confidence, 0.92);
        assert!((zev_ans.probabilities["true"] - 0.92).abs() < 1e-6);
        assert!((zev_ans.probabilities["false"] - 0.08).abs() < 1e-6);
        assert!((zev_ans.uncertainty.margin.unwrap() - 0.84).abs() < 1e-6);
    }

    #[test]
    fn test_clef_parse_api_error_response() {
        let err_json = r#"{
            "result": null,
            "success": false,
            "errors": [
                {"code": 10000, "message": "Authentication error"},
                {"code": 10001, "message": "Account ID not found"}
            ],
            "messages": []
        }"#;

        let err = CloudflareClefProvider::parse_response_body(err_json).unwrap_err();
        match err {
            ZevError::Internal(msg) => {
                assert!(msg.contains("Cloudflare Workers AI error"));
                assert!(msg.contains("Authentication error"));
                assert!(msg.contains("Account ID not found"));
            }
            _ => panic!("Expected ZevError::Internal, got {:?}", err),
        }
    }

    #[test]
    fn test_clef_model_custom() {
        let provider = CloudflareClefProvider::new("my_acc", "my_token")
            .with_model("@cf/custom/fine-tuned-clef");

        assert_eq!(provider.model(), "@cf/custom/fine-tuned-clef");
        assert_eq!(
            provider.endpoint_url(),
            "https://api.cloudflare.com/client/v4/accounts/my_acc/ai/run/@cf/custom/fine-tuned-clef"
        );
    }
}
