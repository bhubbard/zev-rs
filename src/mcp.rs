use crate::error::{Result, ZevError};
use crate::types::{ChoiceQuestion, OptionDef, Policy, Question, ScoreQuestion, ZevRequest};
use crate::ZevEngine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

// -------------------------------------------------------------------------------------------------
// JSON-RPC 2.0 & MCP Protocol Data Structures
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    #[serde(rename = "text")]
    Text { text: String },
}

// -------------------------------------------------------------------------------------------------
// Tool Arguments and Response Types
// -------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ClassifyArgs {
    pub state: String,
    pub instructions: String,
    pub options: Vec<OptionDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassifyResult {
    pub winning_option: String,
    pub decision: String,
    pub confidence: f64,
    pub probabilities: BTreeMap<String, f64>,
}

fn default_threshold() -> f64 {
    0.5
}

#[derive(Debug, Deserialize)]
struct FilterArgs {
    pub state: String,
    pub predicate: String,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilterResult {
    pub pass: bool,
    pub confidence: f64,
}

#[derive(Debug, Deserialize)]
struct ScoreArgs {
    pub state: String,
    pub instructions: String,
    pub levels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreResult {
    pub score: f64,
    pub expected_value: f64,
    pub probabilities: BTreeMap<String, f64>,
    pub level_probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct BatchArgs {
    pub rows: Vec<Value>,
    pub predicate: String,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BatchResult {
    pub rows: Vec<Value>,
    pub count: usize,
}

// -------------------------------------------------------------------------------------------------
// Protocol Message Dispatch
// -------------------------------------------------------------------------------------------------

/// Handles a single incoming JSON-RPC 2.0 MCP message synchronously.
/// Notifications (requests without an `id` or `notifications/*`) produce no response and return `None`.
pub fn handle_message(msg_str: &str, engine: &ZevEngine) -> Option<String> {
    let raw_val: Value = match serde_json::from_str(msg_str) {
        Ok(v) => v,
        Err(_) => {
            let err_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: "Parse error".to_string(),
                    data: None,
                }),
            };
            return Some(serde_json::to_string(&err_resp).unwrap());
        }
    };

    let req: JsonRpcRequest = match serde_json::from_value(raw_val) {
        Ok(r) => r,
        Err(e) => {
            let err_resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: Value::Null,
                result: None,
                error: Some(JsonRpcError {
                    code: -32600,
                    message: format!("Invalid Request: {}", e),
                    data: None,
                }),
            };
            return Some(serde_json::to_string(&err_resp).unwrap());
        }
    };

    // Notifications must not be responded to
    if req.id.is_none() || req.method.starts_with("notifications/") {
        return None;
    }

    let id = req.id.unwrap();

    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(id),
        "ping" => handle_ping(id),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(id, req.params, engine),
        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
                data: None,
            }),
        },
    };

    Some(serde_json::to_string(&resp).unwrap())
}

/// Asynchronous in-memory message processing helper for deterministic testing without blocking on stdin.
pub async fn process_message(req_str: &str, engine: &ZevEngine) -> Option<String> {
    handle_message(req_str, engine)
}

fn handle_initialize(id: Value) -> JsonRpcResponse {
    let result = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "zev-mcp",
            "version": env!("CARGO_PKG_VERSION")
        }
    });
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    }
}

fn handle_ping(id: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!({})),
        error: None,
    }
}

fn handle_tools_list(id: Value) -> JsonRpcResponse {
    let tools = json!({
        "tools": [
            {
                "name": "zev_classify",
                "description": "Zero-token calibrated classification across multiple options.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "Context or input text to evaluate"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Classification instruction or criteria"
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": { "type": "string", "description": "Unique identifier for option" },
                                    "description": { "type": "string", "description": "Description of the option" }
                                },
                                "required": ["id", "description"]
                            },
                            "description": "Candidate options for classification"
                        }
                    },
                    "required": ["state", "instructions", "options"]
                }
            },
            {
                "name": "zev_filter",
                "description": "Zero-token boolean filtering of an input state against a predicate.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "Input text to evaluate against predicate"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Filter predicate or condition to test"
                        },
                        "threshold": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Probability threshold required to pass (default: 0.5)"
                        }
                    },
                    "required": ["state", "predicate"]
                }
            },
            {
                "name": "zev_score",
                "description": "Zero-token ordinal scoring and expected value calculation across discrete levels.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "Input text to score"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Scoring instructions or evaluation criteria"
                        },
                        "levels": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Ordered scale levels from lowest to highest"
                        }
                    },
                    "required": ["state", "instructions", "levels"]
                }
            },
            {
                "name": "zev_batch",
                "description": "Zero-token batch filtering of tabular rows against a predicate.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": { "type": "object" },
                            "description": "Array of JSON row objects to filter"
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Filter predicate condition"
                        },
                        "threshold": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Probability threshold to pass (default: 0.5)"
                        }
                    },
                    "required": ["rows", "predicate", "threshold"]
                }
            }
        ]
    });
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(tools),
        error: None,
    }
}

fn handle_tools_call(id: Value, params: Option<Value>, engine: &ZevEngine) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Missing params in tools/call".to_string(),
                    data: None,
                }),
            };
        }
    };

    let name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32602,
                    message: "Missing 'name' in tools/call params".to_string(),
                    data: None,
                }),
            };
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    let tool_res = match name {
        "zev_classify" => call_zev_classify(arguments, engine),
        "zev_filter" => call_zev_filter(arguments, engine),
        "zev_score" => call_zev_score(arguments, engine),
        "zev_batch" => call_zev_batch(arguments, engine),
        _ => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Unknown tool: {}", name),
            }],
            is_error: true,
        },
    };

    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::to_value(&tool_res).unwrap()),
        error: None,
    }
}

// -------------------------------------------------------------------------------------------------
// Tool Implementations
// -------------------------------------------------------------------------------------------------

fn call_zev_classify(arguments: Value, engine: &ZevEngine) -> CallToolResult {
    let args: ClassifyArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for zev_classify: {}", e),
                }],
                is_error: true,
            };
        }
    };

    if args.options.len() < 2 {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Options must contain at least 2 candidates for classification".to_string(),
            }],
            is_error: true,
        };
    }

    let question = Question::Choice(ChoiceQuestion {
        instructions: args.instructions,
        options: args.options,
        policy: Policy {
            allow_abstain: false,
            ..Default::default()
        },
    });

    let mut questions = BTreeMap::new();
    questions.insert("classify".to_string(), question);

    let req = ZevRequest {
        state: Value::String(args.state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    match engine.evaluate(&req) {
        Ok(resp) => {
            if let Some(ans) = resp.answers.get("classify") {
                let winning_option = match &ans.decision {
                    Some(Value::String(s)) => s.clone(),
                    _ => {
                        // Fallback to max probability key if decision wasn't a string
                        ans.probabilities
                            .iter()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(k, _)| k.clone())
                            .unwrap_or_default()
                    }
                };

                let res = ClassifyResult {
                    winning_option: winning_option.clone(),
                    decision: winning_option,
                    confidence: ans.confidence,
                    probabilities: ans.probabilities.clone(),
                };

                let text = serde_json::to_string_pretty(&res)
                    .unwrap_or_else(|e| format!("Serialization error: {}", e));
                CallToolResult {
                    content: vec![ToolContent::Text { text }],
                    is_error: false,
                }
            } else {
                CallToolResult {
                    content: vec![ToolContent::Text {
                        text: "Internal error: missing classify answer in response".to_string(),
                    }],
                    is_error: true,
                }
            }
        }
        Err(e) => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Classification evaluation error: {}", e),
            }],
            is_error: true,
        },
    }
}

fn call_zev_filter(arguments: Value, engine: &ZevEngine) -> CallToolResult {
    let args: FilterArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for zev_filter: {}", e),
                }],
                is_error: true,
            };
        }
    };

    let question = Question::Choice(ChoiceQuestion {
        instructions: format!("Evaluate whether the context satisfies: {}", args.predicate),
        options: vec![
            OptionDef {
                id: "pass".to_string(),
                description: format!("Matches or confirms: {}", args.predicate),
            },
            OptionDef {
                id: "reject".to_string(),
                description: "Unrelated, benign, neutral, other topic, or does not match".to_string(),
            },
        ],
        policy: Policy {
            allow_abstain: false,
            ..Default::default()
        },
    });

    let mut questions = BTreeMap::new();
    questions.insert("filter".to_string(), question);

    let req = ZevRequest {
        state: Value::String(args.state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    match engine.evaluate(&req) {
        Ok(resp) => {
            if let Some(ans) = resp.answers.get("filter") {
                let p_pass = ans.probabilities.get("pass").copied().unwrap_or(0.0);
                let pass_logit = ans.logits.get("pass").copied().unwrap_or(0.5);
                let has_match = pass_logit > 0.5 + 1e-6;
                let pass = has_match && p_pass >= args.threshold;
                let confidence = if pass { p_pass } else { 1.0 - p_pass };
                let res = FilterResult {
                    pass,
                    confidence,
                };
                let text = serde_json::to_string_pretty(&res)
                    .unwrap_or_else(|e| format!("Serialization error: {}", e));
                CallToolResult {
                    content: vec![ToolContent::Text { text }],
                    is_error: false,
                }
            } else {
                CallToolResult {
                    content: vec![ToolContent::Text {
                        text: "Internal error: missing filter answer in response".to_string(),
                    }],
                    is_error: true,
                }
            }
        }
        Err(e) => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Filter evaluation error: {}", e),
            }],
            is_error: true,
        },
    }
}

fn call_zev_score(arguments: Value, engine: &ZevEngine) -> CallToolResult {
    let args: ScoreArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for zev_score: {}", e),
                }],
                is_error: true,
            };
        }
    };

    if args.levels.len() < 2 {
        return CallToolResult {
            content: vec![ToolContent::Text {
                text: "Levels must contain at least 2 entries for scoring".to_string(),
            }],
            is_error: true,
        };
    }

    let question = Question::Score(ScoreQuestion {
        instructions: args.instructions,
        levels: args.levels.clone(),
        policy: Policy {
            allow_abstain: false,
            ..Default::default()
        },
    });

    let mut questions = BTreeMap::new();
    questions.insert("score".to_string(), question);

    let req = ZevRequest {
        state: Value::String(args.state),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    };

    match engine.evaluate(&req) {
        Ok(resp) => {
            if let Some(ans) = resp.answers.get("score") {
                let score = ans.expected_value.unwrap_or(0.0);
                let mut level_probabilities = BTreeMap::new();
                for (i, lvl) in args.levels.iter().enumerate() {
                    let p = ans.probabilities.get(&i.to_string()).copied().unwrap_or(0.0);
                    level_probabilities.insert(lvl.clone(), p);
                }

                let res = ScoreResult {
                    score,
                    expected_value: score,
                    probabilities: ans.probabilities.clone(),
                    level_probabilities,
                };

                let text = serde_json::to_string_pretty(&res)
                    .unwrap_or_else(|e| format!("Serialization error: {}", e));
                CallToolResult {
                    content: vec![ToolContent::Text { text }],
                    is_error: false,
                }
            } else {
                CallToolResult {
                    content: vec![ToolContent::Text {
                        text: "Internal error: missing score answer in response".to_string(),
                    }],
                    is_error: true,
                }
            }
        }
        Err(e) => CallToolResult {
            content: vec![ToolContent::Text {
                text: format!("Score evaluation error: {}", e),
            }],
            is_error: true,
        },
    }
}

fn call_zev_batch(arguments: Value, engine: &ZevEngine) -> CallToolResult {
    let args: BatchArgs = match serde_json::from_value(arguments) {
        Ok(a) => a,
        Err(e) => {
            return CallToolResult {
                content: vec![ToolContent::Text {
                    text: format!("Invalid arguments for zev_batch: {}", e),
                }],
                is_error: true,
            };
        }
    };

    let mut surviving_rows = Vec::new();

    for row in args.rows {
        let text = if let Some(s) = row.get("text").and_then(|v| v.as_str()) {
            s.to_string()
        } else if let Some(s) = row.get("body").and_then(|v| v.as_str()) {
            s.to_string()
        } else if let Some(s) = row.get("content").and_then(|v| v.as_str()) {
            s.to_string()
        } else if let Some(s) = row.get("state").and_then(|v| v.as_str()) {
            s.to_string()
        } else {
            row.to_string()
        };

        let question = Question::Choice(ChoiceQuestion {
            instructions: format!("Evaluate whether the context satisfies: {}", args.predicate),
            options: vec![
                OptionDef {
                    id: "pass".to_string(),
                    description: format!("Matches or confirms: {}", args.predicate),
                },
                OptionDef {
                    id: "reject".to_string(),
                    description: "Unrelated, benign, neutral, other topic, or does not match".to_string(),
                },
            ],
            policy: Policy {
                allow_abstain: false,
                ..Default::default()
            },
        });

        let mut questions = BTreeMap::new();
        questions.insert("batch_filter".to_string(), question);

        let req = ZevRequest {
            state: Value::String(text),
            questions,
            model: None,
            temperature: None,
            enable_temporal_facts: false,
        };

        match engine.evaluate(&req) {
            Ok(resp) => {
                if let Some(ans) = resp.answers.get("batch_filter") {
                    let p_pass = ans.probabilities.get("pass").copied().unwrap_or(0.0);
                    let pass_logit = ans.logits.get("pass").copied().unwrap_or(0.5);
                    let has_match = pass_logit > 0.5 + 1e-6;
                    if has_match && p_pass >= args.threshold {
                        surviving_rows.push(row);
                    }
                }
            }
            Err(e) => {
                return CallToolResult {
                    content: vec![ToolContent::Text {
                        text: format!("Batch row evaluation error: {}", e),
                    }],
                    is_error: true,
                };
            }
        }
    }

    let count = surviving_rows.len();
    let res = BatchResult {
        rows: surviving_rows,
        count,
    };

    let text = serde_json::to_string_pretty(&res)
        .unwrap_or_else(|e| format!("Serialization error: {}", e));
    CallToolResult {
        content: vec![ToolContent::Text { text }],
        is_error: false,
    }
}

// -------------------------------------------------------------------------------------------------
// Server Loops
// -------------------------------------------------------------------------------------------------

/// Runs the MCP stdio server over standard input and output.
pub async fn run_stdio_server() -> Result<()> {
    let engine = Arc::new(ZevEngine::default());
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    run_mcp_server_io(engine, stdin, stdout).await
}

/// Runs the MCP server loop over arbitrary async reader and writer.
pub async fn run_mcp_server_io<R, W>(
    engine: Arc<ZevEngine>,
    reader: R,
    mut writer: W,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await.map_err(ZevError::Io)? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(resp_str) = handle_message(trimmed, engine.as_ref()) {
            writer
                .write_all(resp_str.as_bytes())
                .await
                .map_err(ZevError::Io)?;
            writer
                .write_all(b"\n")
                .await
                .map_err(ZevError::Io)?;
            writer.flush().await.map_err(ZevError::Io)?;
        }
    }
    Ok(())
}
