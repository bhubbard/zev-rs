//! Candle backend: Kev checkpoints (Qwen3.5 hybrid backbone + LoRA + pointer head) served on /v1/systemone with the
//! request mapping of kev `api.py` / `model.py::encode` (row form, state prefix cached across requests).

pub mod kernels;
pub mod model;
pub mod serve;

use serde_json::{Map, Value};
use tokenizers::Tokenizer;

// kev.model.SPECIAL: <state>, <q>, <opt>, </opt>, <decide>
pub const SPECIAL: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];
pub const SERVE_MAX_STATE: usize = 8192;
pub const SERVE_MAX_BRANCH: usize = 8192;
pub const MAX_OPTIONS: usize = 255;

/// Python `str()` of a JSON scalar as pydantic parses it (int, float, bool, str).
fn py_str(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => py_float(n.as_f64().unwrap_or(f64::NAN)),
        _ => String::new(),
    }
}

/// Python float repr: shortest round-trip digits, exponent form outside [1e-4, 1e16).
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.into();
    }
    let a = f.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{f:e}"); // 1e-5, 1.5e20
        let (m, e) = s.split_once('e').unwrap();
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    if f == f.trunc() {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// kev.api.render: flatten str | object | array into the text the model sees.
pub fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::Array(xs) => xs
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(m) => m
            .iter()
            .map(|(k, x)| match x {
                Value::Object(_) | Value::Array(_) => {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                }
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        s => py_str(s),
    }
}

fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => name.into(),
        Some(Value::String(s)) if s.is_empty() => name.into(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

/// One question as the model sees it plus what the answer needs.
pub struct Question {
    pub id: String,
    pub kind: String,
    pub instr: String,
    pub options: Vec<String>,
    pub keys: Vec<String>,
    pub legend: Option<Map<String, Value>>,
}

/// kev.api.to_record, with SystemOneRequest's validation. -> (state text, questions).
pub fn to_record(req: &Value) -> Result<(String, Vec<Question>), String> {
    let obj = req.as_object().ok_or("request must be an object")?;
    let state = obj.get("state").ok_or("missing state")?;
    let qs = obj
        .get("questions")
        .and_then(Value::as_object)
        .ok_or("questions must be an object")?;
    if qs.is_empty() {
        return Err("questions must have at least 1 item".into());
    }
    let mut out = Vec::new();
    for (id, q) in qs {
        let kind = q
            .get("type")
            .and_then(Value::as_str)
            .ok_or(format!("question {id}: missing type"))?;
        let instr = render(q.get("instructions").unwrap_or(&Value::Null), 0);
        let crit = q.get("criteria");
        let (options, keys, legend) = match kind {
            "noul" => {
                let c = crit.and_then(Value::as_object);
                let opts = vec![
                    option_text("no", c.and_then(|c| c.get("false"))),
                    option_text("yes", c.and_then(|c| c.get("true"))),
                ];
                (opts, vec!["false".into(), "true".into()], None)
            }
            "choice" => {
                let c = crit
                    .and_then(Value::as_object)
                    .ok_or(format!("question {id}: criteria must be an object"))?;
                if c.is_empty() || c.len() > MAX_OPTIONS {
                    return Err(format!(
                        "question {id}: criteria must have 1..{MAX_OPTIONS} options"
                    ));
                }
                (
                    c.iter().map(|(k, v)| option_text(k, Some(v))).collect(),
                    c.keys().cloned().collect(),
                    None,
                )
            }
            "score" => {
                let c = crit
                    .and_then(Value::as_array)
                    .ok_or(format!("question {id}: criteria must be a list"))?;
                if c.is_empty() || c.len() > MAX_OPTIONS {
                    return Err(format!(
                        "question {id}: criteria must have 1..{MAX_OPTIONS} levels"
                    ));
                }
                let opts: Vec<String> = c.iter().map(|x| render(x, 0)).collect();
                let keys: Vec<String> = (0..c.len()).map(|i| i.to_string()).collect();
                let legend = keys
                    .iter()
                    .cloned()
                    .zip(opts.iter().map(|s| Value::String(s.clone())))
                    .collect();
                (opts, keys, Some(legend))
            }
            other => return Err(format!("question {id}: unknown type {other}")),
        };
        out.push(Question {
            id: id.clone(),
            kind: kind.into(),
            instr,
            options,
            keys,
            legend,
        });
    }
    Ok((render(state, 0), out))
}

/// A question's causal row after the state: its tokens and the readout offsets within it.
pub struct RowSpec {
    pub ids: Vec<u32>,
    pub decide: usize,
    pub opts: Vec<usize>,
}

pub struct Encoder {
    pub tok: Tokenizer,
    special: [u32; 5],
    forge: regex::Regex,
}

impl Encoder {
    pub fn new(tok: Tokenizer) -> Result<Self, String> {
        let mut special = [0u32; 5];
        for (i, s) in SPECIAL.iter().enumerate() {
            special[i] = tok.token_to_id(s).ok_or(format!("tokenizer lacks {s}"))?;
        }
        Ok(Self {
            tok,
            special,
            forge: regex::Regex::new(r"<\|([A-Za-z0-9_]+)\|>").unwrap(),
        })
    }

    /// kev.model.user_tokens: caller text can never produce delimiter tokens.
    fn user(&self, text: &str) -> Result<Vec<u32>, String> {
        let safe = self.forge.replace_all(text, "<¦$1¦>");
        Ok(self
            .tok
            .encode(safe.as_ref(), false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    }

    /// kev.model.encode split into rows (rows_of): state ids, one row per question. Positions continue the state.
    pub fn encode(&self, state: &str, qs: &[Question]) -> Result<(Vec<u32>, Vec<RowSpec>), String> {
        let [s_id, q_id, o_id, c_id, d_id] = self.special;
        let st = self.user(state)?;
        let mut s = vec![s_id];
        s.extend_from_slice(&st[..st.len().min(SERVE_MAX_STATE - 1)]);
        let mut rows = Vec::new();
        for q in qs {
            let mut br = vec![q_id];
            br.extend(self.user(&q.instr)?);
            let mut opts = Vec::new();
            for o in &q.options {
                br.push(o_id);
                br.extend(self.user(o)?);
                br.push(c_id);
                opts.push(br.len() - 1);
            }
            br.push(d_id);
            if br.len() > SERVE_MAX_BRANCH - s.len() {
                return Err(format!("branch too long: {} tokens with a {}-token state (row limit {SERVE_MAX_BRANCH})", br.len(), s.len()));
            }
            rows.push(RowSpec {
                decide: br.len() - 1,
                opts,
                ids: br,
            });
        }
        Ok((s, rows))
    }
}

fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// kev.api.to_answers
pub fn to_answers(probs: &[Vec<f64>], qs: &[Question]) -> Map<String, Value> {
    let mut out = Map::new();
    for (p, q) in probs.iter().zip(qs) {
        let argmax = (0..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m });
        let dist: Map<String, Value> = q
            .keys
            .iter()
            .zip(p)
            .map(|(k, v)| (k.clone(), Value::from(round4(*v))))
            .collect();
        let a = match q.kind.as_str() {
            "noul" => serde_json::json!({"type": "noul", "noul": round4(p[1])}),
            "choice" => {
                let k = p.len() as f64;
                let conf = if p.len() == 1 {
                    1.0
                } else {
                    (p[argmax] - 1.0 / k) / (1.0 - 1.0 / k)
                };
                serde_json::json!({"type": "choice", "choice": q.keys[argmax], "confidence": round4(conf), "probabilities": dist})
            }
            _ => {
                let score: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                let conf = if p.len() == 1 {
                    1.0
                } else {
                    1.0 - p
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v * (i as f64 - argmax as f64).abs())
                        .sum::<f64>()
                        / (p.len() - 1) as f64
                };
                serde_json::json!({"type": "score", "score": round4(score), "legend": q.legend, "probabilities": dist, "confidence": round4(conf)})
            }
        };
        out.insert(q.id.clone(), a);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_matches_python() {
        let v: Value = serde_json::from_str(
            r#"{"a": 1, "b": 1.0, "c": true, "d": null, "e": [1, {"x": "y"}], "f": {"g": 1e-5}}"#,
        )
        .unwrap();
        // python: kev.api.render(json.loads(same))
        assert_eq!(
            render(&v, 0),
            "a: 1\nb: 1.0\nc: True\nd: \ne:\n  - 1\n  - x: y\nf:\n  g: 1e-05"
        );
        assert_eq!(py_float(0.1), "0.1");
        assert_eq!(py_float(2.5e16), "2.5e+16");
        assert_eq!(option_text("k", Some(&Value::String(String::new()))), "k");
    }
}
