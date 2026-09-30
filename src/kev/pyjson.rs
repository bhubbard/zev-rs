//! Python's `json.dumps(value, ensure_ascii=False)` and `str(value)`, byte for byte: prompts that embed JSON must
//! tokenize exactly as the reference servers' do.

use serde_json::Value;

/// Python float repr: shortest round-trip digits, exponent form outside [1e-4, 1e16).
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let a = f.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{f:e}");
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

/// json.dumps(v, ensure_ascii=False) with the default separators (", " and ": ").
pub fn dumps(v: &Value) -> String {
    let mut s = String::new();
    write(v, &mut s);
    s
}

fn write(v: &Value, s: &mut String) {
    match v {
        Value::Null => s.push_str("null"),
        Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_i64() || n.is_u64() => s.push_str(&n.to_string()),
        Value::Number(n) => s.push_str(&py_float(n.as_f64().unwrap_or(f64::NAN))),
        Value::String(x) => s.push_str(&serde_json::to_string(x).unwrap_or_default()),
        Value::Array(xs) => {
            s.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                write(x, s);
            }
            s.push(']');
        }
        Value::Object(m) => {
            s.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&serde_json::to_string(k).unwrap_or_default());
                s.push_str(": ");
                write(x, s);
            }
            s.push('}');
        }
    }
}

/// nlohmann::json::dump() (compact, "," and ":" separators, UTF-8 kept), as winnow's protocol.h renders data.
pub fn dumps_compact(v: &Value) -> String {
    let mut s = String::new();
    write_compact(v, &mut s);
    s
}

fn write_compact(v: &Value, s: &mut String) {
    match v {
        Value::Array(xs) => {
            s.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                write_compact(x, s);
            }
            s.push(']');
        }
        Value::Object(m) => {
            s.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&serde_json::to_string(k).unwrap_or_default());
                s.push(':');
                write_compact(x, s);
            }
            s.push('}');
        }
        _ => write(v, s),
    }
}

/// json.dumps(v, ensure_ascii=False, indent=n): one item per line, "," separators, empty containers inline.
pub fn dumps_indent(v: &Value, n: usize) -> String {
    let mut s = String::new();
    write_indent(v, n, 0, &mut s);
    s
}

fn write_indent(v: &Value, n: usize, depth: usize, s: &mut String) {
    let pad = |d: usize| " ".repeat(n * d);
    match v {
        Value::Array(xs) if !xs.is_empty() => {
            s.push('[');
            for (i, x) in xs.iter().enumerate() {
                s.push_str(if i > 0 { ",\n" } else { "\n" });
                s.push_str(&pad(depth + 1));
                write_indent(x, n, depth + 1, s);
            }
            s.push('\n');
            s.push_str(&pad(depth));
            s.push(']');
        }
        Value::Object(m) if !m.is_empty() => {
            s.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                s.push_str(if i > 0 { ",\n" } else { "\n" });
                s.push_str(&pad(depth + 1));
                s.push_str(&serde_json::to_string(k).unwrap_or_default());
                s.push_str(": ");
                write_indent(x, n, depth + 1, s);
            }
            s.push('\n');
            s.push_str(&pad(depth));
            s.push('}');
        }
        _ => write(v, s),
    }
}

/// Python `repr()` of a JSON value as json.loads builds it (dict, list, str, int, float, bool, None).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => {
            let q = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut o = String::from(q);
            for c in s.chars() {
                match c {
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if c == q => {
                        o.push('\\');
                        o.push(c);
                    }
                    c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                        o.push_str(&format!("\\x{:02x}", c as u32))
                    }
                    c => o.push(c),
                }
            }
            o.push(q);
            o
        }
        Value::Array(xs) => format!(
            "[{}]",
            xs.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, x)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => py_str(v),
    }
}

/// Python `str()` of a JSON value as json.loads builds it: scalars as pydantic parses them, containers as repr().
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            match py_float(f).as_str() {
                "NaN" => "nan".into(),
                "Infinity" => "inf".into(),
                "-Infinity" => "-inf".into(),
                s => s.into(),
            }
        }
        Value::Null => "None".into(),
        _ => py_repr(v),
    }
}

/// A string as is, anything else as json.dumps (decider's `_txt`).
pub fn txt(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => dumps(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_matches_python() {
        let v: Value = serde_json::from_str(
            r#"{"a": [1, 2.0, 1e-7, "é\n\u0001\"x\\"], "b": {"c": null, "d": true}, "e": {}, "f": []}"#,
        )
        .unwrap();
        // python3 -c 'import json; print(json.dumps(json.loads(same), ensure_ascii=False))'
        assert_eq!(
            dumps(&v),
            "{\"a\": [1, 2.0, 1e-07, \"é\\n\\u0001\\\"x\\\\\"], \"b\": {\"c\": null, \"d\": true}, \"e\": {}, \"f\": []}"
        );
        assert_eq!(py_float(0.1), "0.1");
        assert_eq!(py_float(2.5e16), "2.5e+16");
        assert_eq!(py_str(&Value::Bool(true)), "True");
        let v: Value = serde_json::from_str(
            r#"{"level": "very", "q": [1, 2.0, null, true, "it's", "a\"b", "x\ny\\z\u0001"]}"#,
        )
        .unwrap();
        // python: str(json.loads(same))
        assert_eq!(
            py_str(&v),
            r#"{'level': 'very', 'q': [1, 2.0, None, True, "it's", 'a"b', 'x\ny\\z\x01']}"#
        );
        let v: Value = serde_json::from_str(r#"{"a":[1,{"b":[]}],"c":{},"d":"x"}"#).unwrap();
        // python: json.dumps(same, ensure_ascii=False, indent=2)
        assert_eq!(dumps_indent(&v, 2), "{\n  \"a\": [\n    1,\n    {\n      \"b\": []\n    }\n  ],\n  \"c\": {},\n  \"d\": \"x\"\n}");
    }
}
