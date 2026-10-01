use std::fs::File;
use std::io::{BufRead, BufReader};
use serde::Deserialize;
use serde_json::Value;
use zev::types::SystemOneRequest;
use zev::wire::WireQuestion;
use zev::ZevEngine;

#[derive(Debug, Deserialize)]
struct JevBenchItem {
    id: String,
    family: String,
    state: Value,
    question: WireQuestion,
    expected: Value,
}

fn extract_prediction_token(ans_val: &Value) -> String {
    if let Some(choice) = ans_val.get("choice").and_then(|v| v.as_str()) {
        choice.to_string()
    } else if let Some(noul_val) = ans_val.get("noul").and_then(|v| v.as_f64()) {
        if noul_val >= 0.5 { "true".to_string() } else { "false".to_string() }
    } else if let Some(probs) = ans_val.get("probabilities").and_then(|p| p.as_object()) {
        probs.iter().max_by(|a, b| {
            let pa = a.1.as_f64().unwrap_or(0.0);
            let pb = b.1.as_f64().unwrap_or(0.0);
            pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
        }).map(|(k, _)| k.as_str()).unwrap_or("0").to_string()
    } else if let Some(score) = ans_val.get("score").and_then(|v| v.as_f64()) {
        format!("{}", score.round() as i64)
    } else {
        String::new()
    }
}

fn matches_expected(pred: &str, expected: &str) -> bool {
    let clean_pred = pred.trim().trim_matches('"').to_lowercase();
    let clean_exp = expected.trim().trim_matches('"').to_lowercase();
    if clean_pred == clean_exp { return true; }
    let is_pred_yes = clean_pred == "true" || clean_pred == "yes" || clean_pred == "1";
    let is_exp_yes = clean_exp == "true" || clean_exp == "yes" || clean_exp == "1";
    if is_pred_yes && is_exp_yes { return true; }
    let is_pred_no = clean_pred == "false" || clean_pred == "no" || clean_pred == "0";
    let is_exp_no = clean_exp == "false" || clean_exp == "no" || clean_exp == "0";
    if is_pred_no && is_exp_no { return true; }
    false
}

fn load_jevbench() -> Vec<JevBenchItem> {
    let mut items = Vec::new();
    for fname in &["easy.jsonl", "original.jsonl", "hard.jsonl"] {
        let path = format!("datasets/jevbench/{fname}");
        if let Ok(file) = File::open(&path) {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    if let Ok(item) = serde_json::from_str::<JevBenchItem>(&line) {
                        items.push(item);
                    }
                }
            }
        }
    }
    items
}

fn main() {
    let engine = ZevEngine::default();
    let items = load_jevbench();
    let mut failures = Vec::new();

    for item in &items {
        let req = SystemOneRequest {
            state: item.state.clone(),
            questions: [("q".to_string(), item.question.clone())].into(),
            model: "zev-default".into(),
        };
        if let Ok(resp) = engine.evaluate_system_one(&req) {
            if let Some(ans) = resp.answers.get("q") {
                let pred = extract_prediction_token(ans);
                let exp_str = match &item.expected {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => b.to_string(),
                    other => other.to_string(),
                };
                if !matches_expected(&pred, &exp_str) {
                    failures.push((item, pred, exp_str));
                }
            }
        }
    }

    println!("Total Failures: {} / {}", failures.len(), items.len());
    println!("Current Accuracy: {:.2}%\n", (items.len() - failures.len()) as f64 / items.len() as f64 * 100.0);

    let mut by_family: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (item, _, _) in &failures {
        *by_family.entry(item.family.clone()).or_insert(0) += 1;
    }
    println!("Failures by Family:");
    for (fam, count) in &by_family {
        println!("  {:<20}: {}", fam, count);
    }

    println!("\nSample Detailed Failures:");
    for (item, pred, exp) in failures.iter().take(25) {
        let state_str = match &item.state {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let preview = if state_str.len() > 100 { format!("{}...", &state_str[..100]) } else { state_str };
        println!("------------------------------------------------------------");
        println!("ID: {} | Family: {}", item.id, item.family);
        println!("State: {}", preview);
        println!("Expected: {} | Predicted: {}", exp, pred);
    }
}
