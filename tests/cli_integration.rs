use std::process::Command;

#[test]
fn test_cli_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_zev"))
        .arg("--help")
        .output()
        .expect("failed to run zev binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Zev: High-performance"));
    assert!(stdout.contains("decide"));
    assert!(stdout.contains("route"));
    assert!(stdout.contains("gate"));
    assert!(stdout.contains("tev1"));
}

#[test]
fn test_cli_route() {
    let output = Command::new(env!("CARGO_BIN_EXE_zev"))
        .args(&[
            "route",
            "--state",
            "Critical: database server is returning 500 error code",
            "--routes",
            r#"{"billing": "Invoice questions", "infra": "Database and server outages"}"#,
        ])
        .output()
        .expect("failed to run zev route");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["destination"], "infra");
    assert!(json["probability"].as_f64().unwrap() > 0.5);
}

#[test]
fn test_cli_gate() {
    let output = Command::new(env!("CARGO_BIN_EXE_zev"))
        .args(&[
            "gate",
            "--state",
            "Yes. Confirms urgent emergency completely",
            "--instructions",
            "urgent emergency",
            "--threshold",
            "0.0",
        ])
        .output()
        .expect("failed to run zev gate");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["decision"], true);
    assert_eq!(json["passed"], true);
}

#[test]
fn test_cli_tev1_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_zev"))
        .args(&[
            "tev1",
            "--state",
            "Returns are allowed within 30 days. This purchase was 12 days ago.",
            "--question",
            "Is this return within the allowed window?",
            "--options",
            "A: Yes, B: No, C: Not enough information",
        ])
        .output()
        .expect("failed to run zev tev1");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["answer"], "A");
    assert_eq!(json["choice"], "Yes");
}

#[test]
fn test_cli_tev1_prompt_text() {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_zev"))
        .arg("tev1")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn zev tev1");

    let prompt = r#"
State       Returns are allowed within 30 days. This purchase was 12 days ago.
Question    Is this return within the allowed window?
Options     A: Yes   B: No   C: Not enough information
Answer      A
"#;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(prompt.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["answer"], "A");
}

#[test]
fn test_cli_decide_native() {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_zev"))
        .arg("decide")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn zev decide");

    let req = r#"{
      "state": "Payment failed due to card expiration",
      "questions": {
        "cat": {
          "type": "choice",
          "instructions": "Category",
          "options": [
            {"id": "billing", "description": "Card billing and invoice"},
            {"id": "shipping", "description": "Package delivery"}
          ]
        }
      }
    }"#;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(req.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert_eq!(json["answers"]["cat"]["decision"], "billing");
}

#[test]
fn test_cli_systemone() {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_zev"))
        .arg("systemone")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn zev systemone");

    let req = r#"{
      "state": "Database connection timed out",
      "questions": {
        "urgent": {
          "type": "noul",
          "instructions": "Is urgent outage"
        }
      }
    }"#;

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(req.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("valid json output");
    assert!(json["answers"]["urgent"]["noul"].is_number());
}
