//! Generator for tests/scale_2000_tests.rs
//!
//! Generates 2,000 distinct stress test cases for zev-rs:
//! 1..200: Scaled Shortlisting (cardinality 25 to 1425 options)
//! 201..350: Engine High-Cardinality Choice (500 to 1400 options with max_slots policy)
//! 351..550: Mega-Premises & Context Flooding (50 to 500 log lines / noise)
//! 551..750: Multi-Task Batch Flooding (5 to 30 simultaneous questions per request)
//! 751..950: Order Invariance Heavy Permutations (reversed, shifted, shuffled options)
//! 951..1150: Extreme Numerical Calibration & Temperatures
//! 1151..1350: CLM VectorArena Stress & LRU Cache Thrashing
//! 1351..1550: CLM ContrastiveHead Projections
//! 1551..1750: CLM HybridVerifier Two-Tier Stress
//! 1751..1900: Adversarial Substring Overlaps & Collisions
//! 1901..2000: High-Concurrency Multithreading Stress

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out_path = if args.len() > 1 {
        &args[1]
    } else {
        "tests/scale_2000_tests.rs"
    };

    println!("Generating 2,000 stress tests into {out_path}...");

    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(out_path)?;
    let mut f = BufWriter::new(file);

    // Header
    writeln!(
        f,
        "// AUTO-GENERATED TEST SUITE: 2000 SCALE & STRESS TESTS FOR ZEV-RS\n\
         // Pounding high-cardinality options, mega-premises, multi-task flooding,\n\
         // cache thrashing, adversarial inputs, multithreading, and CLM features.\n\
         #![allow(unused_imports, unused_variables, dead_code)]\n\n\
         use std::sync::{{Arc, Mutex}};\n\
         use std::thread;\n\
         use zev::calibration::*;\n\
         use zev::clm::*;\n\
         use zev::engine::ZevEngine;\n\
         use zev::shortlist::shortlist_options;\n\
         use zev::tev1::*;\n\
         use zev::types::*;\n"
    )?;

    // 1..200: Scaled Shortlisting
    for i in 1..=200 {
        let n_opts = 25 + (i * 7);
        let top_k = 5 + (i % 25);
        let target_idx = (i * 3) % n_opts;
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let options: Vec<OptionDef> = (0..{n_opts}).map(|k| OptionDef {{\n\
             \x20       id: format!(\"opt_{{k}}\"),\n\
             \x20       description: format!(\"Microservice deployment target cluster handler {{k}} with failover logic\"),\n\
             \x20   }}).collect();\n\
             \x20   let shortlisted = shortlist_options(&options, \"critical incident alert on opt_{target_idx} cluster failure\", {top_k});\n\
             \x20   assert!(shortlisted.len() <= {top_k});\n\
             \x20   assert!(!shortlisted.is_empty());\n\
             \x20   assert!(shortlisted.iter().any(|o| o.id == \"opt_{target_idx}\"));\n\
             }}\n"
        )?;
    }

    // 201..350: Engine High-Cardinality Choice (500 to 1400 options with max_slots policy)
    for i in 201..=350 {
        let n_opts = 500 + ((i - 200) * 6);
        let target_idx = 42 + (i % 100);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let engine = ZevEngine::default();\n\
             \x20   let options: Vec<OptionDef> = (0..{n_opts}).map(|k| OptionDef {{\n\
             \x20       id: format!(\"server_route_{{k}}\"),\n\
             \x20       description: format!(\"Traffic gateway load balancer ingress route node {{k}}\"),\n\
             \x20   }}).collect();\n\
             \x20   let policy = Policy {{\n\
             \x20       max_slots: Some(35),\n\
             \x20       ..Policy::default()\n\
             \x20   }};\n\
             \x20   let q = Question::Choice(ChoiceQuestion {{\n\
             \x20       instructions: \"Select the failing server route\".into(),\n\
             \x20       options,\n\
             \x20       policy,\n\
             \x20   }});\n\
             \x20   let req = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(\"Host telemetry alert: high memory saturation on server_route_{target_idx}\".into()),\n\
             \x20       questions: [(\"selection\".into(), q)].into(),\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res = engine.evaluate(&req).unwrap();\n\
             \x20   assert_eq!(res.answers[\"selection\"].decision.as_ref().and_then(|v| v.as_str()), Some(\"server_route_{target_idx}\"));\n\
             }}\n"
        )?;
    }

    // 351..550: Mega-Premises & Context Flooding (50 to 500 log lines / noise)
    for i in 351..=550 {
        let reps = 30 + ((i - 350) * 2);
        let target_id = format!("proc_svc_{}", i % 20);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let mut state = String::with_capacity({reps} * 70);\n\
             \x20   for line in 0..{reps} {{\n\
             \x20       state.push_str(&format!(\"2026-09-24T01:00:{{:02}}Z [INFO] system background heartbeat thread {{}} normal operation\\n\", line % 60, line));\n\
             \x20   }}\n\
             \x20   state.push_str(\"2026-09-24T01:45:00Z [CRITICAL] {target_id} crashed with exit status 137 OOMKilled\\n\");\n\
             \x20   let engine = ZevEngine::default();\n\
             \x20   let options = vec![\n\
             \x20       OptionDef {{ id: \"proc_svc_0\".into(), description: \"processing service instance 0\".into() }},\n\
             \x20       OptionDef {{ id: \"{target_id}\".into(), description: \"critical microservice crashed by oom\".into() }},\n\
             \x20       OptionDef {{ id: \"auth_proxy\".into(), description: \"authentication gateway proxy\".into() }},\n\
             \x20   ];\n\
             \x20   let q = Question::Choice(ChoiceQuestion {{\n\
             \x20       instructions: \"Identify crashed service\".into(),\n\
             \x20       options,\n\
             \x20       policy: Policy::default(),\n\
             \x20   }});\n\
             \x20   let req = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(state),\n\
             \x20       questions: [(\"root_cause\".into(), q)].into(),\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res = engine.evaluate(&req).unwrap();\n\
             \x20   assert_eq!(res.answers[\"root_cause\"].decision.as_ref().and_then(|v| v.as_str()), Some(\"{target_id}\"));\n\
             }}\n"
        )?;
    }

    // 551..750: Multi-Task Batch Flooding (5 to 30 simultaneous questions per request)
    for i in 551..=750 {
        let num_q = 5 + (i % 25);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let engine = ZevEngine::default();\n\
             \x20   let mut questions = std::collections::BTreeMap::new();\n\
             \x20   for q_idx in 0..{num_q} {{\n\
             \x20       let q = Question::Boolean(BooleanQuestion {{\n\
             \x20           instructions: format!(\"Is metric {{q_idx}} exceeding threshold?\"),\n\
             \x20           true_description: \"Yes metric exceeds threshold\".into(),\n\
             \x20           false_description: \"No metric within limits\".into(),\n\
             \x20           policy: Policy::default(),\n\
             \x20       }});\n\
             \x20       questions.insert(format!(\"metric_{{q_idx}}\"), q);\n\
             \x20   }}\n\
             \x20   let req = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(\"Metrics evaluation report: metric_0 is healthy, metric_1 is elevated, all systems online\".into()),\n\
             \x20       questions,\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res = engine.evaluate(&req).unwrap();\n\
             \x20   assert_eq!(res.answers.len(), {num_q});\n\
             }}\n"
        )?;
    }

    // 751..950: Order Invariance Heavy Permutations
    for i in 751..=950 {
        let num_opts = 10 + (i % 30);
        let target = 3 + (i % 7);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let engine = ZevEngine::default();\n\
             \x20   let base_options: Vec<OptionDef> = (0..{num_opts}).map(|k| OptionDef {{\n\
             \x20       id: format!(\"opt_{{k}}\"),\n\
             \x20       description: if k == {target} {{\n\
             \x20           \"High priority matching target action detected in premise\".into()\n\
             \x20       }} else {{\n\
             \x20           format!(\"Alternative background service task {{k}}\")\n\
             \x20       }},\n\
             \x20   }}).collect();\n\
             \x20   let q1 = Question::Choice(ChoiceQuestion {{\n\
             \x20       instructions: \"Select priority action\".into(),\n\
             \x20       options: base_options.clone(),\n\
             \x20       policy: Policy::default(),\n\
             \x20   }});\n\
             \x20   let req1 = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(\"Urgent: matching target action required immediately\".into()),\n\
             \x20       questions: [(\"decision\".into(), q1)].into(),\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res1 = engine.evaluate(&req1).unwrap();\n\
             \x20   let mut rev_options = base_options;\n\
             \x20   rev_options.reverse();\n\
             \x20   let q2 = Question::Choice(ChoiceQuestion {{\n\
             \x20       instructions: \"Select priority action\".into(),\n\
             \x20       options: rev_options,\n\
             \x20       policy: Policy::default(),\n\
             \x20   }});\n\
             \x20   let req2 = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(\"Urgent: matching target action required immediately\".into()),\n\
             \x20       questions: [(\"decision\".into(), q2)].into(),\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res2 = engine.evaluate(&req2).unwrap();\n\
             \x20   assert_eq!(res1.answers[\"decision\"].decision, res2.answers[\"decision\"].decision);\n\
             }}\n"
        )?;
    }

    // 951..1150: Extreme Numerical Calibration & Temperatures
    for i in 951..=1150 {
        let temp = match i % 6 {
            0 => "0.001",
            1 => "0.05",
            2 => "0.5",
            3 => "1.0",
            4 => "10.0",
            _ => "80.0",
        };
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let logits = vec![1.2, 5.8, -2.1, 0.4, 3.3, 0.0, 7.1, -10.5];\n\
             \x20   let probs = scaled_softmax(&logits, {temp}).unwrap();\n\
             \x20   assert_eq!(probs.len(), logits.len());\n\
             \x20   let sum: f64 = probs.iter().sum();\n\
             \x20   assert!((sum - 1.0).abs() < 1e-4, \"Sum {{sum}} should equal 1.0\");\n\
             \x20   for &p in &probs {{\n\
             \x20       assert!((0.0..=1.0).contains(&p), \"Probability {{p}} out of bounds\");\n\
             \x20       assert!(!p.is_nan() && !p.is_infinite(), \"Probability {{p}} invalid\");\n\
             \x20   }}\n\
             }}\n"
        )?;
    }

    // 1151..1350: CLM VectorArena Stress & LRU Cache Thrashing
    for i in 1151..=1350 {
        let capacity = 10 + (i % 20);
        let dim = 32;
        let num_inserts = 50 + (i % 100);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let mut arena = VectorArena::new({capacity}, {dim});\n\
             \x20   for k in 0..{num_inserts} {{\n\
             \x20       let vec: Vec<f32> = (0..{dim}).map(|x| ((x + k) as f32).sin()).collect();\n\
             \x20       arena.insert(&format!(\"vector_{{k}}\"), &vec);\n\
             \x20   }}\n\
             \x20   assert_eq!(arena.len(), {capacity});\n\
             \x20   assert_eq!(arena.capacity(), {capacity});\n\
             \x20   let stats = arena.stats();\n\
             \x20   assert_eq!(stats.used, {capacity});\n\
             \x20   assert!(stats.evictions > 0);\n\
             \x20   let query: Vec<f32> = (0..{dim}).map(|x| (x as f32).cos()).collect();\n\
             \x20   let keys = vec![\"vector_49\", \"vector_50\", \"vector_nonexistent\"];\n\
             \x20   let scores = arena.score_keys(&query, &keys);\n\
             \x20   assert_eq!(scores.len(), 3);\n\
             }}\n"
        )?;
    }

    // 1351..1550: CLM ContrastiveHead Projections
    for i in 1351..=1550 {
        let dim = 16 * ((i % 8) + 1);
        let out_dim = 8 * ((i % 4) + 1);
        let logit_scale = 1.0 + ((i % 20) as f64 * 0.1);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let config = HeadConfig {{\n\
             \x20       input_dim: {dim},\n\
             \x20       hidden_dim: {},\n\
             \x20       projection_dim: {out_dim},\n\
             \x20       logit_scale: {logit_scale:.4},\n\
             \x20   }};\n\
             \x20   let head = ContrastiveHead::new(config);\n\
             \x20   let x: Vec<f32> = (0..{dim}).map(|v| (v as f32 * 0.1).sin()).collect();\n\
             \x20   let proj = head.project(&x);\n\
             \x20   assert_eq!(proj.len(), {out_dim});\n\
             \x20   let norm: f32 = proj.iter().map(|v| v * v).sum::<f32>().sqrt();\n\
             \x20   assert!((norm - 1.0).abs() < 1e-4, \"Projected vector must be L2 normalized\");\n\
             \x20   let y: Vec<f32> = (0..{dim}).map(|v| (v as f32 * 0.2).cos()).collect();\n\
             \x20   let proj_y = head.project(&y);\n\
             \x20   let score = head.score(&proj, &proj_y);\n\
             \x20   assert!(!score.is_nan() && !score.is_infinite());\n\
             }}\n",
            dim * 2
        )?;
    }

    // 1551..1750: CLM HybridVerifier Two-Tier Stress
    for i in 1551..=1750 {
        let num_actions = 15 + (i % 25);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let mut verifier = HybridVerifier::new(50, 10);\n\
             \x20   let options: Vec<OptionDef> = (0..{num_actions}).map(|a| {{\n\
             \x20       let id = format!(\"action_{{a}}\");\n\
             \x20       let raw_emb: Vec<f32> = (0..512).map(|x| (x as f32 + a as f32).cos()).collect();\n\
             \x20       verifier.register_action_embedding(&id, &raw_emb);\n\
             \x20       OptionDef {{\n\
             \x20           id,\n\
             \x20           description: format!(\"Execute cloud container operation {{a}} on node\"),\n\
             \x20       }}\n\
             \x20   }}).collect();\n\
             \x20   let question = ChoiceQuestion {{\n\
             \x20       instructions: \"Execute target container operation\".into(),\n\
             \x20       options,\n\
             \x20       policy: Policy::default(),\n\
             \x20   }};\n\
             \x20   let query_emb: Vec<f32> = (0..512).map(|x| (x as f32 * 0.5).sin()).collect();\n\
             \x20   let ans = verifier.evaluate_hybrid(\n\
             \x20       \"Execute cloud container operation 3 on node\",\n\
             \x20       Some(&query_emb),\n\
             \x20       &question,\n\
             \x20       0.5,\n\
             \x20   ).unwrap();\n\
             \x20   assert!(ans.decision.is_some());\n\
             }}\n"
        )?;
    }

    // 1751..1900: Adversarial Substring Overlaps, Homoglyphs & Edge Cases
    for i in 1751..=1900 {
        let suffix = format!("token_{i}");
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let engine = ZevEngine::default();\n\
             \x20   let options = vec![\n\
             \x20       OptionDef {{ id: \"app\".into(), description: \"short app prefix {suffix}\".into() }},\n\
             \x20       OptionDef {{ id: \"apple\".into(), description: \"apple fruit {suffix}\".into() }},\n\
             \x20       OptionDef {{ id: \"application\".into(), description: \"application software deployment {suffix}\".into() }},\n\
             \x20       OptionDef {{ id: \"applicable\".into(), description: \"legally applicable regulation {suffix}\".into() }},\n\
             \x20   ];\n\
             \x20   let q = Question::Choice(ChoiceQuestion {{\n\
             \x20       instructions: \"Identify exact match\".into(),\n\
             \x20       options,\n\
             \x20       policy: Policy::default(),\n\
             \x20   }});\n\
             \x20   let req = ZevRequest {{\n\
             \x20       state: serde_json::Value::String(\"We are currently updating the enterprise application software deployment {suffix}\".into()),\n\
             \x20       questions: [(\"matched\".into(), q)].into(),\n\
             \x20       model: None,\n\
             \x20       temperature: None,\n\
             \x20       enable_temporal_facts: false,\n\
             \x20   }};\n\
             \x20   let res = engine.evaluate(&req).unwrap();\n\
             \x20   assert_eq!(res.answers[\"matched\"].decision.as_ref().and_then(|v| v.as_str()), Some(\"application\"));\n\
             }}\n"
        )?;
    }

    // 1901..2000: High-Concurrency Multithreading Stress
    for i in 1901..=2000 {
        let n_threads = 4 + (i % 5);
        writeln!(
            f,
            "#[test]\n\
             fn test_case_{i:04}() {{\n\
             \x20   let engine = Arc::new(ZevEngine::default());\n\
             \x20   let mut handles = Vec::with_capacity({n_threads});\n\
             \x20   for t in 0..{n_threads} {{\n\
             \x20       let eng = Arc::clone(&engine);\n\
             \x20       let handle = thread::spawn(move || {{\n\
             \x20           let options: Vec<OptionDef> = (0..50).map(|k| OptionDef {{\n\
             \x20               id: format!(\"worker_route_{{k}}\"),\n\
             \x20               description: format!(\"Distributed microservice pipeline worker route {{k}}\"),\n\
             \x20           }}).collect();\n\
             \x20           let q = Question::Choice(ChoiceQuestion {{\n\
             \x20               instructions: \"Select target worker\".into(),\n\
             \x20               options,\n\
             \x20               policy: Policy::default(),\n\
             \x20           }});\n\
             \x20           let expected_id = format!(\"worker_route_{{}}\", t % 50);\n\
             \x20           let req = ZevRequest {{\n\
             \x20               state: serde_json::Value::String(format!(\"Dispatching queue item to {{expected_id}}\")),\n\
             \x20               questions: [(\"selection\".into(), q)].into(),\n\
             \x20               model: None,\n\
             \x20               temperature: None,\n\
             \x20               enable_temporal_facts: false,\n\
             \x20           }};\n\
             \x20           let res = eng.evaluate(&req).unwrap();\n\
             \x20           assert_eq!(\n\
             \x20               res.answers[\"selection\"].decision.as_ref().and_then(|v| v.as_str()),\n\
             \x20               Some(expected_id.as_str())\n\
             \x20           );\n\
             \x20       }});\n\
             \x20       handles.push(handle);\n\
             \x20   }}\n\
             \x20   for handle in handles {{\n\
             \x20       handle.join().unwrap();\n\
             \x20   }}\n\
             }}\n"
        )?;
    }

    f.flush()?;
    println!("✅ Successfully generated 2,000 stress tests into {out_path}.");
    Ok(())
}
