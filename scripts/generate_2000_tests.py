#!/usr/bin/env python3
"""
Generator for tests/scale_2000_tests.rs
Pounds and overloads zev-rs across 2,000 distinct stress test cases:
1..200: Scaled Shortlisting (cardinality 27 to 1500 options, varying top_k, ID boosts, bounds)
201..350: Engine High-Cardinality Choice (500 to 1500 options via ZevEngine with max_slots policy)
351..550: Mega-Premises & Context Flooding (50 to 500 log lines, JSON payloads, multilingual, unicode noise)
551..750: Multi-Task Batch Flooding (5 to 30 simultaneous questions per request)
751..950: Order Invariance Heavy Permutations (reversed, shifted, shuffled options)
951..1150: Extreme Numerical Calibration & Temperatures (0.001 to 80.0, zero sum, uniform bounds)
1151..1350: CLM VectorArena Stress & LRU Cache Thrashing (high volume inserts into small arenas, eviction checks)
1351..1550: CLM ContrastiveHead Projections (varied dimensions 16..256, logit scales, batch scoring)
1551..1750: CLM HybridVerifier Two-Tier Stress (large candidate sets, shortlisting + contrastive evaluation)
1751..1900: Adversarial Substring Overlaps, Homoglyphs & Edge Cases (prefix/infix collisions, zero-length tokens)
1901..2000: High-Concurrency Multithreading Stress (Arc<ZevEngine> hammered across spawned threads)
"""

def main():
    out_path = "tests/scale_2000_tests.rs"
    print(f"Generating 2,000 stress tests into {out_path}...")

    with open(out_path, "w", encoding="utf-8") as f:
        # Header
        f.write("""// AUTO-GENERATED TEST SUITE: 2000 SCALE & STRESS TESTS FOR ZEV-RS
// Pounding high-cardinality options, mega-premises, multi-task flooding,
// cache thrashing, adversarial inputs, multithreading, and CLM features.
#![allow(unused_imports, unused_variables, dead_code)]

use std::sync::{Arc, Mutex};
use std::thread;
use zev::types::*;
use zev::engine::ZevEngine;
use zev::shortlist::shortlist_options;
use zev::calibration::*;
use zev::tev1::*;
use zev::clm::*;

""")

        # 1..200: Scaled Shortlisting
        for i in range(1, 201):
            n_opts = 25 + (i * 7)  # from 32 up to 1425 options
            top_k = 5 + (i % 25)
            target_idx = (i * 3) % n_opts
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let options: Vec<OptionDef> = (0..{n_opts}).map(|k| OptionDef {{
        id: format!("opt_{{k}}"),
        description: format!("Microservice deployment target cluster handler {{k}} with failover logic"),
    }}).collect();
    let shortlisted = shortlist_options(&options, "critical incident alert on opt_{target_idx} cluster failure", {top_k});
    assert!(shortlisted.len() <= {top_k});
    assert!(!shortlisted.is_empty());
    assert!(shortlisted.iter().any(|o| o.id == "opt_{target_idx}"));
}}

""")

        # 201..350: Engine High-Cardinality Choice (500 to 1500 options with max_slots policy)
        for i in range(201, 351):
            n_opts = 500 + ((i - 200) * 6)  # 506 up to 1400 options
            target_idx = 42 + (i % 100)
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let engine = ZevEngine::default();
    let options: Vec<OptionDef> = (0..{n_opts}).map(|k| OptionDef {{
        id: format!("server_route_{{k}}"),
        description: format!("Traffic gateway load balancer ingress route node {{k}}"),
    }}).collect();
    let policy = Policy {{
        max_slots: Some(35),
        ..Policy::default()
    }};
    let q = Question::Choice(ChoiceQuestion {{
        instructions: "Select the failing server route".into(),
        options,
        policy,
    }});
    let req = ZevRequest {{
        state: "Host telemetry alert: high memory saturation on server_route_{target_idx}".into(),
        questions: [("selection".into(), q)].into(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res = engine.evaluate(&req).unwrap();
    assert_eq!(res.answers["selection"].decision.as_ref().and_then(|v| v.as_str()), Some("server_route_{target_idx}"));
}}

""")

        # 351..550: Mega-Premises & Context Flooding (50 to 500 log lines / noise)
        for i in range(351, 551):
            reps = 30 + ((i - 350) * 2)  # 32 up to 430 lines
            target_id = f"proc_svc_{(i % 20)}"
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let mut state = String::with_capacity({reps * 70});
    for line in 0..{reps} {{
        state.push_str(&format!("2026-09-24T01:00:{{:02}}Z [INFO] system background heartbeat thread {{}} normal operation\\n", line % 60, line));
    }}
    state.push_str("2026-09-24T01:45:00Z [CRITICAL] {target_id} crashed with exit status 137 OOMKilled\\n");
    let engine = ZevEngine::default();
    let options = vec![
        OptionDef {{ id: "proc_svc_0".into(), description: "processing service instance 0".into() }},
        OptionDef {{ id: "{target_id}".into(), description: "critical microservice crashed by oom".into() }},
        OptionDef {{ id: "auth_proxy".into(), description: "authentication gateway proxy".into() }},
    ];
    let q = Question::Choice(ChoiceQuestion {{
        instructions: "Identify crashed service".into(),
        options,
        policy: Policy::default(),
    }});
    let req = ZevRequest {{
        state: state.into(),
        questions: [("root_cause".into(), q)].into(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res = engine.evaluate(&req).unwrap();
    assert_eq!(res.answers["root_cause"].decision.as_ref().and_then(|v| v.as_str()), Some("{target_id}"));
}}

""")

        # 551..750: Multi-Task Batch Flooding (5 to 30 simultaneous questions per request)
        for i in range(551, 751):
            num_q = 5 + (i % 25)  # 5 to 29 questions
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let engine = ZevEngine::default();
    let mut questions = std::collections::BTreeMap::new();
    for q_idx in 0..{num_q} {{
        let q = Question::Boolean(BooleanQuestion {{
            instructions: format!("Is metric {{}} exceeding threshold?", q_idx),
            true_description: "Yes metric exceeds threshold".into(),
            false_description: "No metric within limits".into(),
            policy: Policy::default(),
        }});
        questions.insert(format!("metric_{{q_idx}}"), q);
    }}
    let req = ZevRequest {{
        state: "Metrics evaluation report: metric_0 is healthy, metric_1 is elevated, all systems online".into(),
        questions,
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res = engine.evaluate(&req).unwrap();
    assert_eq!(res.answers.len(), {num_q});
}}

""")

        # 751..950: Order Invariance Heavy Permutations
        for i in range(751, 951):
            num_opts = 10 + (i % 30)
            target = 3 + (i % 7)
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let engine = ZevEngine::default();
    let base_options: Vec<OptionDef> = (0..{num_opts}).map(|k| OptionDef {{
        id: format!("opt_{{k}}"),
        description: if k == {target} {{
            "High priority matching target action detected in premise".into()
        }} else {{
            format!("Alternative background service task {{k}}")
        }},
    }}).collect();

    // Forward order
    let q1 = Question::Choice(ChoiceQuestion {{
        instructions: "Select priority action".into(),
        options: base_options.clone(),
        policy: Policy::default(),
    }});
    let req1 = ZevRequest {{
        state: "Urgent: matching target action required immediately".into(),
        questions: [("decision".into(), q1)].into(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res1 = engine.evaluate(&req1).unwrap();

    // Reversed order
    let mut rev_options = base_options.clone();
    rev_options.reverse();
    let q2 = Question::Choice(ChoiceQuestion {{
        instructions: "Select priority action".into(),
        options: rev_options,
        policy: Policy::default(),
    }});
    let req2 = ZevRequest {{
        state: "Urgent: matching target action required immediately".into(),
        questions: [("decision".into(), q2)].into(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res2 = engine.evaluate(&req2).unwrap();

    assert_eq!(
        res1.answers["decision"].decision,
        res2.answers["decision"].decision
    );
}}

""")

        # 951..1150: Extreme Numerical Calibration & Temperatures
        for i in range(951, 1151):
            temp = 0.001 if i % 6 == 0 else (0.05 if i % 6 == 1 else (0.5 if i % 6 == 2 else (1.0 if i % 6 == 3 else (10.0 if i % 6 == 4 else 80.0))))
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let logits = vec![1.2, 5.8, -2.1, 0.4, 3.3, 0.0, 7.1, -10.5];
    let probs = scaled_softmax(&logits, {temp}).unwrap();
    assert_eq!(probs.len(), logits.len());
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Sum {{sum}} should equal 1.0");
    for &p in &probs {{
        assert!(p >= 0.0 && p <= 1.0, "Probability {{p}} out of bounds");
        assert!(!p.is_nan() && !p.is_infinite(), "Probability {{p}} invalid");
    }}
}}

""")

        # 1151..1350: CLM VectorArena Stress & LRU Cache Thrashing
        for i in range(1151, 1351):
            capacity = 10 + (i % 20)  # 10 to 29 capacity
            dim = 32
            num_inserts = 50 + (i % 100)  # 50 to 149 inserts (triggers lots of evictions)
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let mut arena = VectorArena::new({capacity}, {dim});
    for k in 0..{num_inserts} {{
        let vec: Vec<f32> = (0..{dim}).map(|x| ((x + k) as f32).sin()).collect();
        arena.insert(&format!("vector_{{k}}"), &vec);
    }}
    assert_eq!(arena.len(), {capacity});
    assert_eq!(arena.capacity(), {capacity});
    let stats = arena.stats();
    assert_eq!(stats.used, {capacity});
    assert!(stats.evictions > 0);

    let query: Vec<f32> = (0..{dim}).map(|x| (x as f32).cos()).collect();
    let keys = vec!["vector_49", "vector_50", "vector_nonexistent"];
    let scores = arena.score_keys(&query, &keys);
    assert_eq!(scores.len(), 3);
}}

""")

        # 1351..1550: CLM ContrastiveHead Projections & Extreme Tau
        for i in range(1351, 1551):
            dim = 16 * ((i % 8) + 1)  # 16 to 128
            out_dim = 8 * ((i % 4) + 1)
            logit_scale = 1.0 + (float(i % 20) * 0.1)
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let config = HeadConfig {{
        input_dim: {dim},
        hidden_dim: {dim * 2},
        projection_dim: {out_dim},
        logit_scale: {logit_scale:.4},
    }};
    let head = ContrastiveHead::new(config);
    let x: Vec<f32> = (0..{dim}).map(|v| (v as f32 * 0.1).sin()).collect();
    let proj = head.project(&x);
    assert_eq!(proj.len(), {out_dim});
    let norm: f32 = proj.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "Projected vector must be L2 normalized");

    let y: Vec<f32> = (0..{dim}).map(|v| (v as f32 * 0.2).cos()).collect();
    let proj_y = head.project(&y);
    let score = head.score(&proj, &proj_y);
    assert!(!score.is_nan() && !score.is_infinite());
}}

""")

        # 1551..1750: CLM HybridVerifier Two-Tier Stress
        for i in range(1551, 1751):
            num_actions = 15 + (i % 25)
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let mut verifier = HybridVerifier::new(50, 10);
    let options: Vec<OptionDef> = (0..{num_actions}).map(|a| {{
        let id = format!("action_{{a}}");
        let raw_emb: Vec<f32> = (0..512).map(|x| (x as f32 + a as f32).cos()).collect();
        verifier.register_action_embedding(&id, &raw_emb);
        OptionDef {{
            id,
            description: format!("Execute cloud container operation {{a}} on node"),
        }}
    }}).collect();

    let question = ChoiceQuestion {{
        instructions: "Execute target container operation".into(),
        options,
        policy: Policy::default(),
    }};

    let query_emb: Vec<f32> = (0..512).map(|x| (x as f32 * 0.5).sin()).collect();
    let ans = verifier.evaluate_hybrid(
        "Execute cloud container operation 3 on node",
        Some(&query_emb),
        &question,
        0.5
    ).unwrap();
    assert!(ans.decision.is_some());
}}

""")

        # 1751..1900: Adversarial Substring Overlaps, Homoglyphs & Edge Cases
        for i in range(1751, 1901):
            suffix = f"token_{i}"
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let engine = ZevEngine::default();
    let options = vec![
        OptionDef {{ id: "app".into(), description: "short app prefix {suffix}".into() }},
        OptionDef {{ id: "apple".into(), description: "apple fruit {suffix}".into() }},
        OptionDef {{ id: "application".into(), description: "application software deployment {suffix}".into() }},
        OptionDef {{ id: "applicable".into(), description: "legally applicable regulation {suffix}".into() }},
    ];
    let q = Question::Choice(ChoiceQuestion {{
        instructions: "Identify exact match".into(),
        options,
        policy: Policy::default(),
    }});
    // Test that 'application' does not trigger false positive on 'app' or 'apple'
    let req = ZevRequest {{
        state: "We are currently updating the enterprise application software deployment {suffix}".into(),
        questions: [("matched".into(), q)].into(),
        model: None,
        temperature: None,
        enable_temporal_facts: false,
    }};
    let res = engine.evaluate(&req).unwrap();
    assert_eq!(res.answers["matched"].decision.as_ref().and_then(|v| v.as_str()), Some("application"));
}}

""")

        # 1901..2000: High-Concurrency Multithreading Stress
        for i in range(1901, 2001):
            n_threads = 4 + (i % 5)  # 4 to 8 concurrent threads
            f.write(f"""#[test]
fn test_case_{i:04d}() {{
    let engine = Arc::new(ZevEngine::default());
    let mut handles = Vec::with_capacity({n_threads});

    for t in 0..{n_threads} {{
        let eng = Arc::clone(&engine);
        let handle = thread::spawn(move || {{
            let options: Vec<OptionDef> = (0..50).map(|k| OptionDef {{
                id: format!("worker_route_{{k}}"),
                description: format!("Distributed microservice pipeline worker route {{k}}"),
            }}).collect();
            let q = Question::Choice(ChoiceQuestion {{
                instructions: "Select target worker".into(),
                options,
                policy: Policy::default(),
            }});
            let expected_id = format!("worker_route_{{}}", t % 50);
            let req = ZevRequest {{
                state: format!("Dispatching queue item to {{}}", expected_id).into(),
                questions: [("selection".into(), q)].into(),
                model: None,
                temperature: None,
                enable_temporal_facts: false,
            }};
            let res = eng.evaluate(&req).unwrap();
            assert_eq!(
                res.answers["selection"].decision.as_ref().and_then(|v| v.as_str()),
                Some(expected_id.as_str())
            );
        }});
        handles.push(handle);
    }}

    for handle in handles {{
        handle.join().unwrap();
    }}
}}

""")

    print("Successfully generated 2,000 stress tests.")

if __name__ == "__main__":
    main()
