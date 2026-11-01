// ============================================================================
// tests/chaos_tests.rs — Chaos, Fuzz, and Resilience Test Suite for zev-rs
// Stress tests subsystems under adversarial conditions, corrupt inputs, and races.
// ============================================================================

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use zev::{
    clean_text, compute_brier_score, compute_dataset_brier_score, compute_ece, detect_script,
    fast_walsh_hadamard_transform, jaro_winkler_similarity, make_safe_for_division,
    preprocess_state, revin_denormalize, revin_normalize, scaled_softmax,
    smooth_ordinal_probabilities, soundex_code, BooleanQuestion, ChoiceQuestion, DecisionCache,
    GateType, OptionDef, Policy, Question, ScoreQuestion, TopologicalCircuit, ZevEngine,
    ZevRequest,
};

// ============================================================================
// 1. Calibration & Softmax Chaos Tests
// ============================================================================

#[test]
fn test_chaos_scaled_softmax_extreme_temperatures_and_logits() {
    // Non-positive or non-finite temperature must return error without panicking
    assert!(scaled_softmax(&[1.0, 2.0], 0.0).is_err());
    assert!(scaled_softmax(&[1.0, 2.0], -1.0).is_err());
    assert!(scaled_softmax(&[1.0, 2.0], f64::NAN).is_err());
    assert!(scaled_softmax(&[1.0, 2.0], f64::INFINITY).is_err());
    assert!(scaled_softmax(&[], 1.0).is_err());

    // Ultra-tiny temperature (near zero, extreme sharpening): argmax should be ~1.0, others ~0.0
    let sharp = scaled_softmax(&[10.0, 20.0, 15.0], 1e-4).expect("sharp softmax");
    assert!(sharp[1] > 0.9999);
    assert!(sharp[0] < 1e-4);
    assert!(sharp[2] < 1e-4);
    let sum_sharp: f64 = sharp.iter().sum();
    assert!((sum_sharp - 1.0).abs() < 1e-9);

    // Massive temperature: uniform distribution across all categories
    let flat = scaled_softmax(&[10.0, 20.0, 15.0], 1e6).expect("flat softmax");
    for p in &flat {
        assert!((p - 0.3333333333333333).abs() < 1e-4);
    }

    // Extreme dynamic range of logits: [1e15, -1e15, 0.0]
    let extreme = scaled_softmax(&[1e15, -1e15, 0.0], 1.0).expect("extreme dynamic range");
    assert_eq!(extreme[0], 1.0);
    assert_eq!(extreme[1], 0.0);
    assert_eq!(extreme[2], 0.0);

    // Identical logits: exactly uniform
    let uniform = scaled_softmax(&[42.0, 42.0, 42.0, 42.0], 1.0).expect("identical logits");
    for p in &uniform {
        assert!((p - 0.25).abs() < 1e-9);
    }
}

#[test]
fn test_chaos_brier_and_ece_adversarial_distributions() {
    // ECE on empty inputs
    let empty_ece = compute_ece(&[], &[], 10);
    assert_eq!(empty_ece, 0.0);

    // ECE on mismatched lengths
    let mismatched_ece = compute_ece(&[0.9, 0.8], &[true], 10);
    assert_eq!(mismatched_ece, 0.0);

    // Perfectly calibrated predictions
    let confidences = vec![1.0, 1.0, 0.0, 0.0];
    let true_labels = vec![true, true, false, false];
    let perfect_ece = compute_ece(&confidences, &true_labels, 5);
    assert_eq!(perfect_ece, 0.0);

    // Brier score on unnormalized probabilities
    let unnormalized_probs = vec![0.0, 0.0, 0.0];
    let score = compute_brier_score(&unnormalized_probs, 1);
    assert!(score.is_finite());

    // Dataset Brier score on empty set
    let dataset_brier = compute_dataset_brier_score(&[]);
    assert_eq!(dataset_brier, 0.0);
}

// ============================================================================
// 2. Ordinal Smoothing Chaos Tests
// ============================================================================

#[test]
fn test_chaos_ordinal_smoothing_adversarial_inputs() {
    // Empty probabilities
    assert_eq!(smooth_ordinal_probabilities(&[], 0.15), Vec::<f64>::new());

    // Single category
    assert_eq!(smooth_ordinal_probabilities(&[1.0], 0.25), vec![1.0]);

    // Zero or negative alpha (identity)
    assert_eq!(
        smooth_ordinal_probabilities(&[0.3, 0.7], 0.0),
        vec![0.3, 0.7]
    );
    assert_eq!(
        smooth_ordinal_probabilities(&[0.3, 0.7], -0.5),
        vec![0.3, 0.7]
    );

    // Alpha >= 0.5 is clamped to 0.49
    let clamped = smooth_ordinal_probabilities(&[0.0, 1.0, 0.0], 0.99);
    let sum_clamped: f64 = clamped.iter().sum();
    assert!((sum_clamped - 1.0).abs() < 1e-9);

    // Endpoint spike at 0
    let b0 = smooth_ordinal_probabilities(&[1.0, 0.0, 0.0, 0.0], 0.20);
    assert!((b0.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(b0[0], 0.80);
    assert_eq!(b0[1], 0.20);
    assert_eq!(b0[2], 0.0);
    assert_eq!(b0[3], 0.0);

    // Endpoint spike at K-1
    let b_end = smooth_ordinal_probabilities(&[0.0, 0.0, 0.0, 1.0], 0.20);
    assert!((b_end.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(b_end[3], 0.80);
    assert_eq!(b_end[2], 0.20);
    assert_eq!(b_end[0], 0.0);

    // 500 categories under smoothing
    let mut large = vec![0.0; 500];
    large[250] = 1.0;
    let smoothed_large = smooth_ordinal_probabilities(&large, 0.15);
    let sum_large: f64 = smoothed_large.iter().sum();
    assert!((sum_large - 1.0).abs() < 1e-7);
}

// ============================================================================
// 3. Fast Walsh-Hadamard Transform (FWHT) Chaos Tests
// ============================================================================

#[test]
fn test_chaos_fast_walsh_hadamard_transform_invariants() {
    // 0 and 1 length vectors
    let mut empty: Vec<f32> = vec![];
    fast_walsh_hadamard_transform(&mut empty);
    assert!(empty.is_empty());

    let mut single = vec![42.0f32];
    fast_walsh_hadamard_transform(&mut single);
    assert_eq!(single[0], 42.0);

    // Power of two lengths: 2, 4, 8, 16, 64, 256
    for power in [1, 2, 3, 4, 6, 8] {
        let n = 1 << power;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 5.0).collect();
        let original = data.clone();

        // 1. First transform
        fast_walsh_hadamard_transform(&mut data);

        // Energy conservation (Parseval's identity for orthonormal transform)
        let energy_orig: f32 = original.iter().map(|&x| x * x).sum();
        let energy_trans: f32 = data.iter().map(|&x| x * x).sum();
        assert!((energy_orig - energy_trans).abs() < 1e-3);

        // 2. Second transform recovers original vector (involution: H * H = I)
        fast_walsh_hadamard_transform(&mut data);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}

#[test]
#[should_panic(expected = "FWHT length must be a power of two")]
fn test_chaos_fwht_rejects_non_power_of_two() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // len = 5
    fast_walsh_hadamard_transform(&mut data);
}

// ============================================================================
// 4. Decision Cache Multi-threaded Chaos Tests
// ============================================================================

#[test]
fn test_chaos_decision_cache_concurrent_hammering() {
    let cache = Arc::new(DecisionCache::<u64, String>::new(
        100,
        Some(Duration::from_millis(50)),
    ));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // 8 concurrent writer threads
    for t in 0..8 {
        let cache_clone = Arc::clone(&cache);
        let stop_clone = Arc::clone(&stop_flag);
        handles.push(std::thread::spawn(move || {
            let mut iter = 0;
            while !stop_clone.load(Ordering::Relaxed) && iter < 1000 {
                let key = (iter % 200) as u64;
                cache_clone.insert(key, format!("value_from_thread_{}_{}", t, iter));
                iter += 1;
            }
        }));
    }

    // 8 concurrent reader threads
    for _ in 0..8 {
        let cache_clone = Arc::clone(&cache);
        let stop_clone = Arc::clone(&stop_flag);
        handles.push(std::thread::spawn(move || {
            let mut reads = 0;
            while !stop_clone.load(Ordering::Relaxed) && reads < 1000 {
                let key = (reads % 200) as u64;
                let _ = cache_clone.get(&key);
                reads += 1;
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }

    // Assert cache bounds were respected (eviction works under pressure)
    assert!(cache.len() <= 100);
    assert!(cache.hits() + cache.misses() > 0);
}

// ============================================================================
// 5. Preprocessor & Unicode Chaos Tests
// ============================================================================

#[test]
fn test_chaos_preprocessor_adversarial_unicode_and_bombardment() {
    let giant_token = "A".repeat(50_000);
    let adversarial_strings = [
        "",            // Empty
        "   \t\n\r  ", // Whitespace only
        "Null byte \0 in string",
        "\x1b[31;1mRed text\x1b[0m and \x1b[2J clear screen", // ANSI escape codes
        "Unicode RTL override \u{202E}gnirts detrevni\u{202C}", // Right-to-left
        "Surrogate halves and emojis: 🚀🔥🍎👨‍👩‍👧‍👦",
        giant_token.as_str(), // Giant continuous unspaced token
        "Email chain:\n> quote 1\n>> quote 2\n>>> quote 3\n>>>> quote 4\nOriginal text",
    ];

    for s in adversarial_strings {
        let cleaned = clean_text(s);
        // Clean text should never panic and produce valid string slice
        assert!(cleaned.len() <= s.len());

        let preprocessed = preprocess_state(s, false);
        let preprocessed_temporal = preprocess_state(s, true);
        assert!(!preprocessed.is_empty() || s.trim().is_empty());
        assert!(!preprocessed_temporal.is_empty() || s.trim().is_empty());

        // Script detection should classify safely
        let script = detect_script(s);
        let _ = format!("{:?}", script);
    }
}

// ============================================================================
// 6. Series Guardrails & Numeric Chaos Tests
// ============================================================================

#[test]
fn test_chaos_series_guardrails_numeric_boundaries() {
    // make_safe_for_division clamps magnitudes < 1e-6 to +/- 1e-6
    assert_eq!(make_safe_for_division(0.0), 1e-6);
    assert_eq!(make_safe_for_division(-0.0), 1e-6);
    assert_eq!(make_safe_for_division(1e-20), 1e-6);
    assert_eq!(make_safe_for_division(-1e-5), -1e-5);
    assert_eq!(make_safe_for_division(100.0), 100.0);
    assert_eq!(make_safe_for_division(-50.0), -50.0);

    // revin_normalize on zero variance (constant array)
    let constant_series = vec![42.0, 42.0, 42.0, 42.0];
    let (norm, mean, std) = revin_normalize(&constant_series);
    assert_eq!(mean, 42.0);
    assert!(
        std > 0.0,
        "std must be safe for division even when variance is zero"
    );
    for x in &norm {
        assert!((x - 0.0).abs() < 1e-6);
    }

    // Denormalization exact roundtrip
    let denorm = revin_denormalize(&norm, mean, std);
    for (a, b) in constant_series.iter().zip(denorm.iter()) {
        assert!((a - b).abs() < 1e-4);
    }

    // Empty series normalization
    let empty: Vec<f64> = vec![];
    let (empty_norm, _, _) = revin_normalize(&empty);
    assert!(empty_norm.is_empty());
}

// ============================================================================
// 7. Topological Circuit Chaos Tests
// ============================================================================

#[test]
fn test_chaos_topological_circuit_cycles_and_invalid_wires() {
    let mut circuit = TopologicalCircuit::new();
    circuit.num_wires = 3;
    circuit.input_wires = vec![0];
    circuit.output_wires = vec![2];

    // Create a circular dependency: Gate 0 output = wire 1, Gate 1 output = wire 0 (cycle!)
    circuit.add_gate(GateType::Not, &[0], 1);
    circuit.add_gate(GateType::Not, &[1], 0);

    // Compilation must detect the cycle and return an error rather than hanging in infinite loop
    let compile_result = circuit.compile();
    assert!(
        compile_result.is_err(),
        "Cyclic circuit must be rejected by compiler"
    );
}

// ============================================================================
// 8. Fuzzy Match Chaos Tests
// ============================================================================

#[test]
fn test_chaos_fuzzy_match_pathological_inputs() {
    // Both empty: 1.0
    assert_eq!(jaro_winkler_similarity("", ""), 1.0);
    // One empty: 0.0
    assert_eq!(jaro_winkler_similarity("", "abc"), 0.0);
    assert_eq!(jaro_winkler_similarity("xyz", ""), 0.0);

    // Extremely long strings (5,000 chars)
    let s1 = "a".repeat(5000);
    let s2 = "a".repeat(5000);
    assert_eq!(jaro_winkler_similarity(&s1, &s2), 1.0);

    // Dissimilar long strings
    let s3 = "b".repeat(5000);
    assert_eq!(jaro_winkler_similarity(&s1, &s3), 0.0);

    // Emojis and unicode strings
    let sim = jaro_winkler_similarity("🚀 rocket", "🚀 rocketeer");
    assert!(sim > 0.8 && sim <= 1.0);

    // Soundex on non-alphabetic strings returns "0000" padding
    assert_eq!(soundex_code(""), "0000");
    assert_eq!(soundex_code("12345"), "0000");
    assert_eq!(soundex_code("!@#$%"), "0000");

    // Valid soundex length is always 4 chars when text has letters
    let sx = soundex_code("Washington");
    assert_eq!(sx.len(), 4);
}

// ============================================================================
// 9. ZevEngine End-to-End Decision Chaos Tests
// ============================================================================

#[test]
fn test_chaos_zev_engine_adversarial_request_evaluation() {
    let engine = ZevEngine::default();

    // 1. Question with 50 candidates
    let mut options = Vec::new();
    for i in 0..50 {
        options.push(OptionDef {
            id: format!("opt_{}", i),
            description: format!("Description for option index {} with detail", i),
        });
    }

    let mut questions = BTreeMap::new();
    questions.insert(
        "large_choice".into(),
        Question::Choice(ChoiceQuestion {
            instructions: "Select the most appropriate option from 50 candidates".into(),
            options,
            policy: Policy::default(),
        }),
    );

    // 2. Boolean question with inverted descriptions
    questions.insert(
        "inverted_bool".into(),
        Question::Boolean(BooleanQuestion {
            instructions: "Evaluate system health".into(),
            true_description: "".into(), // Empty true description
            false_description: "False description".into(),
            policy: Policy::default(),
        }),
    );

    // 3. Score question with 10 ordinal levels
    let levels: Vec<String> = (0..10).map(|i| format!("Level {} severity", i)).collect();
    questions.insert(
        "scale_score".into(),
        Question::Score(ScoreQuestion {
            instructions: "Rate severity on 10 levels".into(),
            levels,
            ordinal_smoothing: Some(0.15),
            policy: Policy::default(),
        }),
    );

    let state_text =
        "System is operating normally with minor background CPU spikes of 15%.".repeat(50);
    let req = ZevRequest {
        state: serde_json::json!(state_text),
        questions,
        model: None,
        temperature: Some(0.8),
        enable_temporal_facts: false,
        images: None,
    };

    let response = engine
        .evaluate(&req)
        .expect("engine evaluate must survive chaos inputs");

    assert!(response.answers.contains_key("large_choice"));
    assert!(response.answers.contains_key("inverted_bool"));
    assert!(response.answers.contains_key("scale_score"));

    for (k, ans) in &response.answers {
        assert!(
            ans.confidence >= 0.0 && ans.confidence <= 1.0,
            "Confidence for {} out of range",
            k
        );
        assert!(
            ans.temperature > 0.0,
            "Temperature for {} must be positive",
            k
        );
    }
}
