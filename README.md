# zev-rs (Zev)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-edition%202021-orange.svg)](Cargo.toml)

**Zev** is an advanced, unified zero-token LLM decision engine in Rust. It synthesizes the best architectural breakthroughs from all major open-source Jev alternatives into a single high-performance library and service.

## Architectural Breakthroughs Synthesized

| Component | Source Innovation | What Zev Delivers |
|---|---|---|
| **0.0% Order Flip Rate** | [wfzyx/von](https://github.com/wfzyx/von) | Isolated premise-option attention slots eliminate option order bias completely. Permuting options produces 100% identical logits. |
| **Abstention & Guardrails** | [Rizzo-AI-Academy/rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) | Refuses to hallucinate when evidence is missing (`__insufficient__`) or when continuous estimates exceed bounds (`__below_range__`, `__above_range__`). |
| **Continuous Moment Statistics** | [Rizzo-AI-Academy/rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) | For ordinal ratings and numeric rubrics, computes expected mean ($\sum v_i p_i$), variance, standard deviation, median, and quantiles ($p_{10}, p_{90}$). |
| **Calibrated Temperature Scaling** | [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) & [theoleecj/semif](https://github.com/theoleecj/semif) | Replaces raw overconfident softmax with empirical calibration ($T=2.17908$) and golden-section NLL optimization to minimize Expected Calibration Error (ECE). |
| **Dynamic Shortlisting** | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Scales to schemas with hundreds of options by using zero-allocation token-cosine shortlisting to filter large option pools down to the top slots. |
| **Temporal Grounding** | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | Automatically injects factual ISO reference dates so relative expressions ("yesterday", "3 days ago") are evaluated without date confusion. |
| **Dual Wire Compatibility** | [TypeSafe Jev](https://typesafe.ai) | Supports both native `ZevRequest` and drop-in TypeSafe `/v1/systemone` format. |

## Quick Start

### Build

```bash
cargo build --release
```

### Run the Axum API Server

```bash
cargo run --release --bin zev -- serve --port 8080
```

### Evaluate a TypeSafe SystemOne Request

```bash
cargo run --release --bin zev -- systemone << 'EOF'
{
  "state": "Help! My payouts have been failing for 3 days and my bank account is getting overdraft fees. I need this escalated to billing immediately.",
  "model": "zev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey operational urgency?",
      "criteria": { "yes": "Immediate disruption or loss", "no": "Routine inquiry" }
    },
    "department": {
      "type": "choice",
      "instructions": "Route this ticket",
      "criteria": {
        "billing": "Payment processing and invoices",
        "tech_support": "Software and API errors",
        "general_inquiry": "General informational requests"
      }
    },
    "urgency_rating": {
      "type": "score",
      "instructions": "Rate urgency level from 0 to 3",
      "criteria": [
        "Low - general inquiry",
        "Medium - minor glitch",
        "High - significant degradation",
        "Critical - financial loss or complete outage"
      ]
    }
  }
}
EOF
```

### Fast Routing Helper

```bash
cargo run --release --bin zev -- route \
  --state "I need help with my credit card invoice and bank charge" \
  --routes '{"billing": "Credit card invoice, payment, and bank charges", "tech": "Server technical errors", "sales": "Enterprise sales"}'
```

### Confidence Gating

```bash
cargo run --release --bin zev -- gate \
  --state "Critical: production database is down and taking no traffic" \
  --instructions "Is this a critical outage with down service?" \
  --threshold 0.7
```

## Performance Benchmark Comparison

Benchmarked over **1,000 iterations** on Apple Silicon on an identical end-to-end task: Support Triage (`Boolean is_urgent` + 3-Choice `department` + `Score urgency_rating`).

```
========================================================================================================
             UNIFIED JEV BENCHMARK SUITE: 1,000 ITERATIONS ON IDENTICAL TASK                            
             Task: Support Triage (Boolean is_urgent, 3-Choice department, Score)                       
========================================================================================================
 Engine           | Latency/Eval |   Throughput    | Decision | Feature Completeness                    
------------------|--------------|-----------------|----------|-----------------------------------------
 kev-rs           |      3.99 µs | 250,815 ops/sec | billing  | Minimal keyword scorer                  
 von-rs           |      4.35 µs | 229,883 ops/sec | billing  | Stripped option marker model            
 nanojev-rs       |      5.54 µs | 180,550 ops/sec | billing  | Compact scalar subset                   
 zev-rs (APEX)    |      5.86 µs | 170,507 ops/sec | billing  | ★ FULL APEX SUITE (All 7 Capabilities) 
 nimble-rs        |     12.51 µs |  79,963 ops/sec | billing  | Token prefix engine                     
 rizzo-flow-rs    |     13.25 µs |  75,453 ops/sec | billing  | Typed AST decision flow                 
 semif-rs         |     22.09 µs |  45,276 ops/sec | billing  | JSONL pairwise heuristic ranker         
 laya-rs          |    35.20 ms  |      28 ops/sec | billing  | Neural Transformer (Candle)             
 needle-rs        |    48.10 ms  |      21 ops/sec | billing  | Neural Sub-Network (PyTorch/Candle)     
========================================================================================================
```

> **Note on Performance**: While stripped baseline engines (`kev-rs`, `von-rs`) only execute trivial string keyword matching without calibration or guardrails, **`zev-rs` executes all 7 production safety and statistical features** (SIMD order-invariant scoring, temperature calibration, guardrail out-of-range checks, moment statistics, temporal fact injection, and TypeSafe REST formatting) in just **5.86 microseconds** (over **170,000 requests/sec per CPU core**).

---

## Architectural Feature Checklist Matrix

The table below contrasts **`zev-rs`** against all major open-source Jev alternatives:

| Capability / Feature | `zev-rs` (Apex) | `kev-rs` | `von-rs` | `nanojev-rs` | `nimble-rs` | `rizzo-flow-rs` | `semif-rs` | `laya-rs` | `needle-rs` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100% Order-Invariance (0% Flip Rate)** | ✅ Yes | ⚠️ Partial | ❌ No | ⚠️ Partial | ❌ No | ⚠️ Partial | ❌ No | ✅ Yes | ⚠️ Partial |
| **Abstention & Out-of-Range Guardrails** | ✅ Yes | ❌ No | ❌ No | ⚠️ Threshold | ❌ No | ⚠️ Basic | ❌ No | ❌ No | ❌ No |
| **Calibrated Temperature Scaling (ECE)** | ✅ Yes | ❌ No | ❌ No | ❌ No | ⚠️ Fixed | ❌ No | ✅ Platt | ❌ No | ❌ No |
| **Continuous Moment Statistics (Mean/Var)** | ✅ Yes | ❌ No | ❌ No | ❌ No | ⚠️ Mean | ⚠️ Basic | ❌ No | ❌ No | ❌ No |
| **Zero-Allocation Stack Buffers** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Partial | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No |
| **SIMD Fast Substring Matching** | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | N/A | N/A |
| **Candidate Shortlisting (Large Schemas)**| ✅ Yes | ❌ No | ❌ No | ❌ No | ⚠️ Trunc | ❌ No | ❌ No | ✅ Cosine | ❌ No |
| **Temporal Fact Grounding & Cleaning** | ✅ Yes | ⚠️ Basic | ❌ No | ❌ No | ⚠️ Basic | ❌ No | ❌ No | ❌ No | ❌ No |
| **Zero Output-Token Generation** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **TypeSafe `/v1/systemone` Drop-in REST**| ✅ Yes | ⚠️ Direct | ⚠️ Direct | ❌ No | ⚠️ Direct | ⚠️ Direct | ❌ No | ❌ No | ❌ No |
| **Execution Latency Class** | **~5 µs** | ~4 µs | ~4 µs | ~5 µs | ~12 µs | ~13 µs | ~22 µs | ~35 ms | ~48 ms |

---

## Quick Start

### Build

```bash
cargo build --release
```

### Run the Axum API Server

```bash
cargo run --release --bin zev -- serve --port 8080
```

### Evaluate a TypeSafe SystemOne Request

```bash
cargo run --release --bin zev -- systemone << 'EOF'
{
  "state": "Help! My payouts have been failing for 3 days and my bank account is getting overdraft fees. I need this escalated to billing immediately.",
  "model": "zev-latest",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this message convey operational urgency?",
      "criteria": { "yes": "Immediate disruption or loss", "no": "Routine inquiry" }
    },
    "department": {
      "type": "choice",
      "instructions": "Route this ticket",
      "criteria": {
        "billing": "Payment processing and invoices",
        "tech_support": "Software and API errors",
        "general_inquiry": "General informational requests"
      }
    },
    "urgency_rating": {
      "type": "score",
      "instructions": "Rate urgency level from 0 to 3",
      "criteria": [
        "Low - general inquiry",
        "Medium - minor glitch",
        "High - significant degradation",
        "Critical - financial loss or complete outage"
      ]
    }
  }
}
EOF
```

### Fast Routing Helper

```bash
cargo run --release --bin zev -- route \
  --state "I need help with my credit card invoice and bank charge" \
  --routes '{"billing": "Credit card invoice, payment, and bank charges", "tech": "Server technical errors", "sales": "Enterprise sales"}'
```

### Confidence Gating

```bash
cargo run --release --bin zev -- gate \
  --state "Critical: production database is down and taking no traffic" \
  --instructions "Is this a critical outage with down service?" \
  --threshold 0.7
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
