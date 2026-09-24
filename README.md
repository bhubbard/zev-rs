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

## Performance Benchmark

Benchmarked over 1,000 iterations on Apple Silicon on a multi-question triage task:

```text
zev-rs (APEX) :  18.69 µs / eval | 53,503 evals/sec | 0 output tokens generated
```

Includes end-to-end temporal fact grounding, order-invariance, calibrated scaling, guardrails, and moment decoding in under 20 microseconds.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
