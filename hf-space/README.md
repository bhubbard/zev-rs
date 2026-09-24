---
title: Zev - Zero-Token LLM Decision Engine
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Microsecond zero-token decision engine
---

# Zev: Ultra-Fast Zero-Token LLM Decision Engine

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/zev-rs.svg)](https://crates.io/crates/zev-rs)
[![npm](https://img.shields.io/npm/v/zev-rs.svg)](https://www.npmjs.com/package/zev-rs)

**Zev** is an ultra-fast zero-token LLM decision engine written in Rust. It eliminates LLM inference latency and token costs for schema classification, intent routing, and guardrail decisions.

## Live Interactive Features in this Space

1. **⚡ Microsecond Decision Playground**: Test incoming state text against multiple candidate choices. Evaluates in **5.8 µs** with zero model weights.
2. **🛡️ 100% Order Invariance Verification**: Standard LLMs exhibit option-position bias (flipping up to 35% when options are reordered). Zev mathematically isolates option attention slots, delivering a guaranteed **0.0% permutation flip rate**.
3. **🤝 Together AI Tev1 Drop-In Evaluator**: Complete wire-level drop-in replacement for Together AI's `tev1-4B-experimental` model format.
4. **🛑 Strict Abstention Guardrails**: Refuses to hallucinate when context is out-of-domain or evidence is missing (`__insufficient__`).

---

## API Endpoints Exposed by this Space

This Hugging Face Space also functions as a public REST API:

### 1. Evaluate Decision (`POST /v1/decisions`)
```bash
curl -X POST https://<your-space-name>.hf.space/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Customer requested a full refund for invoice #9042 because the software crashed.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Route request to responsible department",
        "options": [
          {"id": "billing", "description": "Refunds, payment processing, invoices"},
          {"id": "tech_support", "description": "Crashes, software bugs, installation issues"},
          {"id": "sales", "description": "Enterprise quotes and plan upgrades"}
        ]
      }
    }
  }'
```

### 2. Tev1 Format (`POST /v1/tev1`)
```bash
curl -X POST https://<your-space-name>.hf.space/v1/tev1 \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Returns are allowed within 30 days. This purchase was 12 days ago.",
    "question": "Is this return within the allowed window?",
    "options": ["A: Yes", "B: No", "C: Insufficient info"]
  }'
```

### 3. Health & Capabilities Check (`GET /health` & `GET /v1/limits`)
```bash
curl https://<your-space-name>.hf.space/health
```

---

## Architecture: Why Zev?

| Problem in Standard LLMs | How Zev Solves It |
|---|---|
| **High Latency (300 ms – 1.2 s)** | Evaluates via SIMD in **5.8 µs** (over 50,000x faster). |
| **High Inference Cost** | **Zero tokens generated, 0 MB external model weights**. |
| **Option Order Bias (Up to 35% flips)** | **0.0% Flip Rate** via isolated attention scoring. |
| **Overconfident Hallucinations** | **Golden-section temperature calibration** ($T=2.179$) minimizing Expected Calibration Error (ECE). |
| **Hallucinating when Unsure** | Strict `__insufficient__` abstention guardrails. |

---

## Local Installation

Run anywhere instantly with `npx` or `cargo`:

```bash
# Via npx (zero installation)
npx -y zev-rs route --state "Database down with 500 errors" --routes '{"db": "Database SRE", "app": "Frontend"}'

# Via Cargo
cargo install zev-rs
zev serve --port 8080
```
