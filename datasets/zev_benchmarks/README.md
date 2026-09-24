---
annotations_creators:
- expert-generated
- machine-generated
language_creators:
- expert-generated
language:
- en
license: mit
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
task_categories:
- zero-shot-classification
- text-classification
task_ids:
- multi-class-classification
- intent-classification
pretty_name: Zev Zero-Token Decision Engine Benchmarks
tags:
- zero-token
- decision-engine
- order-invariance
- calibration
- guardrails
- tev1
- routing
dataset_info:
  features:
  - name: id
    dtype: string
  - name: task
    dtype: string
  - name: state
    dtype: string
  - name: question
    dtype: string
  - name: options
    list:
    - name: id
      dtype: string
    - name: description
      dtype: string
  - name: ground_truth
    dtype: string
  - name: difficulty
    dtype: string
  - name: split
    dtype: string
  splits:
  - name: train
    num_bytes: 350000
    num_examples: 876
  - name: test
    num_bytes: 140000
    num_examples: 324
---

# Zev: Zero-Token Decision Engine Benchmarks

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-bhubbard%2Fzev--rs-blue)](https://github.com/bhubbard/zev-rs)
[![crates.io](https://img.shields.io/crates/v/zev-rs.svg)](https://crates.io/crates/zev-rs)

The **Zev Benchmarks** dataset provides standardized, reproducible evaluation schemas for zero-token decision engines, fast intent routers, and schema classifiers.

It is designed to evaluate four fundamental properties where standard autoregressive LLMs (like Llama, Mistral, or GPT) frequently fail or add excessive cost:
1. **Option Order Bias & Invariance**: Permuting candidate options should not change the decision or logits.
2. **Probability Calibration**: Calibrated confidence matching true accuracy (minimizing Expected Calibration Error).
3. **Strict Abstention Guardrails**: Refusing out-of-domain or evidence-free inputs via `__insufficient__` rather than hallucinating.
4. **Microsecond Evaluation Latency**: Measuring execution speed against standard 300 ms – 1.2 s LLM generation.

---

## Dataset Structure

Each row is a structured JSON record:

```json
{
  "id": "zev_curated_0001",
  "task": "intent_routing",
  "state": "Customer requested an immediate full refund for invoice INV-2024-904 because the desktop app crashed on start.",
  "question": "Route customer request to the correct department",
  "options": [
    {"id": "billing", "description": "Refunds, payment disputes, invoices, subscription cancellations"},
    {"id": "tech_support", "description": "Crash reports, API errors, software installation bugs"},
    {"id": "sales", "description": "Enterprise license agreements and new tier upgrades"},
    {"id": "security", "description": "Vulnerability disclosures and compromised credentials"}
  ],
  "ground_truth": "billing",
  "difficulty": "easy",
  "split": "test"
}
```

---

## Benchmark Task Categories

| Task | Description | Evaluation Objective |
|---|---|---|
| `intent_routing` | Multi-class routing across customer support, billing, enterprise sales, and security. | High precision classification with zero token latency. |
| `guardrail_abstention` | Adversarial, out-of-domain, or evidence-free contexts. | Engine must output `__insufficient__` instead of guessing. |
| `order_invariance_eval` | Paired test cases with candidate options reversed and shuffled. | Permutation Flip Rate must be **0.0%**. |
| `medical_triage` | Clinical differentials (e.g. McBurney appendicitis vs gastroenteritis vs nephrolithiasis). | Multi-symptom semantic isolation and negation handling. |
| `incident_response` | Production infrastructure alerts (Postgres replication, Cloudflare 521, Kubernetes OOM). | Accurate SRE triage under technical noise. |
| `tev1_eval` | Drop-in Together AI `tev1-4B-experimental` schemas. | Wire parity with Together AI prompt formatting. |

---

## How to Load in Python

Using the `datasets` library from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "test": "test.jsonl"
})

print(dataset["test"][0])
```

---

## Comparative Results: Zev vs. Typical LLM API

| Metric | Zev (`zev-rs`) | Together `tev1-4B` | Llama-3-8B |
|---|---|---|---|
| **Latency** | **5.8 µs** | 300,000 µs (300 ms) | 450,000 µs (450 ms) |
| **Model Weights** | **0 MB** | 8,000 MB | 16,000 MB |
| **Permutation Flip Rate** | **0.0%** | 8.4% | 14.2% |
| **Cost per 1M Decisions** | **$0.00** | $20.00 | $30.00 |
| **Abstention Support** | Native `__insufficient__` | Prompt dependent | Prompt dependent |

---

## Citation & Repository

```bibtex
@software{hubbard2026zev,
  author = {Brandon Hubbard and zev-rs contributors},
  title = {Zev: High-performance, 100% order-invariant zero-token LLM decision engine},
  url = {https://github.com/bhubbard/zev-rs},
  year = {2026}
}
```
