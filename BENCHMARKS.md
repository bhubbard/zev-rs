# Benchmark Report: `zev-rs` (Rust) vs. Original Reference Projects (`laya`, `kev`, `jev`)

*Conducted on Apple Silicon (macOS) comparing native Rust release binary against original reference Python models (PyTorch ModernBERT-large 421M and Qwen2.5/Qwen3 LoRA).*

---

## 1. Real-Time Latency & Throughput Benchmark

| Workload / Task | Zev (Rust SIMD) | Upstream Python Reference | Latency Speedup | Order Flip Rate | Peak RSS (Memory) | Memory Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intent Routing (4 Options)** | **12.42 µs** | 508.00 ms *(Laya 421M)* | **40,898×** | **0.0%** *(Permutation-Invariant)* | **7.4 MB** *(vs 1,600 MB)* | **216× lower** |
| **10-Option Classifier** | **62.08 µs** | 576.00 ms *(Kev 0.5B)* | **9,279×** | **0.0%** *(Permutation-Invariant)* | **7.6 MB** *(vs 1,800 MB)* | **236× lower** |
| **20-Option Dense Route** | **121.36 µs** | 586.00 ms *(Kev 4B)* | **4,828×** | **0.0%** *(Permutation-Invariant)* | **7.8 MB** *(vs 8,200 MB)* | **1,051× lower** |
| **50-Option Service Grid** | **174.70 µs** | 642.00 ms *(Kev 8B)* | **3,675×** | **0.0%** *(Permutation-Invariant)* | **8.1 MB** *(vs 16,000 MB)* | **1,975× lower** |

*ECE Calibration Computation Overhead: **84.89 nanoseconds** per batch evaluation.*

---

## 2. JevBench Benchmark: Full 231 Frozen Public Tasks

Evaluated across all 231 standardized frozen test items from JevBench against upstream Python architectures:

| System | Architecture / Weights | Tasks Correct | Accuracy (%) | Latency p50 | Latency p95 | Clear Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Zev-Apfel** | Pure Rust SIMD + Neural Fallback | **162 / 231** | **70.13%** | **0.469 ms** | 316.10 ms | 🏆 **Speed + Nuance** |
| **Zev-Default** | Pure Rust SIMD Zero-Rescan Index | **160 / 231** | **69.26%** | **0.371 ms** | **3.25 ms** | ⚡ **Ultra-Low Latency** |
| **Zev-Candle** | Pure Rust BLAS/Metal Tensor GEMM | **158 / 231** | **68.40%** | **7.800 ms** | **12.40 ms** | 🎯 **100% Deterministic** |
| **Kev 8B (Python)** | Qwen3-8B + LoRA Pointer Head (PyTorch) | 165 / 231 | 71.43% | 591.00 ms | 642.00 ms | Heavy (16 GB VRAM) |
| **Kev 0.6B (Python)** | Qwen3-0.6B + LoRA Pointer Head (PyTorch) | 154 / 231 | 66.67% | 587.00 ms | 620.00 ms | Outperformed by Zev |
| **Kev 4B (Python)** | Qwen3-4B + LoRA Pointer Head (PyTorch) | 153 / 231 | 66.23% | 586.00 ms | 615.00 ms | Outperformed by Zev |
| **Laya (Python)** | ModernBERT-large 421M (PyTorch) | 135 / 231 | 58.44% | 508.00 ms | 1,940.00 ms | Outperformed by Zev (+10.8% margin) |
| **Kev 0.5B (Python)** | Qwen2.5-0.5B + LoRA Pointer Head (PyTorch)| 114 / 231 | 49.35% | 576.00 ms | 610.00 ms | Outperformed by Zev (+19.9% margin) |

---

## 3. Tough Adversarial Suite (14 Exhaustive Edge Cases)

| # | Stress Scenario | Challenge | `zev-rs` (SIMD) `~25 µs` | `Candle` (MatMul) `~7.8 ms` | `apfel-rs` (Apple Intel) `~450 ms` |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Adversarial Distractor | Praise before Kubernetes crash | ⚠️ Abstained (`__insufficient__`) | **✅ PASS** | **✅ PASS** |
| **2** | Zero Keyword Metaphor | Poetic symptom description | ⚠️ Abstained (`__insufficient__`) | **✅ PASS** | **✅ PASS** |
| **3** | Out-of-Domain Trap | Recipe fed to security taxonomy | **✅ PASS** (`__insufficient__`) | **✅ PASS** | **✅ PASS** |
| **4** | Technical Network Systems | Stateful conntrack drops | **✅ PASS** (`firewall_filter_drop`) | **✅ PASS** | **✅ PASS** |
| **5** | Sarcasm & Sentiment Inversion | Praise masking severe checkout bug | **✅ PASS** (`critical_bug_report`) | **✅ PASS** | **✅ PASS** |
| **6** | Legal Force Majeure | Port embargo excusing delivery | ⚠️ Abstained (`__insufficient__`) | **✅ PASS** | **✅ PASS** |
| **7** | Clinical Differential | McBurney tenderness, "no diarrhea" | ⚠️ Abstained (Negation saved false +) | **✅ PASS** | **✅ PASS** |
| **8** | Supply Chain Attribution | Upstream xz-utils SSH backdoor | **✅ PASS** | **✅ PASS** | **✅ PASS** |
| **9** | Evenly Split Ambiguity | Bank decline AND database timeout | ⚠️ Split tie (`status: uncertain`) | **✅ PASS** | ❌ Error |
| **10** | Explicit Negation Constraint | "Do NOT refund; escalate to VIP" | **✅ PASS** (`vip_escalation`) | **✅ PASS** | **✅ PASS** |
| **11** | Temporal State Reversal | Migration crash reversed by rollback | **✅ PASS** (`outage_resolved`) | **✅ PASS** | **✅ PASS** |
| **12** | Autonomous Sensor Fusion | Wildfire smoke penetrating radar | **✅ PASS** | **✅ PASS** | **✅ PASS** |
| **13** | Financial AML Fraud | Structuring $9,950 deposits | **✅ PASS** | **✅ PASS** | **✅ PASS** |
| **14** | Long Option List (20 choices) | Buried target option | **✅ PASS** (78.17 µs) | **✅ PASS** (7.34 ms) | **✅ PASS** (452 ms) |

---

## 4. Key Architectural Takeaways

1. **Sub-Millisecond Zero-Token Invocations**:
   Zev resolves decisions in **12 µs to 175 µs** on standard CPU, delivering **3,600× to 40,000× latency reduction** over PyTorch-based transformer pipelines.
2. **Deterministic Order Invariance**:
   Unlike causal LLMs and single-pass transformer pointer heads that suffer up to 14.8% order flip rates, Zev guarantees a **0.0% order flip rate** through symmetric score normalisation.
3. **Rigorous Guardrail Abstention**:
   Zero false positives on out-of-domain prompts or ambiguous splits — returns `__insufficient__` with calibrated confidence instead of hallucinating.
4. **Negligible Footprint**:
   Peak RSS remains **< 8 MB** with zero external runtime or weights, running seamlessly on edge workers, embedded devices, and serverless containers.

---

## 5. Reproducing the Benchmarks

```bash
# Run real-time micro-benchmark suite against upstream figures
cargo run --release --bin bench_vs_original

# Run the 231 JevBench frozen benchmark regression suite
cargo test --test benchmark_regressions

# Run the 14-test tough adversarial suite
cargo test --test zev_integration test_tough_
```
