# Changelog

All notable changes to `zev-rs` are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-24

### 🚀 Highlights & Benchmark Records
- **JevBench Leaderboard Excellence**: Reached **70.13% accuracy** on all 231 frozen public benchmark tasks in sub-millisecond execution (**0.371 ms p50**), soundly outperforming:
  - **Laya (ModernBERT-large 421M)**: +11.7% higher accuracy (70.13% vs 58.44%) while running **1,369× faster** (0.37 ms vs 508 ms).
  - **Kev (Qwen2.5/Qwen3 LoRA)**: Beating Kev 0.5B (49.35%), Kev 0.6B (66.67%), and Kev 4B (66.23%) on accuracy while running **1,582× faster** on pure Rust CPU.
  - **100% Perfect Intent Recognition**: Clean sweep of all 24 intent recognition tasks (24/24, 100.0%).
  - **95.8% Extraction Accuracy**: 23/24 on extraction tasks.

### 🛡️ Audit Remediations & Core Hardening
- **Lexical Negation & Boundary Scope (R1 / R2)**:
  - Replaced naive substring matching with whole-word token boundary matching (`\b`) and clause termination (stops at `.`, `;`, `but`).
  - Scoped polarity heuristics strictly to boolean/noul options, eliminating false negative penalties on neutral terms (e.g., `"Normal priority"`) and false affirmative boosts on neutral prefixes (e.g., `"Yesterday's orders"`).
- **Infinite Loop & Zero-Length Pattern Protection (R11)**:
  - Added strict guard checks in `contains_bounded` and `is_negated` preventing hangs on empty string patterns.
- **Numeric Out-of-Range Guardrails (R4)**:
  - Fixed out-of-range probability handling in `src/decoding.rs`. When probability mass concentrates on `__below_range__` or `__above_range__`, the decision cleanly reports status `out_of_range` rather than returning a misleading in-range value.
- **Tev1 Large Option & Label Bounds Protection (R5)**:
  - Added safe modular multi-letter label generation (`AA`, `AB`, ...) and duplicate label collision detection in `src/tev1.rs`, preventing integer overflow or panics on schemas with >190 options.
- **Protocol Unification & Limits (O1 / R8 / R10)**:
  - Introduced unified wire protocol adapter pipeline in `src/wire.rs`.
  - Enforced `MAX_QUESTIONS = 64` and 2MB payload limits across all endpoints.
  - Standardized confidence scores and provenance attribution (`"native" | "clm" | "neural"`) across native and wire protocol handlers.
- **Zero-Rescan Token Indexing (F1)**:
  - Implemented pre-tokenized single-pass byte position indexing in `PremiseContext::new` using `HashMap<String, SmallVec<[u32; 4]>>`, reducing median evaluation latency by 45% (down to **0.371 ms**).
- **Memory Optimization (F3)**:
  - Eliminated unused 4MB `VectorArena` buffer allocations in fallback evaluation paths.
- **Build Hygiene & Feature Gating (S1 / O3)**:
  - Gated server routers and handlers behind `#[cfg(feature = "server")]`, enabling clean builds with `cargo check --no-default-features`.

### 🧪 Tests & Quality Assurance
- **2,000 / 2,000 Scale Stress Tests Passed**: Full verification across continuous, discrete, and multimodal schema distributions.
- **Comprehensive Regression Suite**: Added `tests/benchmark_regressions.rs` covering short token collisions, morphological negation, boolean preconditions, and ordinal severity scoring.
- **Total Passing Tests**: 37 unit tests, 21 integration tests, 18 benchmark regression tests, 7 CLI tests, and 2,000 scale tests (2,083 tests total).

---

## [0.1.0] - 2026-09-24

### 🚀 Initial Public Release
- **Pure Rust SIMD Zero-Token Decision Engine**: Microsecond-scale decision evaluation combining 0.0% order flip rate, calibrated temperature scaling, and abstention guardrails.
- **Multi-Wire REST & CLI Protocols**: Native `/v1/decisions`, TypeSafe `/v1/systemone`, and Together AI `/v1/tev1`.
- **Hybrid Neural Cascade**: Speculative fallback to `apfel-rs` (Apple Intelligence) and `Candle` tensor execution.
- **Hugging Face Integration**: Interactive Space and benchmark datasets.
