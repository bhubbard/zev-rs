# 🛡️ zev-rs Security, Safety & Architecture Audit

**Audited By:** [`rust-code-mcp`](https://github.com/molaco/rust-code-mcp) (HIR-driven Hypergraph Analysis, AST Body Auditing & Tantivy Symbol Graph)  
**Date:** September 28, 2026  
**Target:** [`zev-rs`](file:///Users/bhubbard/PROJECTS/zev-rs) (61 Rust files, 4,910 indexed AST chunks, 713 hypergraph nodes, 857 bindings)

---

## 📊 Executive Scorecard

| Category | Findings | Status | Key Risk / Impact |
| :--- | :---: | :---: | :--- |
| **Unsafe Blocks** | **11** | ⚠️ Warning | 11 unsafe blocks in `kev/kernels.rs` & `kev/model.rs` without `// SAFETY:` justifications. |
| **Panic & Unwrap Invariants** | **57** | 🚨 High Risk | Raw `.unwrap()` on string splitting, slice indexing, and queue popping that can crash production threads. |
| **Unbounded Recursion** | **2** | ⚠️ Warning | Stack overflow hazard in recursive JSON / AST tree serializers (`object_to_json`). |
| **Dead Public API Surface** | **193** | ℹ️ Hygiene | Internal implementation functions and structs exposed as `pub` rather than `pub(crate)`. |
| **Missing Public Docs** | **86** | ℹ️ Hygiene | 86 exported library functions, structs, and constants lack `///` documentation comments. |
| **Global Mutable State** | **1** | ℹ️ Low Risk | Single `OnceLock<ApfelNeuralBackend>` in `neural.rs`. |
| **Channel Concurrency** | **1** | ℹ️ Checked | Bounded `sync_channel(MAX_QUEUE_DEPTH)` in `kev/serve.rs`. |

---

## 1. 🚨 Critical & High-Risk Findings

### 1.1 Unchecked `.unwrap()` on String Slicing (`order_invariant.rs`)
`order_invariant.rs:942, 1229, 1768, 1990, 3126`

* **The Issue:**
  ```rust
  let next = text[word.len()..].chars().next().unwrap();
  let prev = text[..text.len() - word.len()].chars().last().unwrap();
  ```
* **Why it Crashes:**
  If `text` contains multi-byte UTF-8 sequences (e.g. emojis, accented characters, Cyrillic, Chinese), byte slicing `text[word.len()..]` will panic with:
  `byte index ... is not a char boundary; it is inside '...'`.
  Furthermore, if `text[word.len()..]` is empty, `.chars().next().unwrap()` panics immediately.
* **Fix:**
  Replace with safe boundary checks using `.chars()` iterator or `floor_char_boundary` / `ceil_char_boundary`.

---

### 1.2 Unsafe Blocks Lacking `// SAFETY:` Comments (`kev/kernels.rs`)
`src/kev/kernels.rs:6591, 9386, 10307, 10897, 13427, 14094, 14779, 14981, 16068`  
`src/kev/model.rs:7820, 7885`

* **The Issue:**
  11 `unsafe` blocks directly invoke low-level CUDA driver pointers and memory mappings without documenting the required safety invariants:
  ```rust
  // Current:
  unsafe {
      cuda_fwd(...);
  }
  ```
* **Fix:**
  Comply with the standard Rust Safety RFC. Document precondition guarantees:
  ```rust
  // SAFETY: Pointers are verified non-null, properly aligned to 32 bytes,
  // and buffer allocations have valid lifetimes tied to the GPU context.
  unsafe {
      cuda_fwd(...);
  }
  ```

---

### 1.3 Panic Hazards in Engine Helpers (`engine.rs`)
`src/engine.rs:34706, 36214`

* **The Issue:**
  ```rust
  // in confidence_gate:
  let resp = self.evaluate(&req)?;
  let ans = resp.answers.get("gate").cloned().unwrap();

  // in route_with_distribution:
  let resp = self.evaluate(&req)?;
  let ans = resp.answers.get("route").unwrap();
  ```
* **Why it's Dangerous:**
  If the evaluation skips or abstains from the `"gate"` or `"route"` question, `resp.answers.get(...)` returns `None`, causing an unrecoverable worker thread crash.
* **Fix:**
  Replace `.unwrap()` with `ok_or_else(|| ZevError::EvaluationFailed("Missing question in output"))?`.

---

### 1.4 Unbounded Stack Recursion (`convert_head.rs`)
`src/bin/convert_head.rs:1905-1922` (`convert_head::object_to_json`)

* **The Issue:**
  `object_to_json` recursively calls itself when serializing nested structures. A maliciously deep or cyclic payload causes an uncatchable OS stack overflow (`SIGSEGV`).
* **Fix:**
  Pass a `depth: usize` counter with a `MAX_DEPTH = 64` bail-out guard, or rewrite as an iterative loop with an explicit heap-allocated worklist.

---

## 2. 🧹 Architecture, API & Code Hygiene

### 2.1 Over-Exposed Public Surface (193 Dead `pub` Items)
`dead_pub_report` flagged **193 public declarations** that are never consumed outside their defining modules:

* **Internal Helpers to Demote to `pub(crate)`:**
  * `zev::abstain::ABSTAIN_PREFIXES`
  * `zev::calibration::scaled_softmax_slice`
  * `zev::calibration::family_calibrated_temperature`
  * `zev::clm::VectorArena` & `zev::clm::ContrastiveHead`
  * `zev::cascade::SequentialCascadeRunner`
  * `zev::grep::CodeDeclaration`
* **Recommendation:**
  Default to `pub(crate)` across internal helper crates to prevent accidental breaking changes to external users.

---

### 2.2 Missing Documentation Comments (86 Public Items)
`missing_docs_audit` identified 86 exported symbols without doc comments (`///`), including key public types:
* `zev::calibration::resolve_temperature`
* `zev::clm::ArenaStats`
* `zev::concept_knowledge::boost_science_concept_associations`
* `zev::cascade::StageKind`

---

## 3. 🛠️ Actionable Remediation Checklist

- [ ] **R-01 (UTF-8 Safety):** Replace byte slicing in `order_invariant.rs` with `char_indices()` and `is_char_boundary()` to eliminate multi-byte unicode panics.
- [ ] **R-02 (Safety Invariants):** Add `// SAFETY:` doc blocks explaining pointer alignment, null checks, and bounds on all 11 `unsafe` blocks in `src/kev/kernels.rs`.
- [ ] **R-03 (API Robustness):** Replace `.unwrap()` with `?` error propagation in `engine::confidence_gate` and `engine::route_with_distribution`.
- [ ] **R-04 (Recursion Guard):** Add depth tracking to `convert_head::object_to_json` (`if depth > MAX_RECURSION { return Err(...); }`).
- [ ] **R-05 (Visibility Demotion):** Restrict non-public internal helpers from `pub` to `pub(crate)`.
