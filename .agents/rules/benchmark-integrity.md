---
trigger: always_on
---

# Benchmark Integrity Rule

Any agent or developer working in this repository MUST strictly follow these rules:

1. **Zero Tolerance for Benchmark Cheats**:
   - NEVER hardcode matching strings from benchmark questions, options, or test states into `src/`.
   - NEVER inject dataset-specific entity names, ticket IDs, or numbers to force test passes.
   - NEVER add artificial logit boosts (e.g. `+= 6.0`, `+= 8.0`) or early returns that bypass lexical/neural evaluation.

2. **Honest Baselines**:
   - The pure zero-token SIMD/WASM engine has a natural lexical baseline (~54% on JevBench).
   - Do NOT paper over this baseline with question overrides.
   - Complex reasoning (word count constraints, unproved precondition inference, multi-hop lookups) belongs in speculative neural fallback (`apfel`, `gemma`, `clef`), not in hardcoded fast-path heuristics.

3. **Continuous Enforcement**:
   - All code in `src/` must pass `cargo test --test anti_overfitting_guard`.
   - Any commit adding benchmark case IDs, prompt memorization, or artificial score escalations will be rejected.
