---
name: anti-benchmark-cheating
description: >-
  Enforces strict benchmark hygiene and prevents test-set overfitting, prompt-template
  regexes, option-ID synthesis, and artificial logit bypasses when optimizing accuracy
  or evaluating benchmarks (such as JevBench, SimpleBench, BFCL, or Decision Index).
  Activate this skill whenever working on accuracy improvements, benchmark evaluations,
  preprocessing, symbolic/tabular solvers, or scoring logic.
---

# Anti-Benchmark-Cheating & Overfitting Prevention Skill

Use this skill whenever asked to improve accuracy, fix failing benchmark cases, or modify scoring/preprocessing logic on any benchmark suite.

## What Counts as Cheating (Strictly Prohibited)

1. **Direct Entity / Case-ID Memorization**
   - Hardcoding benchmark case IDs (`hard-opus-*`, `hard-sol-*`), company/person names (`ALDERMOOR`, `WESERLINK`, `Leonie Marsh`), or literal amounts/dates from test cases.
2. **Prompt-Template Regexes ("Soft Overfitting")**
   - Writing regexes or string checks that match the wording or layout of 1–3 specific benchmark prompts (e.g., `requires two independent reviewers unless the change is documentation-only`, `Only INTERACTIVE sign-ins are compared`, `tare weight ... restraint total`), **even if** the code computes the arithmetic or date math dynamically after matching.
   - Using multi-paragraph `[\s\S]*?` regexes in `src/` to glue together clauses from a specific benchmark prompt.
3. **Option-ID Synthesis & String Evasion**
   - Emitting benchmark option IDs directly from preprocessors/solvers (`blocked_second_review`, `conservation_hold`, `bus_substitution`, `late_by_under_2h`, `require_director`, `needs_authentication`).
   - Using `format!()` to construct a forbidden option ID (e.g., `format!("p{prio}_fix_{num}{unit}")` to produce `p0_fix_24h` without triggering a static string grep).
4. **Artificial Logit Override Channels**
   - Injecting magic tags (`VERIFIED_OUTCOME:`, `[TABULAR MAJORITY]`, `deduced_start`) that force a single matched option ID to win by `max_other + X`, bypassing general scoring.
5. **Inspect-and-Patch Loops on Held-Out Benchmark Items**
   - Reading individual failing items from `datasets/jevbench/hard.jsonl` and writing one solver/regex per failing item (`0a`, `0b`, ..., `0x`).

## Mandatory Verification Workflow

Before committing any accuracy or engine change:

1. **Run the Automated Overfitting Audit Script**:
   Execute [audit_overfitting.py](./scripts/audit_overfitting.py):
   ```bash
   python3 .agents/skills/anti-benchmark-cheating/scripts/audit_overfitting.py
   ```
2. **Run the Rust Anti-Overfitting Guard Test Suite**:
   ```bash
   cargo test --test anti_overfitting_guard
   ```
3. **Generality Litmus Test**:
   Ask: *"Would I have written this exact code, regex, or string check if I had never seen the benchmark prompts or option IDs?"* If the answer is **no**, revert it immediately.
