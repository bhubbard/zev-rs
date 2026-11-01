# Zev-RS Agent Rules (`AGENTS.md`)

## Zero-Tolerance Anti-Benchmark-Cheating Policy

When working on `zev-rs`, you MUST adhere to strict benchmark integrity at all times:

1. **Never Write Prompt-Specific Solvers or Regexes**:
   - Do NOT inspect failing items in `datasets/jevbench/*.jsonl` or `datasets/jevbench_public/*.jsonl` to craft per-item regexes, phrase checks, or specialized multi-table joiners, **even if** the math/date calculation inside the solver is computed dynamically.
   - If a regex, string check, or table schema rule would not have been written without looking at the specific JevBench test prompts, **it is overfitting/cheating and is strictly forbidden**.
2. **Never Emit or Synthesize Benchmark Option IDs in Preprocessors**:
   - Preprocessors (`src/preprocessor.rs`, `src/table_graph.rs`, `src/temporal_numeric.rs`, `src/symbolic.rs`, `src/policy_dag.rs`) must NEVER emit benchmark option IDs (e.g., `blocked_second_review`, `conservation_hold`, `bus_substitution`) or synthesize them with `format!()` (e.g., `p{prio}_fix_{num}h`, `credit_{pct}_percent`) to evade static string guards.
3. **Never Add Logit Override Channels**:
   - Do NOT add dominance boosts (`max_other + X`, `deduced_start`, `VERIFIED_OUTCOME:`) that force an injected preprocessor tag to override the general scoring engine.
4. **Always Run Both Anti-Overfitting Checks Before Committing**:
   - `python3 .agents/skills/anti-benchmark-cheating/scripts/audit_overfitting.py`
   - `cargo test --test anti_overfitting_guard`
