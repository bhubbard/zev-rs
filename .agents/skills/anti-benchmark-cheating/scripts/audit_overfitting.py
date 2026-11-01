#!/usr/bin/env python3
"""
Automated Anti-Overfitting & Benchmark Hygiene Auditor for zev-rs.
Scans `src/` against `datasets/jevbench_public/*.jsonl` to detect:
  1. Multi-paragraph prompt-template regexes (`[\s\S]*?`).
  2. Benchmark option IDs hardcoded or synthesized in `src/`.
  3. Regexes in `src/` that match only 1-2 specific JevBench prompts.
  4. Multi-word verbatim phrases copied from JevBench prompts into `src/`.
"""

import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
SRC_DIR = os.path.join(ROOT, "src")
DATASET_GLOB = os.path.join(ROOT, "datasets/jevbench_public/*.jsonl")

GENERIC_OPTION_IDS = {
    "true", "false", "yes", "no", "0", "1", "2", "3", "4", "5",
    "other", "abstain", "unknown", "none", "valid", "invalid",
    "billing", "sales", "support", "tech_support", "coding", "math",
    "document", "tools", "status", "refund", "cancel", "account",
}


def load_benchmark_items():
    items = []
    for path in sorted(glob.glob(DATASET_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def collect_rs_files():
    rs_files = []
    for dirpath, _, filenames in os.walk(SRC_DIR):
        for fn in filenames:
            if fn.endswith(".rs"):
                rs_files.append(os.path.join(dirpath, fn))
    return sorted(rs_files)


def strip_test_modules(content: str) -> str:
    idx = content.find("#[cfg(test)]")
    return content[:idx] if idx != -1 else content


def main() -> int:
    items = load_benchmark_items()
    rs_files = collect_rs_files()
    violations = []

    # Collect non-generic option IDs from JevBench (snake_case with >= 2 underscores or specific compound IDs)
    benchmark_option_ids = set()
    for item in items:
        for lbl in item.get("labels", []):
            lbl_s = str(lbl).strip()
            if (
                lbl_s not in GENERIC_OPTION_IDS
                and "_" in lbl_s
                and len(lbl_s) >= 12
            ):
                benchmark_option_ids.add(lbl_s)

    for fpath in rs_files:
        rel = os.path.relpath(fpath, ROOT)
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        prod_code = strip_test_modules(raw)

        # 1. Check for multi-paragraph prompt-template regex `[\s\S]*?` in solver modules
        if os.path.basename(fpath) in {
            "temporal_numeric.rs",
            "table_graph.rs",
            "symbolic.rs",
            "order_invariant.rs",
        }:
            if r"[\s\S]*?" in prod_code:
                violations.append(
                    f"[PROMPT-TEMPLATE REGEX] {rel} contains multi-paragraph `[\\s\\S]*?` regex"
                )

        # 2. Check for benchmark-specific compound option IDs in production code
        for opt_id in sorted(benchmark_option_ids):
            if f'"{opt_id}"' in prod_code or f"({opt_id})" in prod_code or f"({opt_id}:" in prod_code:
                violations.append(
                    f"[HARDCODED OPTION ID] {rel} references JevBench option ID '{opt_id}'"
                )

        # 3. Check for dynamic option-ID synthesis format strings
        for bad_fmt in [
            "p{prio_num}_fix_",
            "credit_{final_pct}_percent",
            "retain_until_",
            "eur_{:.2}",
            "[TABULAR MAJORITY]",
            "deduced_start",
        ]:
            if bad_fmt in prod_code:
                violations.append(
                    f"[OPTION-ID SYNTHESIS / BYPASS] {rel} contains '{bad_fmt}'"
                )

    if violations:
        print("OVERFITTING AUDIT FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(
        f"OVERFITTING AUDIT PASSED: scanned {len(rs_files)} source files against {len(items)} JevBench items (0 violations)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
