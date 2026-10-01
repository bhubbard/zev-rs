#!/usr/bin/env python3
"""
WorkflowEvals Benchmark Evaluation for Zev
Evaluates Zev against Hugging Face collection: https://huggingface.co/collections/typesafe/workflowevals
Datasets:
  1. typesafe/evalsafe-customer-service
  2. typesafe/evalsafe-invoice-processing
  3. typesafe/evalsafe-security-incidents
  4. typesafe/evalsafe-agent-trace-observability
"""

import argparse
import duckdb
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORKFLOW_DATASETS = {
    "customer_service": "typesafe/evalsafe-customer-service",
    "invoice_processing": "typesafe/evalsafe-invoice-processing",
    "security_incidents": "typesafe/evalsafe-security-incidents",
    "agent_trace_observability": "typesafe/evalsafe-agent-trace-observability",
}

BASELINE_DECISION_ACCURACY = {
    "customer_service": {
        "jev-1.13.0": 75.98,
        "claude-opus-5": 72.39,
        "claude-sonnet-5": 69.28,
        "claude-haiku-4-5": 55.39,
        "gpt-5.6-sol": 78.27,
        "deepseek-v4-flash": 76.80,
    },
    "invoice_processing": {
        "jev-1.13.0": 61.78,
        "claude-opus-5": 78.44,
        "claude-sonnet-5": 72.89,
        "claude-haiku-4-5": 42.89,
        "gpt-5.6-sol": 79.11,
        "deepseek-v4-flash": 69.78,
    },
    "security_incidents": {
        "jev-1.13.0": 61.67,
        "claude-opus-5": 66.25,
        "claude-sonnet-5": 60.83,
        "claude-haiku-4-5": 58.75,
        "gpt-5.6-sol": 62.50,
        "deepseek-v4-flash": 37.92,
    },
    "agent_trace_observability": {
        "jev-1.13.0": 71.62,
        "claude-opus-5": 75.23,
        "claude-sonnet-5": 68.02,
        "claude-haiku-4-5": 57.21,
        "gpt-5.6-sol": 76.58,
        "deepseek-v4-flash": 72.97,
    },
}

def clean_state_for_eval(s_obj):
    """Formats state cleanly so conversation text and metadata are easily parsed."""
    if isinstance(s_obj, str):
        return s_obj
    parts = []
    if isinstance(s_obj, dict):
        if "conversation" in s_obj and isinstance(s_obj["conversation"], list):
            for turn in s_obj["conversation"]:
                spk = turn.get("speaker", "user")
                txt = turn.get("text", "")
                parts.append(f"{spk.capitalize()}: {txt}")
        if "customer" in s_obj and isinstance(s_obj["customer"], dict):
            cust = s_obj["customer"]
            parts.append(f"Customer details: {json.dumps(cust)}")
        if "account_summary" in s_obj and isinstance(s_obj["account_summary"], dict):
            acc = s_obj["account_summary"]
            parts.append(f"Account summary: {json.dumps(acc)}")
        if "invoice" in s_obj:
            parts.append(f"Invoice details: {json.dumps(s_obj['invoice'])}")
        if "incident" in s_obj:
            parts.append(f"Security incident details: {json.dumps(s_obj['incident'])}")
        if "trace" in s_obj:
            parts.append(f"Agent trace: {json.dumps(s_obj['trace'])}")
        if not parts:
            parts.append(json.dumps(s_obj))
    else:
        parts.append(str(s_obj))
    return "\n\n".join(parts)

def eval_single_question(server_url, model, row):
    qid, kind, q_json, s_json, truth, jev_ref = row
    q_obj = json.loads(q_json)
    s_obj = json.loads(s_json)
    
    clean_state = clean_state_for_eval(s_obj)

    payload = {
        "model": model,
        "state": clean_state,
        "questions": {
            "q": q_obj
        }
    }
    
    req = urllib.request.Request(
        f"{server_url}/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read())
    lat_us = (time.perf_counter() - t0) * 1_000_000
    
    ans = res["answers"]["q"]
    pred = None
    if ans["type"] == "noul":
        pred = "true" if ans.get("noul", 0.0) >= 0.5 else "false"
    elif ans["type"] == "choice":
        pred = f'"{ans.get("choice")}"'
    elif ans["type"] == "score":
        probs = ans.get("probabilities", {})
        if probs:
            modal = max(probs.keys(), key=lambda k: probs[k])
            pred = f'"{modal}"'
        else:
            pred = f'"{int(round(ans.get("score", 0.0)))}"'
            
    match_consensus = (pred == truth)
    match_jev = (pred == jev_ref) if jev_ref is not None else None
    
    return {
        "qid": qid,
        "kind": kind,
        "pred": pred,
        "truth": truth,
        "jev_ref": jev_ref,
        "match_consensus": match_consensus,
        "match_jev": match_jev,
        "lat_us": lat_us,
        "source": ans.get("source", "native"),
    }

def run_workflow_benchmark(workflow_name, server_url, model, limit=None, workers=16):
    ds_repo = WORKFLOW_DATASETS[workflow_name]
    q_url = f"https://huggingface.co/datasets/{ds_repo}/resolve/main/data/questions.parquet"
    r_url = f"https://huggingface.co/datasets/{ds_repo}/resolve/main/data/run_results.parquet"
    
    print(f"\n==========================================================================================")
    print(f"  BENCHMARK: {workflow_name.upper()} ({ds_repo})")
    print(f"  Evaluating Model: '{model}' via {server_url}/v1/systemone")
    print(f"==========================================================================================")
    
    con = duckdb.connect()
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
        WITH jev_q AS (
            SELECT 
                case_id,
                q.node_id,
                q.question_id,
                q.kind,
                q.answer_json as jev_answer
            FROM '{r_url}', UNNEST(questions) as t(q)
            WHERE model.name = 'jev-1.13.0'
        )
        SELECT 
            q.question_instance_id,
            q.kind,
            q.question_json,
            q.state_json,
            q.consensus.answer_json as truth,
            j.jev_answer
        FROM '{q_url}' q
        LEFT JOIN jev_q j ON q.case_id = j.case_id AND q.node_id = j.node_id AND q.question_id = j.question_id
        {limit_clause}
    """
    
    print("Fetching questions and ground-truth consensus references...")
    rows = con.execute(query).fetchall()
    print(f"Loaded {len(rows)} question instances for evaluation.")
    
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(eval_single_question, server_url, model, r) for r in rows]
        for idx, f in enumerate(futures, 1):
            results.append(f.result())
            if idx % 100 == 0 or idx == len(rows):
                sys.stdout.write(f"\rProgress: {idx}/{len(rows)} questions ({idx/len(rows)*100:.1f}%)")
                sys.stdout.flush()
    total_time_s = time.perf_counter() - t_start
    print()
    
    # Compute metrics
    by_kind = {}
    latencies = [r["lat_us"] for r in results]
    latencies.sort()
    
    jev_matches = 0
    jev_total = 0
    consensus_matches = 0
    
    for r in results:
        k = r["kind"]
        stats = by_kind.setdefault(k, {"correct": 0, "total": 0, "jev_agree": 0, "jev_total": 0})
        stats["total"] += 1
        if r["match_consensus"]:
            stats["correct"] += 1
            consensus_matches += 1
        if r["match_jev"] is not None:
            stats["jev_total"] += 1
            jev_total += 1
            if r["match_jev"]:
                stats["jev_agree"] += 1
                jev_matches += 1

    p50_us = latencies[len(latencies) // 2]
    p95_us = latencies[int(len(latencies) * 0.95)]
    p99_us = latencies[int(len(latencies) * 0.99)]
    throughput = len(results) / total_time_s

    print("\n------------------------------------------------------------------------------------------")
    print(f"  QUESTION ACCURACY & LATENCY SUMMARY")
    print("------------------------------------------------------------------------------------------")
    print(f"{'Question Kind':<16} | {'Correct / Total':<18} | {'Accuracy (%)':<14} | {'Jev Concordance':<16}")
    print("─────────────────+────────────────────+────────────────+─────────────────")
    for k, s in sorted(by_kind.items()):
        acc = (s["correct"] / s["total"] * 100) if s["total"] else 0
        jev_acc = (s["jev_agree"] / s["jev_total"] * 100) if s["jev_total"] else 0
        print(f"{k:<16} | {s['correct']:>6}/{s['total']:<6}       | {acc:>10.2f}%    | {jev_acc:>11.2f}%")
    print("─────────────────+────────────────────+────────────────+─────────────────")
    overall_acc = consensus_matches / len(results) * 100
    overall_jev = (jev_matches / jev_total * 100) if jev_total else 0
    print(f"{'OVERALL':<16} | {consensus_matches:>6}/{len(results):<6}       | {overall_acc:>10.2f}%    | {overall_jev:>11.2f}%")
    print("------------------------------------------------------------------------------------------")
    print(f"Latency: p50={p50_us/1000:.2f} ms ({p50_us:.0f} µs) | p95={p95_us/1000:.2f} ms | p99={p99_us/1000:.2f} ms")
    print(f"Throughput: {throughput:.1f} decisions/sec | Total Time: {total_time_s:.2f}s")
    
    print("\n------------------------------------------------------------------------------------------")
    print(f"  COMPARISON: PUBLISHED WORKFLOWEVALS LEADERBOARD ({workflow_name})")
    print("------------------------------------------------------------------------------------------")
    baselines = BASELINE_DECISION_ACCURACY.get(workflow_name, {})
    for m_name, score in baselines.items():
        print(f"  {m_name:<28}: {score:.2f}%")
    print("==========================================================================================\n")
    
    return {
        "workflow": workflow_name,
        "model": model,
        "questions": len(results),
        "consensus_acc": overall_acc,
        "jev_concordance": overall_jev,
        "p50_us": p50_us,
        "throughput": throughput,
        "by_kind": by_kind,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Zev on Hugging Face WorkflowEvals")
    parser.add_argument("--workflow", choices=list(WORKFLOW_DATASETS.keys()) + ["all"], default="customer_service")
    parser.add_argument("--model", default="zev-latest", help="Model identifier: zev-latest, zev-gemma, zev-poe, zev-cascade, zev-default")
    parser.add_argument("--server", default="http://127.0.0.1:8999", help="Zev API server URL")
    parser.add_argument("--limit", type=int, default=200, help="Number of questions per dataset (None for all)")
    parser.add_argument("--workers", type=int, default=16, help="Thread pool concurrency")
    args = parser.parse_args()
    
    workflows = list(WORKFLOW_DATASETS.keys()) if args.workflow == "all" else [args.workflow]
    all_summaries = []
    for wf in workflows:
        summary = run_workflow_benchmark(wf, args.server, args.model, limit=args.limit, workers=args.workers)
        all_summaries.append(summary)
        
    if len(all_summaries) > 1:
        print("\n==========================================================================================")
        print("                   CROSS-WORKFLOW SUITE EVALUATION SUMMARY                               ")
        print("==========================================================================================")
        print(f"{'Workflow':<26} | {'Questions':<10} | {'Consensus Acc':<14} | {'Jev Concordance':<16} | {'p50 Latency':<12}")
        print("───────────────────────────+────────────+────────────────+──────────────────+─────────────")
        for s in all_summaries:
            print(f"{s['workflow']:<26} | {s['questions']:<10} | {s['consensus_acc']:>10.2f}%   | {s['jev_concordance']:>12.2f}%    | {s['p50_us']/1000:>8.2f} ms")
        print("==========================================================================================\n")

if __name__ == "__main__":
    main()
