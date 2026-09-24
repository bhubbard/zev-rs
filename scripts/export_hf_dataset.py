#!/usr/bin/env python3
"""
Zev Decision Engine — Hugging Face Benchmark Dataset Generator & Evaluator.

Exports standardized benchmark datasets for zero-token decision engines,
intent routers, order-invariance evaluation, and guardrail abstention.

Usage:
  python3 scripts/export_hf_dataset.py --output-dir datasets/zev_benchmarks
  python3 scripts/export_hf_dataset.py --eval  # Run evaluation against zev CLI
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any

# Curated benchmark definitions across core decision tasks
BENCHMARK_SUITES = [
    # 1. Intent & Department Routing
    {
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
        "difficulty": "easy"
    },
    {
        "task": "intent_routing",
        "state": "Our team needs to upgrade our 5-seat plan to an annual enterprise license with SAML SSO and custom SLA.",
        "question": "Route customer request to the correct department",
        "options": [
            {"id": "billing", "description": "Refunds, payment disputes, invoices, subscription cancellations"},
            {"id": "tech_support", "description": "Crash reports, API errors, software installation bugs"},
            {"id": "sales", "description": "Enterprise license agreements and new tier upgrades"},
            {"id": "security", "description": "Vulnerability disclosures and compromised credentials"}
        ],
        "ground_truth": "sales",
        "difficulty": "easy"
    },
    {
        "task": "intent_routing",
        "state": "Stack trace: Segmentation fault at 0x7fff8921a in libssl.so during TLS handshake under heavy concurrent load.",
        "question": "Route customer request to the correct department",
        "options": [
            {"id": "billing", "description": "Refunds, payment disputes, invoices, subscription cancellations"},
            {"id": "tech_support", "description": "Crash reports, API errors, software installation bugs"},
            {"id": "sales", "description": "Enterprise license agreements and new tier upgrades"},
            {"id": "security", "description": "Vulnerability disclosures and compromised credentials"}
        ],
        "ground_truth": "tech_support",
        "difficulty": "easy"
    },
    {
        "task": "intent_routing",
        "state": "We received an unauthenticated password reset email and noticed an unfamiliar login session from IP 198.51.100.44.",
        "question": "Route customer request to the correct department",
        "options": [
            {"id": "billing", "description": "Refunds, payment disputes, invoices, subscription cancellations"},
            {"id": "tech_support", "description": "Crash reports, API errors, software installation bugs"},
            {"id": "sales", "description": "Enterprise license agreements and new tier upgrades"},
            {"id": "security", "description": "Vulnerability disclosures and compromised credentials"}
        ],
        "ground_truth": "security",
        "difficulty": "easy"
    },

    # 2. Strict Abstention Guardrails (__insufficient__)
    {
        "task": "guardrail_abstention",
        "state": "What is the recommended oven temperature and baking duration for homemade artisan sourdough bread?",
        "question": "Route ticket to appropriate corporate IT support tier",
        "options": [
            {"id": "network_engineering", "description": "VPN gateways, subnets, and BGP routing"},
            {"id": "identity_access", "description": "Active Directory, Okta MFA, and password resets"},
            {"id": "hardware_depot", "description": "Laptop provisioning, monitor replacement, and peripheral shipping"}
        ],
        "ground_truth": "__insufficient__",
        "difficulty": "medium"
    },
    {
        "task": "guardrail_abstention",
        "state": "The quick brown fox jumps over the lazy sleeping dog near the quiet river bank.",
        "question": "Classify clinical symptom differential",
        "options": [
            {"id": "appendicitis", "description": "Right lower quadrant abdominal pain and McBurney sign"},
            {"id": "migraine", "description": "Unilateral pulsating headache with photophobia"},
            {"id": "asthma", "description": "Expiratory wheezing and shortness of breath"}
        ],
        "ground_truth": "__insufficient__",
        "difficulty": "medium"
    },
    {
        "task": "guardrail_abstention",
        "state": "Meeting minutes from Tuesday sync: we will follow up on the Q3 roadmap next week.",
        "question": "Route security incident severity",
        "options": [
            {"id": "sev0_critical", "description": "Active remote code execution or data exfiltration underway"},
            {"id": "sev1_high", "description": "Production database failure or authentication outage"},
            {"id": "sev2_medium", "description": "Non-critical API degradation with active workaround"}
        ],
        "ground_truth": "__insufficient__",
        "difficulty": "medium"
    },

    # 3. Clinical & Medical Triage
    {
        "task": "medical_triage",
        "state": "24-year-old patient presents with acute right lower quadrant abdominal pain, McBurney point tenderness, positive Rovsing sign, low-grade fever, and absence of diarrhea.",
        "question": "Classify clinical differential diagnosis",
        "options": [
            {"id": "acute_appendicitis", "description": "Right lower quadrant focal peritonitis, McBurney point tenderness"},
            {"id": "acute_gastroenteritis", "description": "Diffuse abdominal cramping with profuse watery diarrhea and emesis"},
            {"id": "nephrolithiasis", "description": "Colicky flank pain radiating to groin with severe microscopic hematuria"}
        ],
        "ground_truth": "acute_appendicitis",
        "difficulty": "hard"
    },
    {
        "task": "medical_triage",
        "state": "58-year-old male with substernal crushing chest pressure radiating to left jaw, ST-segment elevations in leads V1-V4 on 12-lead ECG, diaphoresis, and elevated troponin.",
        "question": "Classify clinical differential diagnosis",
        "options": [
            {"id": "anterior_stemi", "description": "ST-segment elevation myocardial infarction with troponin release"},
            {"id": "gastroesophageal_reflux", "description": "Postprandial burning retrosternal pyrosis relieved by antacids"},
            {"id": "costochondritis", "description": "Localized parasternal chest wall tenderness reproducible on palpation"}
        ],
        "ground_truth": "anterior_stemi",
        "difficulty": "hard"
    },
    {
        "task": "medical_triage",
        "state": "Patient reports severe unilateral pulsating temporal headache lasting 8 hours, preceded by visual scintillating scotoma aura, accompanied by nausea and marked photophobia.",
        "question": "Classify clinical differential diagnosis",
        "options": [
            {"id": "migraine_with_aura", "description": "Unilateral pulsating headache with preceding visual aura and photophobia"},
            {"id": "tension_headache", "description": "Bilateral band-like pressure headache without aura or photophobia"},
            {"id": "cluster_headache", "description": "Strictly periorbital excruciating stabbing pain with ipsilateral lacrimation"}
        ],
        "ground_truth": "migraine_with_aura",
        "difficulty": "hard"
    },

    # 4. Infrastructure & SRE Incident Response
    {
        "task": "incident_response",
        "state": "CRITICAL ALERT: PostgreSQL primary cluster replication lag exceeded 180 seconds. Read replicas are returning stale data and connection pool is saturated at 100%.",
        "question": "Route infrastructure incident alert",
        "options": [
            {"id": "database_sre", "description": "PostgreSQL replication lag, connection pools, and database failover"},
            {"id": "frontend_team", "description": "Browser JavaScript exceptions, CSS layout issues, hydration errors"},
            {"id": "cdn_edge", "description": "Cloudflare cache hit ratios, TLS certificate expiry, and DNS resolution"},
            {"id": "billing_ops", "description": "Stripe webhook timeouts and invoice reconciliation errors"}
        ],
        "ground_truth": "database_sre",
        "difficulty": "medium"
    },
    {
        "task": "incident_response",
        "state": "Global edge point of presence returning 521 Origin Down across all European endpoints; origin TLS certificate validation failed.",
        "question": "Route infrastructure incident alert",
        "options": [
            {"id": "database_sre", "description": "PostgreSQL replication lag, connection pools, and database failover"},
            {"id": "frontend_team", "description": "Browser JavaScript exceptions, CSS layout issues, hydration errors"},
            {"id": "cdn_edge", "description": "Cloudflare cache hit ratios, TLS certificate expiry, and DNS resolution"},
            {"id": "billing_ops", "description": "Stripe webhook timeouts and invoice reconciliation errors"}
        ],
        "ground_truth": "cdn_edge",
        "difficulty": "medium"
    },

    # 5. Tev1 Format & Temporal Logic
    {
        "task": "tev1_eval",
        "state": "Returns are eligible within 30 days of shipment delivery. Order #4810 was delivered 14 days ago. Item is unopened in original packaging.",
        "question": "Is the order eligible for standard return?",
        "options": [
            {"id": "A", "description": "Yes, order was delivered within the 30-day window"},
            {"id": "B", "description": "No, delivery exceeded allowed timeframe"},
            {"id": "C", "description": "Insufficient information provided"}
        ],
        "ground_truth": "A",
        "difficulty": "medium"
    },
    {
        "task": "tev1_eval",
        "state": "Warranty policy requires annual maintenance inspections. The customer purchased the HVAC unit 4 years ago and has never had an inspection performed.",
        "question": "Is the warranty claim currently active and valid?",
        "options": [
            {"id": "A", "description": "Yes, unit is within standard warranty period"},
            {"id": "B", "description": "No, warranty is voided due to lack of required annual maintenance"},
            {"id": "C", "description": "Insufficient information to evaluate claim"}
        ],
        "ground_truth": "B",
        "difficulty": "medium"
    },

    # 6. Negation & Resolution Scope Handling
    {
        "task": "negation_scope",
        "state": "Patient reports severe chest pain and palpitations, but denies fever, denies cough, and has no history of shortness of breath.",
        "question": "Identify presenting symptom pattern",
        "options": [
            {"id": "cardiac_presentation", "description": "Chest pain, angina, and palpitations"},
            {"id": "respiratory_infection", "description": "Fever, productive cough, and shortness of breath"},
            {"id": "gastrointestinal", "description": "Abdominal cramping, vomiting, and diarrhea"}
        ],
        "ground_truth": "cardiac_presentation",
        "difficulty": "hard"
    },
    {
        "task": "negation_scope",
        "state": "Database service was experiencing high connection errors earlier today, but we rolled back migration #42, restarted pgbouncer, and confirmed all services are now fully resolved and operating normally.",
        "question": "Classify incident status",
        "options": [
            {"id": "active_outage", "description": "Ongoing service outage and active connection errors"},
            {"id": "incident_resolved", "description": "Outage has been mitigated, rolled back, and verified resolved"},
            {"id": "scheduled_maintenance", "description": "Planned maintenance window announced in advance"}
        ],
        "ground_truth": "incident_resolved",
        "difficulty": "hard"
    }
]

def generate_scaled_benchmarks(count: int = 500) -> List[Dict[str, Any]]:
    """Synthesizes realistic permutations, noise variations, and order-invariance test pairs."""
    records = []
    
    # 1. Base Curated Set
    for i, base in enumerate(BENCHMARK_SUITES):
        rec = dict(base)
        rec["id"] = f"zev_curated_{i+1:04d}"
        rec["split"] = "test"
        records.append(rec)
        
        # Create an inverted/permuted duplicate for Order Invariance validation
        perm_rec = dict(base)
        perm_rec["id"] = f"zev_invariance_perm_{i+1:04d}"
        # Reverse options order
        perm_rec["options"] = list(reversed(base["options"]))
        perm_rec["task"] = "order_invariance_eval"
        perm_rec["split"] = "test"
        records.append(perm_rec)

    # 2. Procedurally generated test cases across domains
    domains = [
        ("fintech", ["wire_transfer", "card_fraud", "chargeback", "kyc_verification"]),
        ("healthcare", ["scheduling", "prescription_refill", "lab_results", "billing_insurance"]),
        ("cloud_infra", ["compute_scaling", "storage_snapshot", "iam_permissions", "network_vpc"]),
        ("ecommerce", ["order_tracking", "address_change", "promo_code", "product_defect"])
    ]

    while len(records) < count:
        domain, intents = random.choice(domains)
        chosen_intent = random.choice(intents)
        
        # Build options
        opts = []
        for it in intents:
            readable = it.replace("_", " ").title()
            opts.append({
                "id": it,
                "description": f"Handling customer inquiries and operations regarding {readable}"
            })
        random.shuffle(opts)

        readable_intent = chosen_intent.replace("_", " ")
        state_templates = [
            f"User reached out with a critical question regarding {readable_intent} for account #10{len(records)}.",
            f"Customer submitted priority support ticket: 'Need immediate assistance with our {readable_intent} setup'.",
            f"Automated notification: {readable_intent} action failed validation threshold on transaction reference {random.randint(10000, 99999)}."
        ]

        state = random.choice(state_templates)
        
        records.append({
            "id": f"zev_scaled_{len(records)+1:04d}",
            "task": f"{domain}_routing",
            "state": state,
            "question": f"Route request to the {domain} sub-team",
            "options": opts,
            "ground_truth": chosen_intent,
            "difficulty": "medium",
            "split": "train" if len(records) % 4 != 0 else "test"
        })

    return records

def main():
    parser = argparse.ArgumentParser(description="Export Zev Hugging Face Benchmarks")
    parser.add_argument("--output-dir", default="datasets/zev_benchmarks", help="Output directory")
    parser.add_argument("--count", type=int, default=1200, help="Number of benchmark items to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    benchmarks = generate_scaled_benchmarks(args.count)

    jsonl_path = os.path.join(args.output_dir, "zev_benchmarks.jsonl")
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    train_count = 0
    test_count = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_all, \
         open(train_path, "w", encoding="utf-8") as f_train, \
         open(test_path, "w", encoding="utf-8") as f_test:
        
        for item in benchmarks:
            line = json.dumps(item, ensure_ascii=False) + "\n"
            f_all.write(line)
            if item.get("split") == "train":
                f_train.write(line)
                train_count += 1
            else:
                f_test.write(line)
                test_count += 1

    print(f"✅ Exported {len(benchmarks)} benchmark items to {args.output_dir}/")
    print(f"   • All:   {jsonl_path} ({len(benchmarks)} records)")
    print(f"   • Train: {train_path} ({train_count} records)")
    print(f"   • Test:  {test_path} ({test_count} records)")

if __name__ == "__main__":
    main()
