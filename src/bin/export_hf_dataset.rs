//! Zev Decision Engine — Hugging Face Benchmark Dataset Generator & Evaluator.
//!
//! Exports standardized benchmark datasets for zero-token decision engines,
//! intent routers, order-invariance evaluation, and guardrail abstention.
//!
//! Usage:
//!   cargo run --bin export_hf_dataset -- --output-dir datasets/zev_benchmarks --count 1200

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "export_hf_dataset")]
#[command(about = "Export Zev Hugging Face Benchmarks in JSONL format")]
struct Args {
    /// Output directory for benchmark datasets
    #[arg(long, default_value = "datasets/zev_benchmarks")]
    output_dir: PathBuf,

    /// Number of benchmark items to generate
    #[arg(long, default_value_t = 1200)]
    count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkOption {
    id: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkRecord {
    id: String,
    task: String,
    state: String,
    question: String,
    options: Vec<BenchmarkOption>,
    ground_truth: String,
    difficulty: String,
    split: String,
}

struct BenchmarkSeed {
    task: &'static str,
    state: &'static str,
    question: &'static str,
    options: &'static [(&'static str, &'static str)],
    ground_truth: &'static str,
    difficulty: &'static str,
}

const BENCHMARK_SUITES: &[BenchmarkSeed] = &[
    // 1. Intent & Department Routing
    BenchmarkSeed {
        task: "intent_routing",
        state: "Customer requested an immediate full refund for invoice INV-2024-904 because the desktop app crashed on start.",
        question: "Route customer request to the correct department",
        options: &[
            ("billing", "Refunds, payment disputes, invoices, subscription cancellations"),
            ("tech_support", "Crash reports, API errors, software installation bugs"),
            ("sales", "Enterprise license agreements and new tier upgrades"),
            ("security", "Vulnerability disclosures and compromised credentials"),
        ],
        ground_truth: "billing",
        difficulty: "easy",
    },
    BenchmarkSeed {
        task: "intent_routing",
        state: "Our team needs to upgrade our 5-seat plan to an annual enterprise license with SAML SSO and custom SLA.",
        question: "Route customer request to the correct department",
        options: &[
            ("billing", "Refunds, payment disputes, invoices, subscription cancellations"),
            ("tech_support", "Crash reports, API errors, software installation bugs"),
            ("sales", "Enterprise license agreements and new tier upgrades"),
            ("security", "Vulnerability disclosures and compromised credentials"),
        ],
        ground_truth: "sales",
        difficulty: "easy",
    },
    BenchmarkSeed {
        task: "intent_routing",
        state: "Stack trace: Segmentation fault at 0x7fff8921a in libssl.so during TLS handshake under heavy concurrent load.",
        question: "Route customer request to the correct department",
        options: &[
            ("billing", "Refunds, payment disputes, invoices, subscription cancellations"),
            ("tech_support", "Crash reports, API errors, software installation bugs"),
            ("sales", "Enterprise license agreements and new tier upgrades"),
            ("security", "Vulnerability disclosures and compromised credentials"),
        ],
        ground_truth: "tech_support",
        difficulty: "easy",
    },
    BenchmarkSeed {
        task: "intent_routing",
        state: "We received an unauthenticated password reset email and noticed an unfamiliar login session from IP 198.51.100.44.",
        question: "Route customer request to the correct department",
        options: &[
            ("billing", "Refunds, payment disputes, invoices, subscription cancellations"),
            ("tech_support", "Crash reports, API errors, software installation bugs"),
            ("sales", "Enterprise license agreements and new tier upgrades"),
            ("security", "Vulnerability disclosures and compromised credentials"),
        ],
        ground_truth: "security",
        difficulty: "easy",
    },

    // 2. Strict Abstention Guardrails (__insufficient__)
    BenchmarkSeed {
        task: "guardrail_abstention",
        state: "What is the recommended oven temperature and baking duration for homemade artisan sourdough bread?",
        question: "Route ticket to appropriate corporate IT support tier",
        options: &[
            ("network_engineering", "VPN gateways, subnets, and BGP routing"),
            ("identity_access", "Active Directory, Okta MFA, and password resets"),
            ("hardware_depot", "Laptop provisioning, monitor replacement, and peripheral shipping"),
        ],
        ground_truth: "__insufficient__",
        difficulty: "medium",
    },
    BenchmarkSeed {
        task: "guardrail_abstention",
        state: "The quick brown fox jumps over the lazy sleeping dog near the quiet river bank.",
        question: "Classify clinical symptom differential",
        options: &[
            ("appendicitis", "Right lower quadrant abdominal pain and McBurney sign"),
            ("migraine", "Unilateral pulsating headache with photophobia"),
            ("asthma", "Expiratory wheezing and shortness of breath"),
        ],
        ground_truth: "__insufficient__",
        difficulty: "medium",
    },
    BenchmarkSeed {
        task: "guardrail_abstention",
        state: "Meeting minutes from Tuesday sync: we will follow up on the Q3 roadmap next week.",
        question: "Route security incident severity",
        options: &[
            ("sev0_critical", "Active remote code execution or data exfiltration underway"),
            ("sev1_high", "Production database failure or authentication outage"),
            ("sev2_medium", "Non-critical API degradation with active workaround"),
        ],
        ground_truth: "__insufficient__",
        difficulty: "medium",
    },

    // 3. Clinical & Medical Triage
    BenchmarkSeed {
        task: "medical_triage",
        state: "24-year-old patient presents with acute right lower quadrant abdominal pain, McBurney point tenderness, positive Rovsing sign, low-grade fever, and absence of diarrhea.",
        question: "Classify clinical differential diagnosis",
        options: &[
            ("acute_appendicitis", "Right lower quadrant focal peritonitis, McBurney point tenderness"),
            ("acute_gastroenteritis", "Diffuse abdominal cramping with profuse watery diarrhea and emesis"),
            ("nephrolithiasis", "Colicky flank pain radiating to groin with severe microscopic hematuria"),
        ],
        ground_truth: "acute_appendicitis",
        difficulty: "hard",
    },
    BenchmarkSeed {
        task: "medical_triage",
        state: "58-year-old male with substernal crushing chest pressure radiating to left jaw, ST-segment elevations in leads V1-V4 on 12-lead ECG, diaphoresis, and elevated troponin.",
        question: "Classify clinical differential diagnosis",
        options: &[
            ("anterior_stemi", "ST-segment elevation myocardial infarction with troponin release"),
            ("gastroesophageal_reflux", "Postprandial burning retrosternal pyrosis relieved by antacids"),
            ("costochondritis", "Localized parasternal chest wall tenderness reproducible on palpation"),
        ],
        ground_truth: "anterior_stemi",
        difficulty: "hard",
    },
    BenchmarkSeed {
        task: "medical_triage",
        state: "Patient reports severe unilateral pulsating temporal headache lasting 8 hours, preceded by visual scintillating scotoma aura, accompanied by nausea and marked photophobia.",
        question: "Classify clinical differential diagnosis",
        options: &[
            ("migraine_with_aura", "Unilateral pulsating headache with preceding visual aura and photophobia"),
            ("tension_headache", "Bilateral band-like pressure headache without aura or photophobia"),
            ("cluster_headache", "Strictly periorbital excruciating stabbing pain with ipsilateral lacrimation"),
        ],
        ground_truth: "migraine_with_aura",
        difficulty: "hard",
    },

    // 4. Infrastructure & SRE Incident Response
    BenchmarkSeed {
        task: "incident_response",
        state: "CRITICAL ALERT: PostgreSQL primary cluster replication lag exceeded 180 seconds. Read replicas are returning stale data and connection pool is saturated at 100%.",
        question: "Route infrastructure incident alert",
        options: &[
            ("database_sre", "PostgreSQL replication lag, connection pools, and database failover"),
            ("frontend_team", "Browser JavaScript exceptions, CSS layout issues, hydration errors"),
            ("cdn_edge", "Cloudflare cache hit ratios, TLS certificate expiry, and DNS resolution"),
            ("billing_ops", "Stripe webhook timeouts and invoice reconciliation errors"),
        ],
        ground_truth: "database_sre",
        difficulty: "medium",
    },
    BenchmarkSeed {
        task: "incident_response",
        state: "Global edge point of presence returning 521 Origin Down across all European endpoints; origin TLS certificate validation failed.",
        question: "Route infrastructure incident alert",
        options: &[
            ("database_sre", "PostgreSQL replication lag, connection pools, and database failover"),
            ("frontend_team", "Browser JavaScript exceptions, CSS layout issues, hydration errors"),
            ("cdn_edge", "Cloudflare cache hit ratios, TLS certificate expiry, and DNS resolution"),
            ("billing_ops", "Stripe webhook timeouts and invoice reconciliation errors"),
        ],
        ground_truth: "cdn_edge",
        difficulty: "medium",
    },

    // 5. Tev1 Format & Temporal Logic
    BenchmarkSeed {
        task: "tev1_eval",
        state: "Returns are eligible within 30 days of shipment delivery. Order #4810 was delivered 14 days ago. Item is unopened in original packaging.",
        question: "Is the order eligible for standard return?",
        options: &[
            ("A", "Yes, order was delivered within the 30-day window"),
            ("B", "No, delivery exceeded allowed timeframe"),
            ("C", "Insufficient information provided"),
        ],
        ground_truth: "A",
        difficulty: "medium",
    },
    BenchmarkSeed {
        task: "tev1_eval",
        state: "Warranty policy requires annual maintenance inspections. The customer purchased the HVAC unit 4 years ago and has never had an inspection performed.",
        question: "Is the warranty claim currently active and valid?",
        options: &[
            ("A", "Yes, unit is within standard warranty period"),
            ("B", "No, warranty is voided due to lack of required annual maintenance"),
            ("C", "Insufficient information to evaluate claim"),
        ],
        ground_truth: "B",
        difficulty: "medium",
    },

    // 6. Negation & Resolution Scope Handling
    BenchmarkSeed {
        task: "negation_scope",
        state: "Patient reports severe chest pain and palpitations, but denies fever, denies cough, and has no history of shortness of breath.",
        question: "Identify presenting symptom pattern",
        options: &[
            ("cardiac_presentation", "Chest pain, angina, and palpitations"),
            ("respiratory_infection", "Fever, productive cough, and shortness of breath"),
            ("gastrointestinal", "Abdominal cramping, vomiting, and diarrhea"),
        ],
        ground_truth: "cardiac_presentation",
        difficulty: "hard",
    },
    BenchmarkSeed {
        task: "negation_scope",
        state: "Database service was experiencing high connection errors earlier today, but we rolled back migration #42, restarted pgbouncer, and confirmed all services are now fully resolved and operating normally.",
        question: "Classify incident status",
        options: &[
            ("active_outage", "Ongoing service outage and active connection errors"),
            ("incident_resolved", "Outage has been mitigated, rolled back, and verified resolved"),
            ("scheduled_maintenance", "Planned maintenance window announced in advance"),
        ],
        ground_truth: "incident_resolved",
        difficulty: "hard",
    },
];

fn to_title_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn generate_scaled_benchmarks(count: usize) -> Vec<BenchmarkRecord> {
    let mut records = Vec::with_capacity(count);
    let mut rng = rand::rng();

    // 1. Base Curated Set + Order Invariance Permutations
    for (i, seed) in BENCHMARK_SUITES.iter().enumerate() {
        let opts: Vec<BenchmarkOption> = seed
            .options
            .iter()
            .map(|&(id, desc)| BenchmarkOption {
                id: id.to_string(),
                description: desc.to_string(),
            })
            .collect();

        records.push(BenchmarkRecord {
            id: format!("zev_curated_{:04}", i + 1),
            task: seed.task.to_string(),
            state: seed.state.to_string(),
            question: seed.question.to_string(),
            options: opts.clone(),
            ground_truth: seed.ground_truth.to_string(),
            difficulty: seed.difficulty.to_string(),
            split: "test".to_string(),
        });

        // Inverted/permuted duplicate for Order Invariance validation
        let mut reversed_opts = opts;
        reversed_opts.reverse();

        records.push(BenchmarkRecord {
            id: format!("zev_invariance_perm_{:04}", i + 1),
            task: "order_invariance_eval".to_string(),
            state: seed.state.to_string(),
            question: seed.question.to_string(),
            options: reversed_opts,
            ground_truth: seed.ground_truth.to_string(),
            difficulty: seed.difficulty.to_string(),
            split: "test".to_string(),
        });
    }

    // 2. Procedurally generated test cases across domains
    let domains: &[(&str, &[&str])] = &[
        (
            "fintech",
            &["wire_transfer", "card_fraud", "chargeback", "kyc_verification"],
        ),
        (
            "healthcare",
            &[
                "scheduling",
                "prescription_refill",
                "lab_results",
                "billing_insurance",
            ],
        ),
        (
            "cloud_infra",
            &[
                "compute_scaling",
                "storage_snapshot",
                "iam_permissions",
                "network_vpc",
            ],
        ),
        (
            "ecommerce",
            &[
                "order_tracking",
                "address_change",
                "promo_code",
                "product_defect",
            ],
        ),
    ];

    while records.len() < count {
        let (domain, intents) = domains.choose(&mut rng).unwrap();
        let chosen_intent = intents.choose(&mut rng).unwrap();

        let mut opts = Vec::with_capacity(intents.len());
        for it in *intents {
            let readable = to_title_case(it);
            opts.push(BenchmarkOption {
                id: it.to_string(),
                description: format!(
                    "Handling customer inquiries and operations regarding {}",
                    readable
                ),
            });
        }
        opts.shuffle(&mut rng);

        let readable_intent = chosen_intent.replace('_', " ");
        let rec_num = records.len();
        let rand_ref: u32 = rng.random_range(10000..=99999);

        let state_templates = [
            format!(
                "User reached out with a critical question regarding {} for account #10{}.",
                readable_intent, rec_num
            ),
            format!(
                "Customer submitted priority support ticket: 'Need immediate assistance with our {} setup'.",
                readable_intent
            ),
            format!(
                "Automated notification: {} action failed validation threshold on transaction reference {}.",
                readable_intent, rand_ref
            ),
        ];

        let state = state_templates.choose(&mut rng).unwrap().clone();
        let split = if records.len() % 4 != 0 {
            "train"
        } else {
            "test"
        };

        records.push(BenchmarkRecord {
            id: format!("zev_scaled_{:04}", records.len() + 1),
            task: format!("{}_routing", domain),
            state,
            question: format!("Route request to the {} sub-team", domain),
            options: opts,
            ground_truth: chosen_intent.to_string(),
            difficulty: "medium".to_string(),
            split: split.to_string(),
        });
    }

    records
}

fn write_jsonl_dataset<P: AsRef<Path>>(
    output_dir: P,
    records: &[BenchmarkRecord],
) -> std::io::Result<(PathBuf, PathBuf, PathBuf, usize, usize)> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let jsonl_path = output_dir.join("zev_benchmarks.jsonl");
    let train_path = output_dir.join("train.jsonl");
    let test_path = output_dir.join("test.jsonl");

    let mut f_all = BufWriter::new(File::create(&jsonl_path)?);
    let mut f_train = BufWriter::new(File::create(&train_path)?);
    let mut f_test = BufWriter::new(File::create(&test_path)?);

    let mut train_count = 0;
    let mut test_count = 0;

    for item in records {
        let line = serde_json::to_string(item).map_err(|e| std::io::Error::other(e.to_string()))?;
        writeln!(f_all, "{}", line)?;
        if item.split == "train" {
            writeln!(f_train, "{}", line)?;
            train_count += 1;
        } else {
            writeln!(f_test, "{}", line)?;
            test_count += 1;
        }
    }

    f_all.flush()?;
    f_train.flush()?;
    f_test.flush()?;

    Ok((jsonl_path, train_path, test_path, train_count, test_count))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let benchmarks = generate_scaled_benchmarks(args.count);
    let (jsonl_path, train_path, test_path, train_count, test_count) =
        write_jsonl_dataset(&args.output_dir, &benchmarks)?;

    println!(
        "✅ Exported {} benchmark items to {}/",
        benchmarks.len(),
        args.output_dir.display()
    );
    println!(
        "   • All:   {} ({} records)",
        jsonl_path.display(),
        benchmarks.len()
    );
    println!(
        "   • Train: {} ({} records)",
        train_path.display(),
        train_count
    );
    println!(
        "   • Test:  {} ({} records)",
        test_path.display(),
        test_count
    );

    Ok(())
}
