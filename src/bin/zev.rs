use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
#[cfg(feature = "server")]
use std::sync::Arc;
#[cfg(feature = "server")]
use zev::create_router;
use zev::{
    SystemOneRequest, TabularBatch, TabularEngine, TabularFilterPredicate, TabularRow, ZevEngine,
    ZevRequest,
};

#[derive(Parser)]
#[command(name = "zev")]
#[command(about = "Zev: High-performance, 100% order-invariant zero-token LLM decision engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evaluate a decision request from native ZevRequest JSON file or stdin
    Decide {
        /// Path to JSON request file (stdin if omitted)
        #[arg(short, long)]
        file: Option<String>,

        /// Temperature override (default: 2.17908)
        #[arg(short, long)]
        temperature: Option<f64>,
    },

    /// Evaluate a TypeSafe SystemOneRequest JSON file or stdin
    Systemone {
        /// Path to JSON request file (stdin if omitted)
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Run confidence gating on an input state
    Gate {
        /// State text
        #[arg(short, long)]
        state: String,

        /// Instructions
        #[arg(short, long)]
        instructions: String,

        /// Confidence threshold [0.0 - 1.0]
        #[arg(short, long, default_value_t = 0.7)]
        threshold: f64,
    },

    /// Route state text to destinations
    Route {
        /// State text context (or pass via --file or stdin)
        #[arg(short, long)]
        state: Option<String>,

        /// Path to file containing state text (or '-' for stdin)
        #[arg(short, long)]
        file: Option<std::path::PathBuf>,

        /// Destinations JSON map, e.g. '{"billing": "Billing & Invoices", "tech": "Tech Support"}'
        #[arg(short, long)]
        routes: String,

        /// Output full probability distribution over all destination candidates
        #[arg(short, long)]
        distribution: bool,

        /// Speculative fallback backend ("gemma", "apfel", "clm", "none")
        #[arg(short, long)]
        backend: Option<String>,

        /// Minimum confidence threshold [0.0 - 1.0]. If top probability is below this, cascades to --backend
        #[arg(long, default_value_t = 0.0)]
        min_confidence: f64,
    },

    /// Evaluate a Tev1 decision request from JSON, raw prompt text, or CLI flags
    Tev1 {
        /// Path to Tev1 JSON request file or prompt text (stdin if omitted)
        #[arg(short, long)]
        file: Option<String>,

        /// State context (when specifying via CLI flags)
        #[arg(short, long)]
        state: Option<String>,

        /// Question prompt (when specifying via CLI flags)
        #[arg(short, long)]
        question: Option<String>,

        /// Comma-separated options (e.g. "A: Yes, B: No, C: Maybe")
        #[arg(short, long)]
        options: Option<String>,
    },

    /// Start the Zev HTTP API server
    #[cfg(feature = "server")]
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        #[arg(short, long)]
        temperature: Option<f64>,

        /// Serve a model on the Candle backend: a checkpoint dir or a Hub id[@revision] (repeat to serve several; the
        /// request's `model` picks one, the first is the default). The family (Kev, decider, JevK5) is detected.
        #[cfg(feature = "candle")]
        #[arg(long = "model")]
        models: Vec<String>,

        /// Serve a Kev checkpoint on the Candle backend: the adapter snapshot (adapter_model.safetensors,
        /// adapter_config.json, head.safetensors + head_meta.json converted from head.pt)
        #[cfg(feature = "candle")]
        #[arg(long)]
        kev: Option<std::path::PathBuf>,

        /// The base model snapshot (config.json, tokenizer.json, *.safetensors)
        #[cfg(feature = "candle")]
        #[arg(long)]
        base: Option<std::path::PathBuf>,

        /// Directory holding head.safetensors + head_meta.json (default: --kev)
        #[cfg(feature = "candle")]
        #[arg(long)]
        head: Option<std::path::PathBuf>,

        /// bf16 (kev.serve's default) or f32
        #[cfg(feature = "candle")]
        #[arg(long, default_value = "bf16")]
        dtype: String,

        /// States kept in the prefix cache (per model worker)
        #[cfg(feature = "candle")]
        #[arg(long, default_value_t = 8)]
        prefix_cache: usize,

        /// Token budget of one batched pass (a larger single request runs alone)
        #[cfg(feature = "candle")]
        #[arg(long, default_value_t = 8192)]
        pass_tokens: usize,

        /// CUDA devices to run a model worker on, e.g. 0,1 (one per GPU) or 0,0 (two instances on GPU 0)
        #[cfg(feature = "candle")]
        #[arg(long, default_value = "0")]
        devices: String,
    },

    /// Execute batch tabular decisions (filter, score, route) over a JSONL file
    Batch {
        /// Path to JSONL file with rows (each row: {"id": "...", "text": "..."})
        #[arg(short, long)]
        file: String,

        /// Batch mode: 'filter', 'score', or 'route'
        #[arg(short, long, default_value = "filter")]
        mode: String,

        /// Instruction text for filter/score
        #[arg(short, long, default_value = "Evaluate row")]
        instructions: String,

        /// Positive criterion for filter/score
        #[arg(long, default_value = "Satisfies condition")]
        positive: String,

        /// Negative criterion for filter/score
        #[arg(long, default_value = "Does not satisfy condition")]
        negative: String,

        /// Threshold probability (default 0.5)
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f64,

        /// Routes JSON map (required for 'route' mode)
        #[arg(short, long)]
        routes: Option<String>,
    },

    /// Semantic codebase search over AST declarations using zero-token Zev decision engine
    Grep {
        /// Semantic query describing the logic, concept, or symbol to find
        query: String,

        /// Root directory path to search (default: ".")
        #[arg(default_value = ".")]
        path: std::path::PathBuf,

        /// Output results as JSON for agents and programmatic consumers
        #[arg(long)]
        json: bool,

        /// Maximum number of declarations to return
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

        /// Minimum relevance threshold [0.0 - 1.0]
        #[arg(short, long, default_value_t = 0.3)]
        threshold: f64,

        /// Maximum number of files to inspect
        #[arg(long, default_value_t = 1000)]
        max_files: usize,
    },

    /// Start the Model Context Protocol (MCP) JSON-RPC stdio server
    Mcp,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Decide { file, temperature } => {
            let content = match file {
                Some(f) => fs::read_to_string(f)?,
                None => {
                    let mut buf = String::new();
                    io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };

            let engine = ZevEngine::new(temperature);

            // Attempt native ZevRequest first, fallback to SystemOneRequest
            if let Ok(native_req) = serde_json::from_str::<ZevRequest>(&content) {
                let resp = engine.evaluate(&native_req)?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            } else if let Ok(sys1_req) = serde_json::from_str::<SystemOneRequest>(&content) {
                let resp = engine.evaluate_system_one(&sys1_req)?;
                println!("{}", serde_json::to_string_pretty(&resp)?);
            } else {
                eprintln!(
                    "Error: Input JSON does not match ZevRequest or SystemOneRequest format."
                );
                std::process::exit(1);
            }
        }

        Commands::Systemone { file } => {
            let content = match file {
                Some(f) => fs::read_to_string(f)?,
                None => {
                    let mut buf = String::new();
                    io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };

            let req: SystemOneRequest = serde_json::from_str(&content)?;
            let engine = ZevEngine::default();
            let resp = engine.evaluate_system_one(&req)?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        Commands::Tev1 {
            file,
            state,
            question,
            options,
        } => {
            let engine = ZevEngine::default();
            let req = if let (Some(s), Some(q), Some(opts_str)) = (state, question, options) {
                let opts: Vec<String> = if opts_str.starts_with('[') {
                    serde_json::from_str(&opts_str)?
                } else {
                    opts_str.split(',').map(|s| s.trim().to_string()).collect()
                };
                zev::Tev1Request {
                    state: s,
                    question: q,
                    options: opts,
                    model: None,
                }
            } else {
                let content = match file {
                    Some(f) => fs::read_to_string(f)?,
                    None => {
                        let mut buf = String::new();
                        io::stdin().read_to_string(&mut buf)?;
                        buf
                    }
                };

                if let Ok(json_req) = serde_json::from_str::<zev::Tev1Request>(&content) {
                    json_req
                } else {
                    zev::Tev1Request::parse_prompt(&content)?
                }
            };

            let resp = engine.evaluate_tev1(&req)?;
            println!("{}", serde_json::to_string_pretty(&resp)?);
        }

        Commands::Gate {
            state,
            instructions,
            threshold,
        } => {
            let engine = ZevEngine::default();
            let q = zev::Question::Boolean(zev::BooleanQuestion {
                instructions: instructions.clone(),
                true_description: format!("Yes. Confirms {}", instructions),
                false_description: format!("No. Contradicts or lacks {}", instructions),
                policy: zev::Policy {
                    allow_abstain: false,
                    ..Default::default()
                },
            });
            let (passed, ans) = engine.confidence_gate(&state, q, threshold)?;
            println!(
                "{}",
                serde_json::json!({
                    "passed": passed,
                    "confidence": ans.confidence,
                    "status": ans.status,
                    "decision": ans.decision,
                })
            );
        }

        Commands::Route {
            state,
            file,
            routes,
            distribution,
            backend,
            min_confidence,
        } => {
            if let Some(b) = backend {
                if (b == "apfel" || b == "neural") && !cfg!(feature = "neural") {
                    eprintln!(
                        "Warning: fallback backend '{}' requested, but zev was compiled without '--features neural'. Reverting to default SIMD heuristics.",
                        b
                    );
                }
                std::env::set_var("ZEV_FALLBACK", b);
            } else if min_confidence > 0.0 && std::env::var("ZEV_FALLBACK").is_err() {
                #[cfg(target_os = "macos")]
                std::env::set_var("ZEV_FALLBACK", "apfel");
                #[cfg(not(target_os = "macos"))]
                std::env::set_var("ZEV_FALLBACK", "gemma");
            }

            if min_confidence > 0.0 {
                std::env::set_var("ZEV_FALLBACK_CONFIDENCE", min_confidence.to_string());
            }

            let text = match (state, file) {
                (Some(s), _) => s,
                (None, Some(f)) => {
                    if f.as_os_str() == "-" {
                        let mut buf = String::new();
                        io::stdin().read_to_string(&mut buf)?;
                        buf
                    } else {
                        fs::read_to_string(f)?
                    }
                }
                (None, None) => {
                    let mut buf = String::new();
                    io::stdin().read_to_string(&mut buf)?;
                    buf
                }
            };

            #[derive(serde::Deserialize)]
            #[serde(untagged)]
            enum RouteTarget {
                Simple(String),
                Detailed {
                    description: String,
                    #[serde(default)]
                    negative: Option<String>,
                },
            }

            let raw_map: BTreeMap<String, RouteTarget> = serde_json::from_str(&routes)?;
            let map: BTreeMap<String, String> = raw_map
                .into_iter()
                .map(|(k, v)| {
                    let desc = match v {
                        RouteTarget::Simple(s) => s,
                        RouteTarget::Detailed {
                            description,
                            negative,
                        } => {
                            if let Some(neg) = negative {
                                format!("{description}. Exclude: {neg}")
                            } else {
                                description
                            }
                        }
                    };
                    (k, desc)
                })
                .collect();

            let engine = ZevEngine::default();
            let (dest, prob, dist) = engine.route_with_distribution(&text, map)?;

            if distribution {
                println!(
                    "{}",
                    serde_json::json!({
                        "destination": dest,
                        "probability": prob,
                        "distribution": dist,
                    })
                );
            } else {
                println!(
                    "{}",
                    serde_json::json!({
                        "destination": dest,
                        "probability": prob,
                    })
                );
            }
        }

        #[cfg(feature = "server")]
        Commands::Serve {
            host,
            port,
            temperature,
            #[cfg(feature = "candle")]
            models,
            #[cfg(feature = "candle")]
            kev,
            #[cfg(feature = "candle")]
            base,
            #[cfg(feature = "candle")]
            head,
            #[cfg(feature = "candle")]
            dtype,
            #[cfg(feature = "candle")]
            prefix_cache,
            #[cfg(feature = "candle")]
            pass_tokens,
            #[cfg(feature = "candle")]
            devices,
        } => {
            #[cfg(feature = "candle")]
            let router = if !models.is_empty() || kev.is_some() {
                use zev::kev::load;
                let o = load::LoadOpts {
                    dtype: match dtype.as_str() {
                        "f32" | "fp32" => candle_core::DType::F32,
                        _ => candle_core::DType::BF16,
                    },
                    devices: devices
                        .split(',')
                        .filter(|d| !d.is_empty())
                        .map(|d| d.trim().parse())
                        .collect::<Result<_, _>>()?,
                    serve: zev::kev::serve::Opts {
                        prefix_cache,
                        pass_tokens,
                    },
                    temperature,
                };
                if kev.is_some() && base.is_none() {
                    return Err(
                        "zev serve: --base <dir> is required when --kev is specified".into(),
                    );
                }
                let mut served = Vec::new();
                if let (Some(kev), Some(base)) = (&kev, &base) {
                    served.push(load::load_kev(kev, base, head.as_ref().unwrap_or(kev), &o)?);
                }
                for m in &models {
                    served.push(load::load(m, &o)?);
                }
                if served.is_empty() {
                    return Err(
                        "zev serve: no models loaded. Specify --model <dir|hub_id> or --kev and --base".into(),
                    );
                }
                zev::kev::serve::router(served)
            } else {
                create_router(Arc::new(ZevEngine::new(temperature)))
            };
            #[cfg(not(feature = "candle"))]
            let router = create_router(Arc::new(ZevEngine::new(temperature)));
            let addr = format!("{}:{}", host, port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            println!("Zev API server running on http://{}", addr);
            axum::serve(listener, router).await?;
        }

        Commands::Batch {
            file,
            mode,
            instructions,
            positive,
            negative,
            threshold,
            routes,
        } => {
            let content = fs::read_to_string(&file)?;
            let mut rows = Vec::new();
            for (line_idx, line) in content.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    let id = val
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&line_idx.to_string())
                        .to_string();
                    let text = val
                        .get("text")
                        .and_then(|v| v.as_str())
                        .or_else(|| val.get("body").and_then(|v| v.as_str()))
                        .unwrap_or(trimmed)
                        .to_string();
                    rows.push(TabularRow::new(id, text));
                }
            }

            let batch = TabularBatch::new(rows);
            let engine = TabularEngine::default();

            match mode.as_str() {
                "filter" => {
                    let pred = TabularFilterPredicate::new(instructions, positive, negative)
                        .with_threshold(threshold);
                    let (survivors, report) = engine.filter_batch(&batch, &pred)?;
                    println!(
                        "Filtered {} -> {} rows in {:.2} ms ({:.0} rows/s)",
                        report.input_rows,
                        report.output_rows,
                        report.elapsed_microseconds as f64 / 1000.0,
                        report.rows_per_second
                    );
                    println!("{}", serde_json::to_string_pretty(&survivors)?);
                }
                "score" => {
                    let (scores, report) =
                        engine.score_batch(&batch, &instructions, &positive, &negative)?;
                    println!(
                        "Scored {} rows in {:.2} ms ({:.0} rows/s)",
                        report.input_rows,
                        report.elapsed_microseconds as f64 / 1000.0,
                        report.rows_per_second
                    );
                    println!("{}", serde_json::to_string_pretty(&scores)?);
                }
                "route" => {
                    let routes_str = routes.ok_or_else(|| {
                        zev::ZevError::InvalidRequest(
                            "Missing --routes JSON map for route mode".into(),
                        )
                    })?;
                    let routes_map: BTreeMap<String, String> = serde_json::from_str(&routes_str)?;
                    let (routed, report) = engine.route_batch(&batch, &routes_map)?;
                    println!(
                        "Routed {} rows in {:.2} ms ({:.0} rows/s)",
                        report.input_rows,
                        report.elapsed_microseconds as f64 / 1000.0,
                        report.rows_per_second
                    );
                    println!("{}", serde_json::to_string_pretty(&routed)?);
                }
                other => {
                    eprintln!(
                        "Unknown batch mode: '{}'. Supported modes: 'filter', 'score', 'route'",
                        other
                    );
                }
            }
        }

        Commands::Grep {
            query,
            path,
            json,
            limit,
            threshold,
            max_files,
        } => {
            let engine = ZevEngine::default();
            let opts = zev::GrepOptions {
                query,
                path,
                limit,
                threshold,
                json,
                include_evidence: true,
                max_files,
            };

            let report = zev::run_grep(&engine, &opts)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", zev::format_human_report(&report));
            }
        }

        Commands::Mcp => {
            zev::mcp::run_stdio_server().await?;
        }
    }

    Ok(())
}
