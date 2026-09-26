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
        /// State text
        #[arg(short, long)]
        state: String,

        /// Destinations JSON map, e.g. '{"billing": "Billing & Invoices", "tech": "Tech Support"}'
        #[arg(short, long)]
        routes: String,
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

        /// States kept in the prefix cache
        #[cfg(feature = "candle")]
        #[arg(long, default_value_t = 8)]
        prefix_cache: usize,
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

        Commands::Route { state, routes } => {
            let map: BTreeMap<String, String> = serde_json::from_str(&routes)?;
            let engine = ZevEngine::default();
            let (dest, prob) = engine.route(&state, map)?;
            println!(
                "{}",
                serde_json::json!({
                    "destination": dest,
                    "probability": prob,
                })
            );
        }

        #[cfg(feature = "server")]
        Commands::Serve {
            host,
            port,
            temperature,
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
        } => {
            #[cfg(feature = "candle")]
            let router = if let (Some(kev), Some(base)) = (kev, base) {
                kev_router(
                    &kev,
                    &base,
                    &head.unwrap_or(kev.clone()),
                    &dtype,
                    prefix_cache,
                    temperature,
                )?
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
    }

    Ok(())
}

#[cfg(feature = "candle")]
fn kev_router(
    kev: &std::path::Path,
    base: &std::path::Path,
    head: &std::path::Path,
    dtype: &str,
    prefix_cache: usize,
    temperature: Option<f64>,
) -> Result<axum::Router, Box<dyn std::error::Error>> {
    use zev::kev::{model::Model, serve, Encoder};
    let dev = if cfg!(feature = "cuda") {
        candle_core::Device::new_cuda(0)?
    } else {
        candle_core::Device::Cpu
    };
    #[cfg(feature = "cuda")]
    zev::kev::kernels::tune(&dev)?;
    let dt = match dtype {
        "f32" | "fp32" => candle_core::DType::F32,
        _ => candle_core::DType::BF16,
    };
    let t0 = std::time::Instant::now();
    let model = Model::load(base, kev, head, dt, &dev, temperature)?;
    let tok =
        tokenizers::Tokenizer::from_file(base.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let enc = Encoder::new(tok)?;
    eprintln!(
        "kev: loaded {} in {:.1}s on {:?}",
        kev.display(),
        t0.elapsed().as_secs_f64(),
        dev
    );
    let card = serde_json::json!({
        "description": format!("Kev pointer head on {}, Candle backend, temperature {:.2}", base.display(), model.temperature),
        "release_date": "2026-09-25", "backend": "candle", "dtype": dtype, "temperature": model.temperature,
    });
    Ok(serve::router(serve::spawn(model, enc, prefix_cache, card)))
}
