use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::sync::Arc;
use clap::{Parser, Subcommand};
use zev::{
    create_router, SystemOneRequest, ZevEngine, ZevRequest,
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
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        #[arg(short, long)]
        temperature: Option<f64>,
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
                eprintln!("Error: Input JSON does not match ZevRequest or SystemOneRequest format.");
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

        Commands::Tev1 { file, state, question, options } => {
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

        Commands::Gate { state, instructions, threshold } => {
            let engine = ZevEngine::default();
            let q = zev::Question::Boolean(zev::BooleanQuestion {
                instructions: instructions.clone(),
                true_description: format!("Yes. Confirms {}", instructions),
                false_description: format!("No. Contradicts or lacks {}", instructions),
                policy: zev::Policy { allow_abstain: false, ..Default::default() },
            });
            let (passed, ans) = engine.confidence_gate(&state, q, threshold)?;
            println!("{}", serde_json::json!({
                "passed": passed,
                "confidence": ans.confidence,
                "status": ans.status,
                "decision": ans.decision,
            }));
        }

        Commands::Route { state, routes } => {
            let map: BTreeMap<String, String> = serde_json::from_str(&routes)?;
            let engine = ZevEngine::default();
            let (dest, prob) = engine.route(&state, map)?;
            println!("{}", serde_json::json!({
                "destination": dest,
                "probability": prob,
            }));
        }

        Commands::Serve { host, port, temperature } => {
            let engine = Arc::new(ZevEngine::new(temperature));
            let router = create_router(engine);
            let addr = format!("{}:{}", host, port);
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            println!("Zev API server running on http://{}", addr);
            axum::serve(listener, router).await?;
        }
    }

    Ok(())
}
