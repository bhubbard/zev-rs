//! head.pt (torch zip pickle) -> head.safetensors + head_meta.json, without Python or PyTorch.
//!
//! Converts a PyTorch checkpoint archive into standard Safetensors format and JSON metadata.
//!
//! Usage:
//!   cargo run --features candle --bin convert_head -- head.pt out_dir

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use candle_core::Tensor;
use clap::Parser;
use zev::kev::convert::head_pt;

#[derive(Parser, Debug)]
#[command(name = "convert_head")]
#[command(about = "Convert PyTorch checkpoint (head.pt) to head.safetensors and head_meta.json")]
struct Args {
    /// Path to input PyTorch checkpoint file (e.g. head.pt)
    #[arg(value_name = "INPUT_PT")]
    input: PathBuf,

    /// Output directory for head.safetensors and head_meta.json
    #[arg(value_name = "OUTPUT_DIR")]
    output_dir: PathBuf,
}

fn convert_checkpoint<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_dir: P2,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = input_path.as_ref();
    let out = output_dir.as_ref();

    if !input.exists() {
        return Err(format!("Input file does not exist: {}", input.display()).into());
    }

    fs::create_dir_all(out)?;

    // 1-2. The metadata in data.pkl and the tensors under its "head" key
    let (tensors, meta): (HashMap<String, Tensor>, _) = head_pt(input)?;
    let tensor_summaries: Vec<String> = tensors
        .iter()
        .map(|(name, t)| format!("{name}: ({:?}, {:?})", t.shape().dims(), t.dtype()))
        .collect();

    // 3. Save to head.safetensors
    let safetensors_path = out.join("head.safetensors");
    candle_core::safetensors::save(&tensors, &safetensors_path)?;

    // 4. Save metadata to head_meta.json
    let meta_path = out.join("head_meta.json");
    let meta_file = File::create(&meta_path)?;
    serde_json::to_writer_pretty(meta_file, &meta)?;

    println!("Converted PyTorch checkpoint to Safetensors:");
    println!(
        "  • Safetensors: {} ({} tensors)",
        safetensors_path.display(),
        tensors.len()
    );
    for summary in &tensor_summaries {
        println!("    - {summary}");
    }
    println!("  • Metadata:    {}", meta_path.display());

    // Print summary of key meta attributes
    if let serde_json::Value::Object(ref map) = meta {
        let keys_of_interest = [
            "base",
            "base_revision",
            "lora",
            "head_dim",
            "option_isolation",
            "special_embeddings",
            "weights_dtype",
            "temperature",
        ];
        let mut key_vals = Vec::new();
        for key in keys_of_interest {
            if let Some(v) = map.get(key) {
                key_vals.push(format!("{key}: {v}"));
            }
        }
        if !key_vals.is_empty() {
            println!("  • Key Meta:    {}", key_vals.join(", "));
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    convert_checkpoint(&args.input, &args.output_dir)?;
    Ok(())
}
