//! head.pt (torch zip pickle) -> head.safetensors + head_meta.json, without Python or PyTorch.
//!
//! Converts a PyTorch checkpoint archive into standard Safetensors format and JSON metadata.
//!
//! Usage:
//!   cargo run --features candle --bin convert_head -- head.pt out_dir

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, PthTensors, Stack};
use candle_core::Tensor;
use clap::Parser;

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

fn object_to_json(obj: &Object) -> serde_json::Value {
    match obj {
        Object::Unicode(s) => serde_json::Value::String(s.clone()),
        Object::Int(i) => serde_json::Value::Number((*i).into()),
        Object::Long(l) => serde_json::Value::Number((*l).into()),
        Object::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Object::Bool(b) => serde_json::Value::Bool(*b),
        Object::None => serde_json::Value::Null,
        Object::Tuple(items) | Object::List(items) => {
            serde_json::Value::Array(items.iter().map(object_to_json).collect())
        }
        Object::Dict(kvs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in kvs {
                let key_str = match k {
                    Object::Unicode(s) => s.clone(),
                    other => format!("{other:?}"),
                };
                map.insert(key_str, object_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        other => serde_json::Value::String(format!("{other:?}")),
    }
}

fn extract_meta_from_pt<P: AsRef<Path>>(path: P) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))?;

    // Find the data.pkl entry
    let data_pkl_name = zip
        .file_names()
        .find(|name| name.ends_with("data.pkl"))
        .map(|s| s.to_string())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "No data.pkl in archive"))?;

    // Validate archive root against path traversal
    let root = data_pkl_name.split('/').next().unwrap_or("");
    if root.is_empty() || root.contains("..") {
        return Err(format!("Invalid or unsafe archive root: {root}").into());
    }

    let pkl_reader = zip.by_name(&data_pkl_name)?;
    let mut buf_reader = BufReader::new(pkl_reader);
    let mut stack = Stack::empty();
    stack.read_loop(&mut buf_reader)?;
    let root_obj = stack.finalize()?;

    let mut meta_map = serde_json::Map::new();
    if let Object::Dict(entries) = root_obj {
        for (k, v) in entries {
            if let Object::Unicode(ref key_name) = k {
                if key_name != "head" {
                    meta_map.insert(key_name.clone(), object_to_json(&v));
                }
            }
        }
    }

    Ok(serde_json::Value::Object(meta_map))
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

    // 1. Extract metadata from data.pkl
    let meta = extract_meta_from_pt(input)?;

    // 2. Load tensors from "head" key via candle PthTensors
    let pth = PthTensors::new(input, Some("head"))?;
    let tensor_infos = pth.tensor_infos();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let mut tensor_summaries = Vec::new();
    for name in tensor_infos.keys() {
        if let Some(t) = pth.get(name)? {
            let shape = format!("{:?}", t.shape().dims());
            let dtype = format!("{:?}", t.dtype());
            tensor_summaries.push(format!("{name}: ({shape}, {dtype})"));
            tensors.insert(name.clone(), t);
        }
    }

    // 3. Save to head.safetensors
    let safetensors_path = out.join("head.safetensors");
    candle_core::safetensors::save(&tensors, &safetensors_path)?;

    // 4. Save metadata to head_meta.json
    let meta_path = out.join("head_meta.json");
    let meta_file = File::create(&meta_path)?;
    serde_json::to_writer_pretty(meta_file, &meta)?;

    println!("Converted PyTorch checkpoint to Safetensors:");
    println!("  • Safetensors: {} ({} tensors)", safetensors_path.display(), tensors.len());
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
