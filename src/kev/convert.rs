//! head.pt (a torch zip pickle holding {"head": state_dict, ...meta}) read without Python or PyTorch.

use candle_core::pickle::{Object, PthTensors, Stack};
use candle_core::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

const MAX_OBJECT_DEPTH: usize = 64;

pub fn object_to_json(obj: &Object) -> serde_json::Value {
    object_to_json_depth(obj, 0)
}

fn object_to_json_depth(obj: &Object, depth: usize) -> serde_json::Value {
    if depth > MAX_OBJECT_DEPTH {
        return serde_json::Value::Null;
    }
    match obj {
        Object::Unicode(s) => serde_json::Value::String(s.clone()),
        Object::Int(i) => serde_json::Value::Number((*i).into()),
        Object::Long(l) => serde_json::Value::Number((*l).into()),
        Object::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Object::Bool(b) => serde_json::Value::Bool(*b),
        Object::None => serde_json::Value::Null,
        Object::Tuple(items) | Object::List(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| object_to_json_depth(item, depth + 1))
                .collect(),
        ),
        Object::Dict(kvs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in kvs {
                let key_str = match k {
                    Object::Unicode(s) => s.clone(),
                    other => format!("{other:?}"),
                };
                map.insert(key_str, object_to_json_depth(v, depth + 1));
            }
            serde_json::Value::Object(map)
        }
        other => serde_json::Value::String(format!("{other:?}")),
    }
}

pub fn extract_meta_from_pt<P: AsRef<Path>>(
    path: P,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file))?;

    // Find the data.pkl entry
    let data_pkl_name = zip
        .file_names()
        .find(|name| name.ends_with("data.pkl"))
        .map(|s| s.to_string())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "No data.pkl in archive")
        })?;

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

/// The head's tensors and the checkpoint's other entries (temperature, base, ...).
pub fn head_pt(
    path: &Path,
) -> Result<(HashMap<String, Tensor>, serde_json::Value), Box<dyn std::error::Error>> {
    let meta = extract_meta_from_pt(path)?;
    let pth = PthTensors::new(path, Some("head"))?;
    let mut tensors = HashMap::new();
    for name in pth.tensor_infos().keys() {
        if let Some(t) = pth.get(name)? {
            tensors.insert(name.clone(), t);
        }
    }
    Ok((tensors, meta))
}
