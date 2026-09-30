//! Full-vocabulary logits of a backbone at chosen positions, for parity checks against transformers.
//!
//!   cargo run --release --features cuda --example kev_logits -- <checkpoint dir> <bf16|f32> <prompts.jsonl> <out dir>
//!
//! Each prompts line: {"ids": [token ids], "positions": [positions]}. Writes <out dir>/<line>.f32: little-endian f32
//! [positions, vocab], the final-norm hidden state times the output rows (lm_head or tied embedding) in the model
//! dtype, with the final logit softcap when the config has one (as the transformers CausalLM head computes them).

use candle_core::{DType, Device};
use std::io::{BufRead, Write};
use zev::kev::kernels::Pack;
use zev::kev::model::Model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!(
            "Usage: {} <checkpoint dir> <bf16|f32> <prompts.jsonl> <out dir>",
            a.first().map_or("kev_logits", |s| s.as_str())
        );
        std::process::exit(1);
    }
    let dt = if a[2] == "f32" {
        DType::F32
    } else {
        DType::BF16
    };
    let dev = if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    #[cfg(feature = "cuda")]
    zev::kev::kernels::tune(&dev)?;
    let m = Model::load(std::path::Path::new(&a[1]), None, dt, &dev)?;
    std::fs::create_dir_all(&a[4])?;
    for (i, line) in std::io::BufReader::new(std::fs::File::open(&a[3])?)
        .lines()
        .enumerate()
    {
        let v: serde_json::Value = serde_json::from_str(&line?)?;
        let ids: Vec<u32> = serde_json::from_value(v["ids"].clone())?;
        let pos: Vec<u32> = serde_json::from_value(v["positions"].clone())?;
        let pack = Pack::new(vec![(ids.len(), None, None)], &dev)?;
        let h = m.forward(&pack, &ids, &pos)?;
        let mut lg = h
            .to_dtype(dt)?
            .matmul(&m.out_rows().t()?)?
            .to_dtype(DType::F32)?;
        if m.final_softcap > 0.0 {
            lg = ((lg / m.final_softcap)?.tanh()? * m.final_softcap)?;
        }
        let flat: Vec<f32> = lg.flatten_all()?.to_vec1()?;
        let mut f = std::fs::File::create(format!("{}/{i}.f32", a[4]))?;
        f.write_all(
            &flat
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        eprintln!("prompt {i}: {} tokens, {} positions", ids.len(), pos.len());
    }
    Ok(())
}
