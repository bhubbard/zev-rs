//! Readouts: from the final-norm hidden states a pass picked, the probabilities of each question.

use super::model::{read_json, Model};
use super::RowSpec;
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

/// Kev's pointer head (kev `model.py` PointerHead): per question, softmax over options of
/// k(h_opt) . q(h_decide) / sqrt(dp) / temperature. Weights [q; k] [2 * dp, hidden] f32.
pub struct PointerHead {
    w: Tensor,
    b: Tensor,
    pub dp: usize,
    pub temperature: f64,
}

impl PointerHead {
    /// `dir`: head.safetensors + head_meta.json (convert_head's output), pointer_head.safetensors (+ config.json
    /// temperature, NeoHorse), or head.pt itself. `temperature` overrides the checkpoint's.
    pub fn load(dir: &Path, temperature: Option<f64>, dev: &Device) -> Result<Self> {
        let st = ["head.safetensors", "pointer_head.safetensors"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists());
        let (head, meta) = match st {
            Some(f) => (
                candle_core::safetensors::load(f, dev)?,
                read_json(&dir.join("head_meta.json"))
                    .or_else(|_| read_json(&dir.join("config.json")))
                    .unwrap_or_default(),
            ),
            None if dir.join("head.pt").exists() => {
                let (t, meta) = super::convert::head_pt(&dir.join("head.pt"))
                    .map_err(|e| candle_core::Error::Msg(format!("head.pt: {e}")))?;
                (
                    t.into_iter()
                        .map(|(k, v)| Ok((k, v.to_device(dev)?)))
                        .collect::<Result<_>>()?,
                    meta,
                )
            }
            None => candle_core::bail!(
                "no head.safetensors, pointer_head.safetensors or head.pt in {}",
                dir.display()
            ),
        };
        let hw = |k: &str| {
            head.get(k)
                .map(|t| t.to_dtype(DType::F32))
                .ok_or_else(|| candle_core::Error::Msg(format!("head lacks {k}")))?
        };
        let t = temperature.unwrap_or_else(|| meta["temperature"].as_f64().unwrap_or(1.0));
        Ok(Self {
            w: Tensor::cat(&[hw("q.weight")?, hw("k.weight")?], 0)?,
            b: Tensor::cat(&[hw("q.bias")?, hw("k.bias")?], 0)?,
            dp: hw("q.bias")?.dims1()?,
            temperature: if t.is_finite() && t > 1e-4 { t } else { 1.0 },
        })
    }

    /// x: f32 [P, hidden], per question <decide> then its k options. -> option probabilities per question.
    pub fn probs(&self, x: &Tensor, ks: &[usize]) -> Result<Vec<Vec<f64>>> {
        let proj: Vec<Vec<f32>> = x.matmul(&self.w.t()?)?.broadcast_add(&self.b)?.to_vec2()?;
        let dp = self.dp;
        let scale = 1.0 / (dp as f64).sqrt();
        let mut at = 0;
        let mut out = Vec::with_capacity(ks.len());
        for &k in ks {
            if at + 1 + k > proj.len() {
                candle_core::bail!(
                    "pointer head: need {} rows, have {}",
                    at + 1 + k,
                    proj.len()
                );
            }
            let q = &proj[at][..dp];
            let z: Vec<f64> = (0..k)
                .map(|j| {
                    let kv = &proj[at + 1 + j][dp..];
                    let dot: f64 = q.iter().zip(kv).map(|(a, b)| *a as f64 * *b as f64).sum();
                    dot * scale / self.temperature
                })
                .collect();
            out.push(softmax(&z));
            at += 1 + k;
        }
        Ok(out)
    }
}

pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    let norm = if s.is_finite() && s > 0.0 { s } else { 1.0 };
    e.into_iter().map(|v| v / norm).collect()
}

/// Label logits (R-LBL): at each answer slot, h . W[label] over that slot's label tokens, with W the output rows
/// (tied embedding or lm_head) in the model dtype and the product rounded to it (the reference's
/// `F.linear(h, W[labels]).float()`) or kept in f32, then an optional softcap c * tanh(x / c). Temperature and
/// softmax belong to the entrant.
pub struct LabelHead {
    pub softcap: f64,
    /// Dot products in f32 (llama.cpp's selected head, Winnow) instead of rounded to the model dtype (the transformers
    /// references): near a softcap of 30 a bf16 logit moves in steps of 0.125.
    pub f32: bool,
}

pub enum Readout {
    Pointer(PointerHead),
    Labels(LabelHead),
}

impl Readout {
    /// Per row, per question group: the pointer head's probabilities, or each slot's label logits.
    pub fn run(&self, m: &Model, x: &Tensor, rows: &[&RowSpec]) -> Result<Vec<Vec<Vec<f64>>>> {
        match self {
            Readout::Pointer(h) => {
                let ks: Vec<usize> = rows.iter().map(|r| r.picks.len() - 1).collect();
                Ok(h.probs(x, &ks)?.into_iter().map(|p| vec![p]).collect())
            }
            Readout::Labels(l) => {
                let mut cols: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                let mut uniq: Vec<u32> = Vec::new();
                for r in rows {
                    for id in r.labels.iter().flat_map(|l| l.iter()) {
                        cols.entry(*id).or_insert_with(|| {
                            uniq.push(*id);
                            uniq.len() - 1
                        });
                    }
                }
                let w = m
                    .out_rows()
                    .index_select(&Tensor::new(uniq.as_slice(), &m.dev)?, 0)?;
                let lg: Vec<Vec<f32>> = if l.f32 {
                    x.matmul(&w.to_dtype(DType::F32)?.t()?)?.to_vec2()?
                } else {
                    x.to_dtype(m.dt)?
                        .matmul(&w.t()?)?
                        .to_dtype(DType::F32)?
                        .to_vec2()?
                };
                let cap = |v: f32| {
                    let v = v as f64;
                    if l.softcap > 0.0 {
                        l.softcap * (v / l.softcap).tanh()
                    } else {
                        v
                    }
                };
                let mut at = 0;
                let mut out = Vec::with_capacity(rows.len());
                for r in rows {
                    let mut per = Vec::with_capacity(r.labels.len());
                    for lab in &r.labels {
                        per.push(lab.iter().map(|id| cap(lg[at][cols[id]])).collect());
                        at += 1;
                    }
                    out.push(per);
                }
                Ok(out)
            }
        }
    }
}
