//! Decoder backbones on Candle over packed passes (kernels.rs), chosen by config.json `model_type`:
//! - `qwen3_5` / `qwen3_5_text`: the Qwen3.5 hybrid (Gated DeltaNet + gated full attention), transformers 5.17
//!   `modeling_qwen3_5.py`;
//! - `qwen3`: dense Qwen3 (`modeling_qwen3.py`);
//! - `gemma3` / `gemma3_text`: Gemma 3 (sandwich norms with 1 + w, sliding and global layers, scaled embeddings);
//! - `gemma4` / `gemma4_text` / `gemma4_unified(_text)`: Gemma 4 dense text (`modeling_gemma4.py`: plain-w norms, value
//!   norm, attention scale 1, k = v on global layers with their own head size, proportional RoPE, layer scalars).
//!
//! An optional LoRA is merged on load. Precision follows the bf16 reference paths: weights `round(W + scale * B @ A)`
//! (delta in f32), the residual stream and projections in the model dtype, norms, attention and the DeltaNet
//! recurrence in f32 math, rounded where the reference rounds.

use super::kernels::{self, Act, AttnSpec, CacheShape, GdnSpec, NormMode, Pack, DK, DV};
use candle_core::quantized::gguf_file;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Result, Tensor};
use serde_json::Value;
use std::path::Path;

struct Gdn {
    spec: GdnSpec,
    lg: usize,
    proj: Proj,      // in_proj_qkv | in_proj_z | in_proj_b | in_proj_a
    conv_w: Tensor,  // taps [K, C] f32
    a_neg: Tensor,   // -exp(A_log) [HV] f32
    dt_bias: Tensor, // [HV] f32
    norm_w: Tensor,  // [DV] f32
    out: Proj,
}

struct Attn {
    spec: AttnSpec,
    qkv: Proj, // q (with its gate) | k | v
    o: Proj,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Layer {
    in_norm: Tensor,
    post_mix: Option<Tensor>, // Gemma: norm of the mixer output before the residual add
    pre_mlp: Tensor,
    post_mlp: Option<Tensor>, // Gemma: norm of the MLP output before the residual add
    scalar: f64,              // Gemma 4 layer_scalar (1 elsewhere)
    mixer: Mixer,
    gate_up: Proj,
    down: Proj,
}

pub struct Model {
    pub model_type: String,
    pub hidden: usize,
    pub dt: DType,
    pub dev: Device,
    pub embed: Tensor,           // [vocab, hidden] model dtype
    pub lm_head: Option<Tensor>, // untied output rows [vocab, hidden] (None: tied to embed)
    pub final_softcap: f64,      // Gemma 4: 30 (0 = none)
    embed_scale: Option<f64>,
    eps: f64,
    norm_mode: NormMode,
    act: Act,
    round_act: bool,
    layers: Vec<Layer>,
    norm: Tensor,
    pub cache: CacheShape,
}

/// A projection weight [out, in]: dense in the model dtype, or GGUF Q8_0 blocks kept on the device and dequantized to
/// the model dtype for each use (a 12B model fits a 24 GB card at 8 bits; one dequantization per layer and pass).
pub enum Proj {
    Dense(Tensor),
    Q8 { q: Tensor, out: usize, inn: usize },
}

impl Proj {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(w) => x.matmul(&w.t()?),
            Proj::Q8 { q, out, inn } => {
                x.matmul(&kernels::dequant_q8(q, *out, *inn, x.dtype())?.t()?)
            }
        }
    }
    fn rows(&self) -> usize {
        match self {
            Proj::Dense(w) => w.dim(0).unwrap_or(0),
            Proj::Q8 { out, .. } => *out,
        }
    }
    fn cat(ps: Vec<Proj>) -> Result<Proj> {
        if ps.iter().all(|p| matches!(p, Proj::Dense(_))) {
            let ws: Vec<Tensor> = ps
                .into_iter()
                .map(|p| {
                    if let Proj::Dense(w) = p {
                        w
                    } else {
                        unreachable!()
                    }
                })
                .collect();
            return Ok(Proj::Dense(Tensor::cat(&ws, 0)?));
        }
        let mut qs = Vec::new();
        let (mut out, mut inn) = (0, 0);
        for p in ps {
            let Proj::Q8 { q, out: o, inn: i } = p else {
                candle_core::bail!("cannot concatenate dense and Q8_0 projections")
            };
            if inn != 0 && i != inn {
                candle_core::bail!("concatenated projections disagree on the input width");
            }
            (out, inn) = (out + o, i);
            qs.push(q);
        }
        Ok(Proj::Q8 {
            q: Tensor::cat(&qs, 0)?,
            out,
            inn,
        })
    }
}

/// Where the tensors come from: safetensors shards, or one GGUF file (tensors renamed from the Hugging Face names).
enum Source {
    St(MmapedSafetensors),
    Gguf {
        content: gguf_file::Content,
        file: std::sync::Mutex<std::fs::File>,
    },
}

/// The GGUF name of a Gemma 4 text tensor (llama.cpp convert_hf_to_gguf, 911f6cdc; no norm shift for Gemma 4).
fn gguf_name(hf: &str) -> Option<String> {
    match hf {
        "embed_tokens.weight" => return Some("token_embd.weight".into()),
        "norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("layers.")?;
    let (n, name) = rest.split_once('.')?;
    let g = match name {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "layer_scalar" => "layer_output_scale.weight",
        _ => return None,
    };
    Some(format!("blk.{n}.{g}"))
}

/// Tensors from a checkpoint, with an optional LoRA folded in, found under whichever name prefix the checkpoint uses.
pub struct Loader {
    base: Source,
    lora: Option<(MmapedSafetensors, f64)>,
    prefix: String,
    pub dev: Device,
    pub dt: DType,
    merged: std::cell::Cell<usize>,
}

fn shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut s: Vec<_> = std::fs::read_dir(dir)
        .map_err(candle_core::Error::wrap)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    s.sort();
    if s.is_empty() {
        candle_core::bail!("no *.safetensors in {}", dir.display());
    }
    Ok(s)
}

pub fn read_json(p: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(p)
        .map_err(|e| candle_core::Error::Msg(format!("{}: {e}", p.display())))?;
    serde_json::from_str(&s).map_err(candle_core::Error::wrap)
}

impl Loader {
    /// `base_dir`: config.json + shards (or one *.gguf in it or under gguf/). `lora_dir`: adapter_model.safetensors +
    /// adapter_config.json.
    pub fn new(base_dir: &Path, lora_dir: Option<&Path>, dt: DType, dev: &Device) -> Result<Self> {
        let (base, prefix) = match shards(base_dir) {
            Ok(sh) => {
                // SAFETY: `sh` references valid safetensors files on disk; mappings are read-only, non-null,
                // and remain valid for the Loader's lifetime; files are assumed immutable during inference.
                let base = unsafe { MmapedSafetensors::multi(&sh)? };
                let names: Vec<String> = base.tensors().into_iter().map(|(n, _)| n).collect();
                let prefix = [
                    "model.language_model.",
                    "model.",
                    "language_model.model.",
                    "",
                ]
                .into_iter()
                .find(|p| {
                    names
                        .iter()
                        .any(|n| *n == format!("{p}embed_tokens.weight"))
                })
                .ok_or_else(|| {
                    candle_core::Error::Msg("checkpoint has no embed_tokens.weight".into())
                })?
                .to_string();
                (Source::St(base), prefix)
            }
            Err(_) => {
                let path = gguf_path(base_dir)?;
                let mut f = std::fs::File::open(&path).map_err(candle_core::Error::wrap)?;
                let content = gguf_file::Content::read(&mut f)?;
                eprintln!(
                    "kev: reading {} ({} tensors)",
                    path.display(),
                    content.tensor_infos.len()
                );
                (
                    Source::Gguf {
                        content,
                        file: std::sync::Mutex::new(f),
                    },
                    String::new(),
                )
            }
        };
        let lora = match lora_dir {
            Some(d) => {
                let acfg = read_json(&d.join("adapter_config.json"))?;
                let scale =
                    acfg["lora_alpha"].as_f64().unwrap_or(0.0) / acfg["r"].as_f64().unwrap_or(1.0);
                Some((
                    // SAFETY: `adapter_model.safetensors` is a valid safetensors file in `d`; read-only
                    // memory map remains valid and non-null for the duration of Loader without concurrent modification.
                    unsafe { MmapedSafetensors::new(d.join("adapter_model.safetensors"))? },
                    scale,
                ))
            }
            None => None,
        };
        Ok(Self {
            base,
            lora,
            prefix,
            dev: dev.clone(),
            dt,
            merged: std::cell::Cell::new(0),
        })
    }

    pub fn has(&self, name: &str) -> bool {
        match &self.base {
            Source::St(b) => b.get(&format!("{}{name}", self.prefix)).is_ok(),
            Source::Gguf { content, .. } => {
                gguf_name(name).is_some_and(|g| content.tensor_infos.contains_key(&g))
            }
        }
    }
    /// A tensor as stored (GGUF tensors dequantized to f32).
    pub fn raw(&self, name: &str) -> Result<Tensor> {
        match &self.base {
            Source::St(b) => b.load(&format!("{}{name}", self.prefix), &self.dev),
            Source::Gguf { content, file } => {
                let g = gguf_name(name)
                    .ok_or_else(|| candle_core::Error::Msg(format!("no GGUF name for {name}")))?;
                let q = content.tensor(&mut *file.lock().unwrap(), &g, &Device::Cpu)?;
                q.dequantize(&Device::Cpu)?.to_device(&self.dev)
            }
        }
    }
    /// Q8_0 blocks of a GGUF tensor, raw, on the device: (bytes, rows, cols), or None if it is not Q8_0.
    fn q8(&self, name: &str) -> Result<Option<(Tensor, usize, usize)>> {
        let Source::Gguf { content, file } = &self.base else {
            return Ok(None);
        };
        let Some(g) = gguf_name(name) else {
            return Ok(None);
        };
        let Some(info) = content.tensor_infos.get(&g) else {
            return Ok(None);
        };
        if info.ggml_dtype != candle_core::quantized::GgmlDType::Q8_0 {
            return Ok(None);
        }
        let (rows, cols) = info.shape.dims2()?;
        let bytes = rows * cols / 32 * 34;
        let mut buf = vec![0u8; bytes];
        {
            use std::io::{Read, Seek};
            let mut f = file.lock().unwrap();
            f.seek(std::io::SeekFrom::Start(
                content.tensor_data_offset + info.offset,
            ))
            .map_err(candle_core::Error::wrap)?;
            f.read_exact(&mut buf).map_err(candle_core::Error::wrap)?;
        }
        Ok(Some((Tensor::from_vec(buf, bytes, &self.dev)?, rows, cols)))
    }
    /// A small parameter as the reference holds it (cast to the model dtype), widened to f32 for the math.
    pub fn param(&self, name: &str) -> Result<Tensor> {
        self.raw(name)?.to_dtype(self.dt)?.to_dtype(DType::F32)
    }
    fn lora_pair(&self, module: &str) -> Option<(String, String)> {
        let (l, _) = self.lora.as_ref()?;
        for p in [
            "base_model.model.",
            "base_model.model.model.",
            "base_model.model.model.language_model.",
            "base_model.model.language_model.",
        ] {
            let a = format!("{p}{module}.lora_A.weight");
            if l.get(&a).is_ok() {
                return Some((a, format!("{p}{module}.lora_B.weight")));
            }
        }
        None
    }
    /// A projection weight with the LoRA folded in: W + scale * B @ A in f32, one rounding to the model dtype. A
    /// GGUF Q8_0 weight without a LoRA stays quantized.
    pub fn weight(&self, module: &str) -> Result<Proj> {
        let name = format!("{module}.weight");
        if self.lora_pair(module).is_none() {
            if let Some((q, out, inn)) = self.q8(&name)? {
                return Ok(Proj::Q8 { q, out, inn });
            }
        }
        let w = self.raw(&name)?;
        let w = match (self.lora_pair(module), &self.lora) {
            (Some((a, b)), Some((l, scale))) => {
                let a = l.load(&a, &self.dev)?.to_dtype(DType::F32)?;
                let b = l.load(&b, &self.dev)?.to_dtype(DType::F32)?;
                self.merged.set(self.merged.get() + 1);
                (w.to_dtype(DType::F32)? + (b.matmul(&a)? * *scale)?)?
            }
            _ => w,
        };
        Ok(Proj::Dense(w.to_dtype(self.dt)?))
    }
    pub fn linear(&self, modules: &[String]) -> Result<Proj> {
        Proj::cat(
            modules
                .iter()
                .map(|m| self.weight(m))
                .collect::<Result<Vec<_>>>()?,
        )
    }
    /// Fails when an adapter tensor went unused (a naming mismatch would otherwise serve the base silently).
    pub fn finish(&self) -> Result<usize> {
        if let Some((l, _)) = &self.lora {
            let n = l.tensors().len();
            if self.merged.get() * 2 != n {
                candle_core::bail!(
                    "merged {} LoRA pairs but the adapter holds {n} tensors",
                    self.merged.get()
                );
            }
        }
        Ok(self.merged.get())
    }
    /// The untied output projection, if the checkpoint has one.
    fn lm_head(&self) -> Result<Option<Tensor>> {
        let Source::St(b) = &self.base else {
            return Ok(None);
        };
        for n in [
            "lm_head.weight",
            "model.lm_head.weight",
            "language_model.lm_head.weight",
        ] {
            if b.get(n).is_ok() {
                return Ok(Some(b.load(n, &self.dev)?.to_dtype(self.dt)?));
            }
        }
        Ok(None)
    }
}

/// The GGUF file of a checkpoint dir: a *.gguf in it or under gguf/ (the Q8_0 one when there are several).
fn gguf_path(dir: &Path) -> Result<std::path::PathBuf> {
    let mut all = Vec::new();
    for d in [dir.to_path_buf(), dir.join("gguf")] {
        if let Ok(rd) = std::fs::read_dir(&d) {
            all.extend(rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| {
                let n = p.file_name().unwrap_or_default().to_string_lossy();
                n.ends_with(".gguf") && !n.starts_with("mmproj")
            }));
        }
    }
    all.sort();
    all.iter()
        .find(|p| p.to_string_lossy().contains("Q8_0"))
        .or(all.first())
        .cloned()
        .ok_or_else(|| {
            candle_core::Error::Msg(format!("no *.safetensors or *.gguf in {}", dir.display()))
        })
}

fn u(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| candle_core::Error::Msg(format!("config lacks {k}")))
}

/// torch: 1.0 / (base ** (arange(0, dim, 2).float() / dim)) for the first `n` pairs, then zeros up to `half` pairs,
/// divided by a linear-scaling factor.
fn inv_freq(
    theta: f64,
    dim: usize,
    n: usize,
    half: usize,
    factor: f64,
    dev: &Device,
) -> Result<Tensor> {
    let v: Vec<f32> = (0..half)
        .map(|i| {
            if i < n {
                1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32) / factor as f32
            } else {
                0.0
            }
        })
        .collect();
    Tensor::new(v, dev)
}

/// RoPE of one layer type from `rope_parameters[type]` (transformers 5) or the older flat keys:
/// -> (inv_freq, as AttnSpec takes it).
fn rope(cfg: &Value, layer_type: &str, hd: usize, dev: &Device) -> Result<Tensor> {
    let per = &cfg["rope_parameters"];
    let p = if per[layer_type].is_object() {
        &per[layer_type]
    } else {
        per
    };
    let local = layer_type == "sliding_attention";
    let theta = p["rope_theta"]
        .as_f64()
        .or(if local {
            cfg["rope_local_base_freq"].as_f64()
        } else {
            None
        })
        .or(cfg["rope_theta"].as_f64())
        .unwrap_or(10000.0);
    let partial = p["partial_rotary_factor"]
        .as_f64()
        .or(cfg["partial_rotary_factor"].as_f64())
        .unwrap_or(1.0);
    let scaling = if p.is_object() && p.get("rope_type").is_some() {
        p
    } else {
        &cfg["rope_scaling"]
    };
    let kind = scaling["rope_type"]
        .as_str()
        .or(scaling["type"].as_str())
        .unwrap_or("default");
    let factor = if kind == "linear" && !local {
        scaling["factor"].as_f64().unwrap_or(1.0)
    } else {
        1.0
    };
    match kind {
        // Gemma 4 global layers: rotate-half over the full head, the first partial/2 pairs rotating with an exponent
        // over the full head dim, the rest identity
        "proportional" => {
            let n = (partial * hd as f64 / 2.0) as usize;
            inv_freq(theta, hd, n, hd / 2, 1.0, dev)
        }
        "default" | "linear" => {
            let rot = (hd as f64 * partial) as usize;
            inv_freq(theta, rot, rot / 2, rot / 2, factor, dev)
        }
        other => candle_core::bail!("rope_type {other:?} not supported"),
    }
}

impl Model {
    /// Load a backbone from its checkpoint dir (config.json + shards), with an optional LoRA merged in.
    pub fn load(base_dir: &Path, lora_dir: Option<&Path>, dt: DType, dev: &Device) -> Result<Self> {
        let full = read_json(&base_dir.join("config.json"))?;
        let cfg = full.get("text_config").cloned().unwrap_or(full.clone());
        let mt = cfg["model_type"]
            .as_str()
            .or(full["model_type"].as_str())
            .unwrap_or("")
            .to_string();
        let ld = Loader::new(base_dir, lora_dir, dt, dev)?;
        let m = match mt.as_str() {
            "qwen3_5_text" | "qwen3_5" => Self::qwen35(&ld, &cfg, &full)?,
            "qwen3" => Self::dense(&ld, &cfg, &full, Family::Qwen3)?,
            "gemma3_text" | "gemma3" => Self::dense(&ld, &cfg, &full, Family::Gemma3)?,
            "gemma4_text" | "gemma4" | "gemma4_unified_text" | "gemma4_unified" => {
                Self::dense(&ld, &cfg, &full, Family::Gemma4)?
            }
            other => candle_core::bail!("unsupported model_type {other:?}"),
        };
        let merged = ld.finish()?;
        eprintln!(
            "kev: {} backbone, {} layers, hidden {}, dtype {dt:?}, merged {merged} LoRA pairs",
            m.model_type,
            m.layers.len(),
            m.hidden
        );
        Ok(m)
    }

    fn qwen35(ld: &Loader, cfg: &Value, full: &Value) -> Result<Self> {
        let (dt, dev) = (ld.dt, &ld.dev);
        let hidden = u(cfg, "hidden_size")?;
        let (nh, nkv, hd) = (
            u(cfg, "num_attention_heads")?,
            u(cfg, "num_key_value_heads")?,
            u(cfg, "head_dim")?,
        );
        let (hk, hv) = (
            u(cfg, "linear_num_key_heads")?,
            u(cfg, "linear_num_value_heads")?,
        );
        if u(cfg, "linear_key_head_dim")? != DK
            || u(cfg, "linear_value_head_dim")? != DV
            || hv % hk != 0
        {
            candle_core::bail!("DeltaNet kernel supports {DK}x{DV} heads only");
        }
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let inv = rope(cfg, "full_attention", hd, dev)?;
        let kk = u(cfg, "linear_conv_kernel_dim")?;
        let types: Vec<String> =
            serde_json::from_value(cfg["layer_types"].clone()).map_err(candle_core::Error::wrap)?;
        let norm1 = |name: &str| -> Result<Tensor> { ld.param(name)? + 1.0 };
        let gspec = GdnSpec {
            hk,
            hv,
            k: kk,
            ld: 2 * hk * DK + hv * DV + hv * DV + 2 * hv,
            eps,
        };
        let (mut lg, mut la) = (0, 0);
        let mut layers = Vec::new();
        for (i, ty) in types.iter().enumerate() {
            let p = format!("layers.{i}");
            let mixer = if ty == "linear_attention" {
                let m = format!("{p}.linear_attn");
                let conv = ld.param(&format!("{m}.conv1d.weight"))?; // [C, 1, K]
                lg += 1;
                Mixer::Gdn(Gdn {
                    spec: gspec.clone(),
                    lg: lg - 1,
                    proj: ld.linear(
                        &["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                            .map(|s| format!("{m}.{s}")),
                    )?,
                    conv_w: conv.squeeze(1)?.t()?.contiguous()?,
                    a_neg: ld.param(&format!("{m}.A_log"))?.exp()?.neg()?,
                    dt_bias: ld.param(&format!("{m}.dt_bias"))?,
                    norm_w: ld.param(&format!("{m}.norm.weight"))?,
                    out: ld.weight(&format!("{m}.out_proj"))?,
                })
            } else {
                let m = format!("{p}.self_attn");
                la += 1;
                Mixer::Attn(Attn {
                    spec: AttnSpec {
                        nh,
                        nkv,
                        hd,
                        q_stride: 2 * hd,
                        k_off: 2 * nh * hd,
                        v_off: 2 * nh * hd + nkv * hd,
                        gate: true,
                        kv_eq: false,
                        v_norm: false,
                        qn: Some(norm1(&format!("{m}.q_norm.weight"))?),
                        kn: Some(norm1(&format!("{m}.k_norm.weight"))?),
                        mode: NormMode::F32,
                        eps,
                        inv_freq: inv.clone(),
                        scale: (hd as f64).powf(-0.5),
                        softcap: 0.0,
                        window: 0,
                        kv_off: (la - 1) * nkv * hd,
                        ld: 2 * nh * hd + 2 * nkv * hd,
                    },
                    qkv: ld.linear(&["q_proj", "k_proj", "v_proj"].map(|s| format!("{m}.{s}")))?,
                    o: ld.weight(&format!("{m}.o_proj"))?,
                })
            };
            layers.push(Layer {
                in_norm: norm1(&format!("{p}.input_layernorm.weight"))?,
                post_mix: None,
                pre_mlp: norm1(&format!("{p}.post_attention_layernorm.weight"))?,
                post_mlp: None,
                scalar: 1.0,
                mixer,
                gate_up: ld.linear(&["gate_proj", "up_proj"].map(|s| format!("{p}.mlp.{s}")))?,
                down: ld.weight(&format!("{p}.mlp.down_proj"))?,
            });
        }
        let tied = full["tie_word_embeddings"]
            .as_bool()
            .or(cfg["tie_word_embeddings"].as_bool())
            .unwrap_or(true);
        Ok(Self {
            model_type: "qwen3_5_text".into(),
            hidden,
            dt,
            dev: dev.clone(),
            embed: ld.raw("embed_tokens.weight")?.to_dtype(dt)?,
            lm_head: if tied { None } else { ld.lm_head()? },
            final_softcap: 0.0,
            embed_scale: None,
            eps,
            norm_mode: NormMode::F32,
            act: Act::Silu,
            round_act: false,
            layers,
            norm: norm1("norm.weight")?,
            cache: CacheShape {
                gdn_layers: lg,
                conv: (kk - 1) * gspec.c(),
                rec: hv * DK * DV,
                kv: la * nkv * hd,
            },
        })
    }

    /// Dense decoders: Qwen3, Gemma 3, Gemma 4 (text).
    fn dense(ld: &Loader, cfg: &Value, full: &Value, fam: Family) -> Result<Self> {
        let (dt, dev) = (ld.dt, &ld.dev);
        let hidden = u(cfg, "hidden_size")?;
        let nl = u(cfg, "num_hidden_layers")?;
        let nh = u(cfg, "num_attention_heads")?;
        let hd = cfg["head_dim"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or(hidden / nh);
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6);
        let gemma = fam != Family::Qwen3;
        // Gemma 3 norms are zero-centred (1 + w); Qwen3 and Gemma 4 use w
        let nw = |name: &str| -> Result<Tensor> {
            let w = ld.param(name)?;
            if fam == Family::Gemma3 {
                w + 1.0
            } else {
                Ok(w)
            }
        };
        let types: Vec<String> = match cfg["layer_types"].as_array() {
            Some(a) => a
                .iter()
                .map(|x| x.as_str().unwrap_or("full_attention").to_string())
                .collect(),
            None if fam == Family::Gemma3 => {
                let pat = cfg["sliding_window_pattern"].as_u64().unwrap_or(6) as usize;
                (0..nl)
                    .map(|i| {
                        if (i + 1) % pat == 0 {
                            "full_attention"
                        } else {
                            "sliding_attention"
                        }
                        .to_string()
                    })
                    .collect()
            }
            None => vec!["full_attention".into(); nl],
        };
        let window = cfg["sliding_window"].as_u64().unwrap_or(0) as usize;
        let use_window = fam != Family::Qwen3 || cfg["use_sliding_window"].as_bool() == Some(true);
        let mut layers = Vec::with_capacity(nl);
        let mut kv_off = 0;
        for (i, ty) in types.iter().enumerate().take(nl) {
            let p = format!("layers.{i}");
            let m = format!("{p}.self_attn");
            let global = ty != "sliding_attention";
            if fam == Family::Gemma4
                && (cfg["num_kv_shared_layers"].as_u64().unwrap_or(0) > 0
                    || cfg["hidden_size_per_layer_input"].as_u64().unwrap_or(0) > 0)
            {
                candle_core::bail!(
                    "Gemma 4 with per-layer inputs or shared KV layers (E2B/E4B) is not supported"
                );
            }
            // head size and KV heads from the tensors: Gemma 4 global layers have their own (global_head_dim, or
            // per_layer_config), and keys double as values where there is no v_proj
            let qn = nw(&format!("{m}.q_norm.weight"))?;
            let lhd = qn.dims1()?;
            let kv_eq = fam == Family::Gemma4 && global && !ld.has(&format!("{m}.v_proj.weight"));
            let mut projs = vec![
                ld.weight(&format!("{m}.q_proj"))?,
                ld.weight(&format!("{m}.k_proj"))?,
            ];
            let lkv = projs[1].rows() / lhd;
            if !kv_eq {
                projs.push(ld.weight(&format!("{m}.v_proj"))?);
            }
            let scale = match fam {
                Family::Gemma4 => 1.0,
                Family::Gemma3 => cfg["query_pre_attn_scalar"]
                    .as_f64()
                    .unwrap_or(hd as f64)
                    .powf(-0.5),
                Family::Qwen3 => (lhd as f64).powf(-0.5),
            };
            let spec = AttnSpec {
                nh,
                nkv: lkv,
                hd: lhd,
                q_stride: lhd,
                k_off: nh * lhd,
                v_off: nh * lhd + lkv * lhd,
                gate: false,
                kv_eq,
                v_norm: fam == Family::Gemma4,
                qn: Some(qn),
                kn: Some(nw(&format!("{m}.k_norm.weight"))?),
                mode: if fam == Family::Qwen3 {
                    NormMode::Rounded
                } else {
                    NormMode::F32
                },
                eps,
                inv_freq: rope(
                    cfg,
                    if global {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    },
                    lhd,
                    dev,
                )?,
                scale,
                softcap: cfg["attn_logit_softcapping"].as_f64().unwrap_or(0.0),
                window: if !global && use_window { window } else { 0 },
                kv_off,
                ld: nh * lhd + if kv_eq { lkv * lhd } else { 2 * lkv * lhd },
            };
            kv_off += lkv * lhd;
            let scalar = if fam == Family::Gemma4 && ld.has(&format!("{p}.layer_scalar")) {
                ld.raw(&format!("{p}.layer_scalar"))?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?[0] as f64
            } else {
                1.0
            };
            layers.push(Layer {
                in_norm: nw(&format!("{p}.input_layernorm.weight"))?,
                post_mix: if gemma {
                    Some(nw(&format!("{p}.post_attention_layernorm.weight"))?)
                } else {
                    None
                },
                pre_mlp: nw(&format!(
                    "{p}.{}",
                    if gemma {
                        "pre_feedforward_layernorm.weight"
                    } else {
                        "post_attention_layernorm.weight"
                    }
                ))?,
                post_mlp: if gemma {
                    Some(nw(&format!("{p}.post_feedforward_layernorm.weight"))?)
                } else {
                    None
                },
                scalar,
                mixer: Mixer::Attn(Attn {
                    spec,
                    qkv: Proj::cat(projs)?,
                    o: ld.weight(&format!("{m}.o_proj"))?,
                }),
                gate_up: ld.linear(&["gate_proj", "up_proj"].map(|s| format!("{p}.mlp.{s}")))?,
                down: ld.weight(&format!("{p}.mlp.down_proj"))?,
            });
        }
        let tied = full["tie_word_embeddings"]
            .as_bool()
            .or(cfg["tie_word_embeddings"].as_bool())
            .unwrap_or(true);
        // Gemma scales the embeddings by sqrt(hidden), a tensor in the model dtype
        let embed_scale = gemma.then(|| {
            let s = (hidden as f64).sqrt();
            if dt == DType::BF16 {
                half::bf16::from_f64(s).to_f64()
            } else {
                s as f32 as f64
            }
        });
        Ok(Self {
            model_type: cfg["model_type"].as_str().unwrap_or("").into(),
            hidden,
            dt,
            dev: dev.clone(),
            embed: match ld.q8("embed_tokens.weight")? {
                Some((q, r, c)) => kernels::dequant_q8(&q, r, c, dt)?,
                None => ld.raw("embed_tokens.weight")?.to_dtype(dt)?,
            },
            lm_head: if tied { None } else { ld.lm_head()? },
            final_softcap: cfg["final_logit_softcapping"].as_f64().unwrap_or(0.0),
            embed_scale,
            eps,
            norm_mode: if fam == Family::Qwen3 {
                NormMode::Rounded
            } else {
                NormMode::F32
            },
            act: if gemma { Act::GeluTanh } else { Act::Silu },
            round_act: true,
            layers,
            norm: nw("norm.weight")?,
            cache: CacheShape {
                gdn_layers: 0,
                conv: 0,
                rec: 0,
                kv: kv_off,
            },
        })
    }

    /// The output rows of the vocabulary (the untied lm_head, else the tied embedding).
    pub fn out_rows(&self) -> &Tensor {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    /// One pass over the packed tokens `ids` (laid out as `pack`): fills the caches of the state-building sequences
    /// and returns the final-norm hidden states at the packed positions `picks` (rounded to the model dtype, as the
    /// reference's last_hidden_state), f32 [picks, hidden]. With no picks the last layer stops once its caches are
    /// written.
    pub fn forward(&self, pack: &Pack, ids: &[u32], picks: &[u32]) -> Result<Tensor> {
        let (mode, eps) = (self.norm_mode, self.eps);
        let mut x = self.embed.index_select(&Tensor::new(ids, &self.dev)?, 0)?;
        if let Some(s) = self.embed_scale {
            x = (x * s)?;
        }
        let mut h = kernels::add_norm(&x, None, Some(&self.layers[0].in_norm), eps, mode, 1.0)?.1;
        let picks_t = Tensor::new(picks, &self.dev)?;
        let nl = self.layers.len();
        for (i, l) in self.layers.iter().enumerate() {
            let last = i + 1 == nl;
            let Some(core) = self.mixer(&l.mixer, &h, pack, last && picks.is_empty())? else {
                break;
            };
            let (core, xr) = if last {
                (
                    core.index_select(&picks_t, 0)?,
                    x.index_select(&picks_t, 0)?,
                )
            } else {
                (core, x)
            };
            let mut mixed = match &l.mixer {
                Mixer::Gdn(g) => g.out.forward(&core)?,
                Mixer::Attn(a) => a.o.forward(&core)?,
            };
            if let Some(w) = &l.post_mix {
                mixed = kernels::add_norm(&mixed, None, Some(w), eps, mode, 1.0)?.1;
            }
            let (xn, hn) = kernels::add_norm(&xr, Some(&mixed), Some(&l.pre_mlp), eps, mode, 1.0)?;
            let mut m = l.down.forward(&kernels::act_mul(
                &l.gate_up.forward(&hn)?,
                self.act,
                self.round_act,
            )?)?;
            if let Some(w) = &l.post_mlp {
                m = kernels::add_norm(&m, None, Some(w), eps, mode, 1.0)?.1;
            }
            let next = if last {
                &self.norm
            } else {
                &self.layers[i + 1].in_norm
            };
            let (xn2, hn2) = kernels::add_norm(&xn, Some(&m), Some(next), eps, mode, l.scalar)?;
            if last {
                return hn2.to_dtype(DType::F32);
            }
            x = xn2;
            h = hn2;
        }
        Tensor::zeros((0, self.hidden), DType::F32, &self.dev)
    }

    fn mixer(
        &self,
        m: &Mixer,
        h: &Tensor,
        pack: &Pack,
        cache_only: bool,
    ) -> Result<Option<Tensor>> {
        Ok(match m {
            Mixer::Gdn(g) => {
                let p = g.proj.forward(h)?;
                let qkv = kernels::conv(&p, &g.conv_w, pack, &g.spec, g.lg)?;
                let o = kernels::gdn(&qkv, &p, &g.a_neg, &g.dt_bias, pack, &g.spec, g.lg)?;
                if cache_only {
                    return Ok(None);
                }
                Some(kernels::gated_norm(&o, &p, &g.norm_w, &g.spec)?)
            }
            Mixer::Attn(a) => {
                let p = a.qkv.forward(h)?;
                let (q, k, v) = kernels::qkv_prep(&p, pack, &a.spec)?;
                if cache_only {
                    return Ok(None);
                }
                Some(kernels::attention(&q, &k, &v, &p, pack, &a.spec)?)
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Family {
    Qwen3,
    Gemma3,
    Gemma4,
}
