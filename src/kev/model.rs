//! Kev decision model on Candle: the Qwen3.5 hybrid text backbone (Gated DeltaNet + gated full attention), a merged
//! LoRA adapter and Kev's pointer head. Mirrors transformers 5.17 `modeling_qwen3_5.py` (Qwen3_5TextModel) and
//! kev `model.py` (PointerHead, row form with a cached state prefix).
//!
//! Precision follows kev.serve's bf16 path: weights `round_bf16(W + scale * B @ A)` (delta in f32), the residual stream
//! and projections in the model dtype, every norm, the attention scores/softmax and the DeltaNet recurrence in f32.

use super::kernels::{ConvSilu, GdnOp, MaskedSoftmax, DK, DV};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{ops, Linear};
use std::path::Path;

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Rope {
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}
fn one() -> f64 {
    1.0
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: Rope,
}

/// Per-layer cache of a token sequence: attention keys/values [B, kv_heads, T, head_dim] (f32, post-RoPE), or the
/// DeltaNet conv state (last K-1 pre-conv inputs [B, K-1, conv_dim], model dtype) and recurrent state [B, HV, DK, DV] f32.
#[derive(Clone)]
pub enum LayerState {
    Attn { k: Tensor, v: Tensor },
    Gdn { conv: Tensor, rec: Tensor },
}

/// What a batch of rows continues from: per layer the batched state (attention caches right-padded to `smax`), and
/// each row's real state length.
pub struct Past {
    pub layers: Vec<LayerState>,
    pub plen: Vec<u32>,
    pub smax: usize,
}

struct Attn {
    qkv: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
}

struct Gdn {
    proj: Linear,    // in_proj_qkv | in_proj_z | in_proj_b | in_proj_a
    conv_w: Tensor,  // taps [K, conv_dim] f32
    a_neg: Tensor,   // -exp(A_log) [HV] f32
    dt_bias: Tensor, // [HV] f32
    norm_w: Tensor,  // [DV] f32
    out: Linear,
}

enum Mixer {
    Attn(Attn),
    Gdn(Gdn),
}

struct Layer {
    in_norm: Tensor,   // 1 + w, f32
    post_norm: Tensor, // 1 + w, f32
    mixer: Mixer,
    gate_up: Linear,
    down: Linear,
}

pub struct Model {
    pub cfg: Config,
    pub dt: DType,
    pub dev: Device,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    inv_freq: Tensor,
    head_w: Tensor, // [q; k] [2 * dp, hidden] f32
    head_b: Tensor, // [2 * dp] f32
    pub head_dim: usize,
    pub temperature: f64,
}

struct Loader {
    base: MmapedSafetensors,
    lora: MmapedSafetensors,
    scale: f64,
    dev: Device,
    dt: DType,
    merged: std::cell::Cell<usize>,
}

impl Loader {
    fn raw(&self, name: &str) -> Result<Tensor> {
        self.base
            .load(&format!("model.language_model.{name}"), &self.dev)
    }
    /// A small parameter as kev.serve holds it (cast to the model dtype), widened to f32 for the math.
    fn param(&self, name: &str) -> Result<Tensor> {
        self.raw(name)?.to_dtype(self.dt)?.to_dtype(DType::F32)
    }
    /// A projection weight with the LoRA folded in: W + scale * B @ A in f32, one rounding to the model dtype.
    fn weight(&self, module: &str) -> Result<Tensor> {
        let w = self
            .raw(&format!("{module}.weight"))?
            .to_dtype(DType::F32)?;
        let a_key = format!("base_model.model.{module}.lora_A.weight");
        let w = if self.lora.get(&a_key).is_ok() {
            let a = self.lora.load(&a_key, &self.dev)?.to_dtype(DType::F32)?;
            let b = self
                .lora
                .load(
                    &format!("base_model.model.{module}.lora_B.weight"),
                    &self.dev,
                )?
                .to_dtype(DType::F32)?;
            self.merged.set(self.merged.get() + 1);
            (w + (b.matmul(&a)? * self.scale)?)?
        } else {
            w
        };
        w.to_dtype(self.dt)
    }
    fn linear(&self, modules: &[String]) -> Result<Linear> {
        let ws = modules
            .iter()
            .map(|m| self.weight(m))
            .collect::<Result<Vec<_>>>()?;
        Ok(Linear::new(Tensor::cat(&ws, 0)?, None))
    }
}

fn profiling() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("KEV_PROFILE").is_ok_and(|v| v == "1"))
}

thread_local! {
    static MARKS: std::cell::RefCell<(std::time::Instant, Vec<(&'static str, f64)>)> = std::cell::RefCell::new((std::time::Instant::now(), Vec::new()));
}

/// KEV_PROFILE=1: synchronise and charge the time since the previous mark to `name` (a debugging aid; it serialises
/// the GPU). `None` resets and returns the per-name totals so far.
fn mark(dev: &Device, name: Option<&'static str>) -> Result<Vec<(&'static str, f64)>> {
    if !profiling() {
        return Ok(Vec::new());
    }
    dev.synchronize()?;
    MARKS.with(|m| {
        let mut m = m.borrow_mut();
        let dt = m.0.elapsed().as_secs_f64() * 1e3;
        m.0 = std::time::Instant::now();
        match name {
            Some(n) => {
                match m.1.iter_mut().find(|(k, _)| *k == n) {
                    Some(e) => e.1 += dt,
                    None => m.1.push((n, dt)),
                }
                Ok(Vec::new())
            }
            None => Ok(std::mem::take(&mut m.1)),
        }
    })
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // log(1 + e^x) = relu(x) + log(1 + e^-|x|)
    x.relu()? + ((x.abs()?.neg()?.exp()? + 1.0)?.log()?)
}

impl Model {
    /// base_dir: the base snapshot (config.json + shards); kev_dir: adapter_model.safetensors + adapter_config.json;
    /// head: head.safetensors + head_meta.json converted from head.pt.
    pub fn load(
        base_dir: &Path,
        kev_dir: &Path,
        head_dir: &Path,
        dt: DType,
        dev: &Device,
        temperature: Option<f64>,
    ) -> Result<Self> {
        let read = |p: &Path| std::fs::read_to_string(p).map_err(candle_core::Error::wrap);
        let full: serde_json::Value = serde_json::from_str(&read(&base_dir.join("config.json"))?)
            .map_err(candle_core::Error::wrap)?;
        let cfg: Config = serde_json::from_value(full.get("text_config").cloned().unwrap_or(full))
            .map_err(candle_core::Error::wrap)?;
        if cfg.linear_key_head_dim != DK
            || cfg.linear_value_head_dim != DV
            || cfg.linear_num_value_heads % cfg.linear_num_key_heads != 0
        {
            candle_core::bail!("DeltaNet kernel supports {DK}x{DV} heads only, got {cfg:?}");
        }
        let acfg: serde_json::Value =
            serde_json::from_str(&read(&kev_dir.join("adapter_config.json"))?)
                .map_err(candle_core::Error::wrap)?;
        let scale = acfg["lora_alpha"].as_f64().unwrap_or(0.0) / acfg["r"].as_f64().unwrap_or(1.0);
        let meta: serde_json::Value =
            serde_json::from_str(&read(&head_dir.join("head_meta.json"))?)
                .map_err(candle_core::Error::wrap)?;
        let mut shards: Vec<_> = std::fs::read_dir(base_dir)
            .map_err(candle_core::Error::wrap)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        let ld = Loader {
            base: unsafe { MmapedSafetensors::multi(&shards)? },
            lora: unsafe { MmapedSafetensors::new(kev_dir.join("adapter_model.safetensors"))? },
            scale,
            dev: dev.clone(),
            dt,
            merged: std::cell::Cell::new(0),
        };
        let norm1 = |name: &str| -> Result<Tensor> { ld.param(name)? + 1.0 };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let mixer = if cfg.layer_types[i] == "linear_attention" {
                let m = format!("{p}.linear_attn");
                let conv = ld.param(&format!("{m}.conv1d.weight"))?; // [C, 1, K]
                let conv_w = conv.squeeze(1)?.t()?.contiguous()?;
                Mixer::Gdn(Gdn {
                    proj: ld.linear(
                        &["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                            .map(|s| format!("{m}.{s}")),
                    )?,
                    conv_w,
                    a_neg: ld.param(&format!("{m}.A_log"))?.exp()?.neg()?,
                    dt_bias: ld.param(&format!("{m}.dt_bias"))?,
                    norm_w: ld.param(&format!("{m}.norm.weight"))?,
                    out: Linear::new(ld.weight(&format!("{m}.out_proj"))?, None),
                })
            } else {
                let m = format!("{p}.self_attn");
                Mixer::Attn(Attn {
                    qkv: ld.linear(&["q_proj", "k_proj", "v_proj"].map(|s| format!("{m}.{s}")))?,
                    o: Linear::new(ld.weight(&format!("{m}.o_proj"))?, None),
                    q_norm: norm1(&format!("{m}.q_norm.weight"))?,
                    k_norm: norm1(&format!("{m}.k_norm.weight"))?,
                })
            };
            layers.push(Layer {
                in_norm: norm1(&format!("{p}.input_layernorm.weight"))?,
                post_norm: norm1(&format!("{p}.post_attention_layernorm.weight"))?,
                mixer,
                gate_up: ld.linear(&["gate_proj", "up_proj"].map(|s| format!("{p}.mlp.{s}")))?,
                down: Linear::new(ld.weight(&format!("{p}.mlp.down_proj"))?, None),
            });
        }
        let lora_tensors = ld.lora.tensors().len();
        if ld.merged.get() * 2 != lora_tensors {
            candle_core::bail!(
                "merged {} LoRA pairs but the adapter holds {lora_tensors} tensors",
                ld.merged.get()
            );
        }
        let rot = (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor) as usize;
        // torch: 1.0 / (base ** (arange(0, dim, 2).float() / dim)), all f32
        let inv: Vec<f32> = (0..rot / 2)
            .map(|i| {
                1.0f32 / (cfg.rope_parameters.rope_theta as f32).powf((2 * i) as f32 / rot as f32)
            })
            .collect();
        let head = candle_core::safetensors::load(head_dir.join("head.safetensors"), dev)?;
        let hw = |k: &str| {
            head.get(k)
                .cloned()
                .ok_or_else(|| candle_core::Error::Msg(format!("head.safetensors lacks {k}")))
        };
        let head_w = Tensor::cat(&[hw("q.weight")?, hw("k.weight")?], 0)?;
        let head_b = Tensor::cat(&[hw("q.bias")?, hw("k.bias")?], 0)?;
        let head_dim = hw("q.bias")?.dims1()?;
        let temperature =
            temperature.unwrap_or_else(|| meta["temperature"].as_f64().unwrap_or(1.0));
        eprintln!("kev: merged {} LoRA pairs (scale {scale}), head dp={head_dim}, temperature {temperature:.4}, dtype {dt:?}", ld.merged.get());
        Ok(Self {
            embed: ld.raw("embed_tokens.weight")?.to_dtype(dt)?,
            norm: norm1("norm.weight")?,
            layers,
            inv_freq: Tensor::new(inv, dev)?,
            head_w,
            head_b,
            head_dim,
            temperature,
            dt,
            dev: dev.clone(),
            cfg,
        })
    }

    fn rms(&self, x: &Tensor, w1: &Tensor) -> Result<Tensor> {
        ops::rms_norm(
            &x.to_dtype(DType::F32)?.contiguous()?,
            w1,
            self.cfg.rms_norm_eps as f32,
        )?
        .to_dtype(self.dt)
    }

    /// Hidden states after the final norm, f32 [B, T, hidden], for right-padded rows `ids` [B, T] (u32) at positions
    /// `pos` [B, T] (u32), `lens` real tokens per row, continuing `past` when given. With `want`, also the cache of these
    /// tokens (attention k/v of the whole padded pass; DeltaNet states taken at each row's own length).
    pub fn forward(
        &self,
        ids: &Tensor,
        pos: &Tensor,
        lens: &[u32],
        past: Option<&Past>,
        want: bool,
    ) -> Result<(Tensor, Vec<LayerState>)> {
        let (b, t) = ids.dims2()?;
        let h = self.cfg.hidden_size;
        let mut x = self
            .embed
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, t, h))?;
        let freqs = pos
            .to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_mul(&self.inv_freq.reshape((1, 1, ()))?)?;
        let (cos, sin) = (freqs.cos()?.unsqueeze(1)?, freqs.sin()?.unsqueeze(1)?); // [B, 1, T, rot/2]
        let (plen, smax) = past.map_or((vec![0; b], 0), |p| (p.plen.clone(), p.smax));
        let mut states = Vec::new();
        mark(&self.dev, None)?;
        for (i, l) in self.layers.iter().enumerate() {
            let hn = self.rms(&x, &l.in_norm)?;
            mark(&self.dev, Some("norm"))?;
            let (mixed, st) = match &l.mixer {
                Mixer::Gdn(g) => {
                    let init = past.map(|p| match &p.layers[i] {
                        LayerState::Gdn { conv, rec } => (conv, rec),
                        _ => unreachable!(),
                    });
                    self.gdn(g, &hn, lens, init, want)?
                }
                Mixer::Attn(a) => {
                    let pk = past.map(|p| match &p.layers[i] {
                        LayerState::Attn { k, v } => (k, v),
                        _ => unreachable!(),
                    });
                    self.attn(a, &hn, &cos, &sin, pk, &plen, smax)?
                }
            };
            if want {
                states.push(st.expect("state requested"));
            }
            x = (x + mixed)?;
            mark(&self.dev, Some("residual"))?;
            let hn = self.rms(&x, &l.post_norm)?;
            mark(&self.dev, Some("norm"))?;
            let gu = l.gate_up.forward(&hn)?;
            let n = self.cfg.intermediate_size;
            let m = (ops::silu(&gu.narrow(2, 0, n)?)? * gu.narrow(2, n, n)?)?;
            x = (x + l.down.forward(&m)?)?;
            mark(&self.dev, Some("mlp"))?;
        }
        if profiling() {
            let marks = mark(&self.dev, None)?;
            let total: f64 = marks.iter().map(|m| m.1).sum();
            let parts: Vec<String> = marks.iter().map(|(k, v)| format!("{k} {v:.1}")).collect();
            eprintln!(
                "kev profile: B={b} T={t} past={smax} total {total:.1} ms: {}",
                parts.join(", ")
            );
        }
        let out = ops::rms_norm(
            &x.to_dtype(DType::F32)?.contiguous()?,
            &self.norm,
            self.cfg.rms_norm_eps as f32,
        )?;
        Ok((out, states))
    }

    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = cos.dim(D::Minus1)?;
        let d = x.dim(D::Minus1)?;
        let (x1, x2) = (x.narrow(3, 0, half)?, x.narrow(3, half, half)?);
        let r1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
        let r2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
        Tensor::cat(&[&r1, &r2, &x.narrow(3, 2 * half, d - 2 * half)?], 3)?.contiguous()
    }

    fn attn(
        &self,
        a: &Attn,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
        plen: &[u32],
        smax: usize,
    ) -> Result<(Tensor, Option<LayerState>)> {
        let (b, t, _) = x.dims3()?;
        let (nh, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let p = a.qkv.forward(x)?;
        mark(&self.dev, Some("attn.proj"))?;
        let qg = p.narrow(2, 0, nh * hd * 2)?.reshape((b, t, nh, 2 * hd))?;
        let gate = qg
            .narrow(3, hd, hd)?
            .contiguous()?
            .reshape((b, t, nh * hd))?;
        let q = self.rms(&qg.narrow(3, 0, hd)?, &a.q_norm)?;
        let k = self.rms(
            &p.narrow(2, nh * hd * 2, nkv * hd)?
                .reshape((b, t, nkv, hd))?,
            &a.k_norm,
        )?;
        let v = p
            .narrow(2, nh * hd * 2 + nkv * hd, nkv * hd)?
            .reshape((b, t, nkv, hd))?;
        let f = |z: &Tensor| z.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous();
        let q = Self::rope(&f(&q)?, cos, sin)?;
        let k = Self::rope(&f(&k)?, cos, sin)?;
        let v = f(&v)?;
        let rep = nh / nkv;
        let scale = (hd as f64).powf(-0.5);
        let qg = q.reshape((b, nkv, rep * t, hd))?; // q heads g*rep..g*rep+rep share kv head g
        mark(&self.dev, Some("attn.qk_norm_rope"))?;
        let (kf, vf) = match past {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            None => (k.clone(), v.clone()),
        };
        mark(&self.dev, Some("attn.cat"))?;
        let sc = qg.matmul(&kf.t()?)?;
        mark(&self.dev, Some("attn.qk"))?;
        let pr = sc.apply_op1_no_bwd(&MaskedSoftmax {
            plen: plen.to_vec(),
            smax,
            t,
            scale: scale as f32,
        })?;
        mark(&self.dev, Some("attn.softmax"))?;
        let o = pr.matmul(&vf)?; // [B, nkv, rep*T, hd]
        mark(&self.dev, Some("attn.pv"))?;
        let o = o
            .reshape((b, nh, t, hd))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, nh * hd))?;
        let o = (o.to_dtype(self.dt)? * ops::sigmoid(&gate)?)?;
        mark(&self.dev, Some("attn.gate"))?;
        let o = a.o.forward(&o)?;
        mark(&self.dev, Some("attn.out"))?;
        Ok((o, Some(LayerState::Attn { k, v })))
    }

    fn gdn(
        &self,
        g: &Gdn,
        x: &Tensor,
        lens: &[u32],
        init: Option<(&Tensor, &Tensor)>,
        want: bool,
    ) -> Result<(Tensor, Option<LayerState>)> {
        let (b, t, _) = x.dims3()?;
        let c = &self.cfg;
        let (hv, kdim, vdim) = (
            c.linear_num_value_heads,
            c.linear_num_key_heads * DK,
            c.linear_num_value_heads * DV,
        );
        let conv_dim = 2 * kdim + vdim;
        let kk = c.linear_conv_kernel_dim;
        let p = g.proj.forward(x)?;
        let mixed = p.narrow(2, 0, conv_dim)?;
        let z = p.narrow(2, conv_dim, vdim)?;
        let bb = p.narrow(2, conv_dim + vdim, hv)?;
        let a = p.narrow(2, conv_dim + vdim + hv, hv)?;
        let prev = match init {
            Some((cs, _)) => cs.clone(),
            None => Tensor::zeros((b, kk - 1, conv_dim), self.dt, &self.dev)?,
        };
        mark(&self.dev, Some("gdn.proj"))?;
        let qkv = p.apply_op3_no_bwd(&prev, &g.conv_w, &ConvSilu { c: conv_dim })?; // f32 [B, T, C]
        mark(&self.dev, Some("gdn.conv"))?;
        let decay = softplus(&a.to_dtype(DType::F32)?.broadcast_add(&g.dt_bias)?)?
            .broadcast_mul(&g.a_neg)?;
        let beta = ops::sigmoid(&bb.to_dtype(DType::F32)?)?;
        let gb = Tensor::cat(&[&decay, &beta], 2)?.contiguous()?;
        let s0 = match init {
            Some((_, rec)) => rec.contiguous()?,
            None => Tensor::zeros((b, hv, DK, DV), DType::F32, &self.dev)?,
        };
        mark(&self.dev, Some("gdn.gates"))?;
        let res = qkv.apply_op3_no_bwd(
            &gb,
            &s0,
            &GdnOp {
                lens: lens.to_vec(),
                kheads: c.linear_num_key_heads,
                vheads: hv,
            },
        )?;
        mark(&self.dev, Some("gdn.kernel"))?;
        let n1 = b * t * hv * DV;
        let o = res.narrow(0, 0, n1)?.reshape((b, t, hv, DV))?;
        let o = ops::rms_norm(&o, &g.norm_w, c.rms_norm_eps as f32)?;
        let zf = z.to_dtype(DType::F32)?.reshape((b, t, hv, DV))?;
        let o = (o * ops::silu(&zf)?)?
            .to_dtype(self.dt)?
            .reshape((b, t, vdim))?;
        mark(&self.dev, Some("gdn.norm"))?;
        let out = g.out.forward(&o)?;
        mark(&self.dev, Some("gdn.out"))?;
        let st = if want {
            let rec = res
                .narrow(0, n1, b * hv * DK * DV)?
                .reshape((b, hv, DK, DV))?;
            let xin = Tensor::cat(&[&prev, &mixed], 1)?; // [B, K-1+T, C]: the last K-1 inputs of row i end at lens[i]
            let convs = (0..b)
                .map(|i| xin.narrow(0, i, 1)?.narrow(1, lens[i] as usize, kk - 1))
                .collect::<Result<Vec<_>>>()?;
            Some(LayerState::Gdn {
                conv: Tensor::cat(&convs, 0)?,
                rec,
            })
        } else {
            None
        };
        Ok((out, st))
    }

    /// Pointer head: for picked hidden states `x` [P, hidden] f32 laid out per question as <decide> then its options,
    /// the option probabilities per question (softmax of k(h_opt) . q(h_decide) / sqrt(dp) / temperature).
    pub fn head(&self, x: &Tensor, ks: &[usize]) -> Result<Vec<Vec<f64>>> {
        let proj = x.matmul(&self.head_w.t()?)?.broadcast_add(&self.head_b)?;
        let proj: Vec<Vec<f32>> = proj.to_vec2()?;
        let dp = self.head_dim;
        let scale = 1.0 / (dp as f64).sqrt();
        let mut at = 0;
        let mut out = Vec::with_capacity(ks.len());
        for &k in ks {
            let q = &proj[at][..dp];
            let z: Vec<f64> = (0..k)
                .map(|j| {
                    let kv = &proj[at + 1 + j][dp..];
                    let dot: f64 = q.iter().zip(kv).map(|(a, b)| *a as f64 * *b as f64).sum();
                    dot * scale / self.temperature
                })
                .collect();
            let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
            let s: f64 = e.iter().sum();
            out.push(e.into_iter().map(|v| v / s).collect());
            at += 1 + k;
        }
        Ok(out)
    }
}
