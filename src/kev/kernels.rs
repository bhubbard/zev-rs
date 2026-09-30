//! Packed-varlen kernels for the decoder layers (CUDA via NVRTC, compiled on first use) and their CPU references.
//!
//! A pass packs the tokens of many sequences back to back ([N, ...], no padding). A sequence either builds a state
//! (it has a `keep` cache that the pass fills layer by layer) or continues one (its `past`: a state built by an
//! earlier pass, or by this pass for a row whose state is in it). Kernels read and write those caches through
//! per-sequence device pointers, so a row pass never gathers, pads or copies a cached state. Within a layer the
//! state-building sequences run first (tier 0), then the rows that continue them (tier 1).
//!
//! Every op has a CPU reference (plain loops over host copies); the tests check each CUDA kernel against it, and the
//! `candle` CPU build runs on the references.

use candle_core::{DType, Device, Result, Tensor};
use std::sync::Arc;

// The DeltaNet kernel is written for these head sizes (every Qwen3.5 size); Model::load checks the config.
pub const DK: usize = 128;
pub const DV: usize = 128;

/// One sequence's cache across all layers, filled in place by the pass that builds it. DeltaNet layers: conv
/// [Lg, K-1, C] (model dtype, the last K-1 conv inputs) and rec [Lg, HV, DK, DV] f32; attention layers: k and v flat
/// in the model dtype, layer `la` holding [len, width_la] at offset len * kv_off[la].
pub struct StateCache {
    pub len: usize,
    pub conv: Option<Tensor>,
    pub rec: Option<Tensor>,
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

/// The cache shapes of one model, to allocate a StateCache.
#[derive(Clone, Debug, Default)]
pub struct CacheShape {
    pub gdn_layers: usize,
    pub conv: usize, // (K-1) * C per layer
    pub rec: usize,  // HV * DK * DV per layer
    pub kv: usize,   // sum over attention layers of their widths (per position)
}

impl StateCache {
    pub fn alloc(len: usize, s: &CacheShape, dt: DType, dev: &Device) -> Result<Self> {
        let t = |n: usize, dt: DType| -> Result<Option<Tensor>> {
            if n == 0 {
                Ok(None)
            } else {
                empty(n, dt, dev).map(Some)
            }
        };
        Ok(Self {
            len,
            conv: t(s.gdn_layers * s.conv, dt)?,
            rec: t(s.gdn_layers * s.rec, DType::F32)?,
            k: t(len * s.kv, dt)?,
            v: t(len * s.kv, dt)?,
        })
    }
    pub fn bytes(&self) -> usize {
        [&self.conv, &self.rec, &self.k, &self.v]
            .iter()
            .filter_map(|t| t.as_ref())
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// An uninitialised 1-D tensor (every element is written by the kernel that fills it).
pub fn empty(n: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    match dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(d) => cu::empty(d, n, dt),
        _ => Tensor::zeros(n, dt, dev),
    }
}

/// A sequence of a pass: `len` tokens from `start` in the packed batch.
pub struct PackSeq {
    pub start: usize,
    pub len: usize,
    pub past: Option<Arc<StateCache>>,
    pub keep: Option<Arc<StateCache>>,
}

/// A pass's layout: the packed sequences plus the per-token and per-sequence metadata, uploaded once per pass.
pub struct Pack {
    pub n: usize,
    pub seqs: Vec<PackSeq>,
    pub pos: Vec<u32>,
    pub tok2seq: Vec<u32>,
    pub cu: Vec<u32>,
    pub plen: Vec<u32>,
    pub tier0: Vec<u32>,
    pub tier1: Vec<u32>,
    pub tiles: Vec<u32>,
    pub tiles64: Vec<u32>,
    #[cfg(feature = "cuda")]
    meta: Option<cu::Meta>,
}

pub const BQ: usize = 16; // attention queries per block (CUDA-core kernel; the tensor-core one takes 64)

/// A pass's sequence before packing: (token count, past, keep).
pub type SeqIn = (usize, Option<Arc<StateCache>>, Option<Arc<StateCache>>);

impl Pack {
    /// A sequence has a past or a keep, not both.
    pub fn new(seqs: Vec<SeqIn>, dev: &Device) -> Result<Self> {
        let mut out = Vec::with_capacity(seqs.len());
        let mut tiles64 = Vec::new();
        let (mut pos, mut tok2seq, mut cu, mut plen, mut tier0, mut tier1, mut tiles) = (
            Vec::new(),
            Vec::new(),
            vec![0u32],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut n = 0;
        for (b, (len, past, keep)) in seqs.into_iter().enumerate() {
            if past.is_some() && keep.is_some() {
                candle_core::bail!("pack: a sequence continues a state or builds one, not both");
            }
            if let Some(k) = &keep {
                if k.len != len {
                    candle_core::bail!("pack: keep cache of {} for {len} tokens", k.len);
                }
            }
            let p = past.as_ref().map_or(0, |s| s.len);
            pos.extend((0..len).map(|j| (p + j) as u32));
            tok2seq.extend(std::iter::repeat_n(b as u32, len));
            plen.push(p as u32);
            if keep.is_some() {
                &mut tier0
            } else {
                &mut tier1
            }
            .push(b as u32);
            for q0 in (0..len).step_by(BQ) {
                tiles.extend([b as u32, q0 as u32]);
            }
            for q0 in (0..len).step_by(64) {
                tiles64.extend([b as u32, q0 as u32]);
            }
            out.push(PackSeq {
                start: n,
                len,
                past,
                keep,
            });
            n += len;
            cu.push(n as u32);
        }
        #[allow(unused_mut)]
        let mut pack = Self {
            n,
            seqs: out,
            pos,
            tok2seq,
            cu,
            plen,
            tier0,
            tier1,
            tiles,
            tiles64,
            #[cfg(feature = "cuda")]
            meta: None,
        };
        #[cfg(feature = "cuda")]
        if let Device::Cuda(d) = dev {
            pack.meta = Some(cu::Meta::upload(&pack, d)?);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        Ok(pack)
    }
}

/// MLP activation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Silu,
    GeluTanh,
}

/// How an RMSNorm applies its weight: `F32` rounds once after `x_norm * w` (Qwen3.5, Gemma), `Rounded` rounds
/// x_norm to the model dtype first (Qwen3, Llama: `w * x_norm.to(dtype)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormMode {
    F32,
    Rounded,
}

/// One attention layer's shape and options, for qkv_prep and attention.
#[derive(Clone, Debug)]
pub struct AttnSpec {
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub q_stride: usize, // per query head in the projection (2 * hd when a gate follows each head)
    pub k_off: usize,
    pub v_off: usize,
    pub gate: bool,         // sigmoid output gate after each query head (Qwen3.5)
    pub kv_eq: bool,        // value = the raw key projection (Gemma 4 global layers)
    pub v_norm: bool,       // unscaled RMSNorm on values (Gemma 4)
    pub qn: Option<Tensor>, // q/k RMSNorm weights, f32, as applied (1 + w for zero-centred norms)
    pub kn: Option<Tensor>,
    pub mode: NormMode,
    pub eps: f64,
    pub inv_freq: Tensor, // [half] f32 (zeros where a dimension pair is not rotated)
    pub scale: f64,
    pub softcap: f64,  // 0 = none
    pub window: usize, // 0 = full causal; else keys within the last `window` positions
    pub kv_off: usize, // this layer's offset (per position) in a StateCache's k/v
    pub ld: usize,     // projection width
}

impl AttnSpec {
    fn half(&self) -> usize {
        self.inv_freq.dim(0).unwrap_or(0)
    }
}

/// One DeltaNet layer's shape: projection columns [qkv (C) | z (HV*DV) | b (HV) | a (HV)].
#[derive(Clone, Debug)]
pub struct GdnSpec {
    pub hk: usize,
    pub hv: usize,
    pub k: usize, // conv kernel
    pub ld: usize,
    pub eps: f64,
}

impl GdnSpec {
    pub fn c(&self) -> usize {
        2 * self.hk * DK + self.hv * DV
    }
    fn z_off(&self) -> usize {
        self.c()
    }
    fn b_off(&self) -> usize {
        self.c() + self.hv * DV
    }
    fn a_off(&self) -> usize {
        self.b_off() + self.hv
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Dispatch: CUDA kernel or CPU reference
// ---------------------------------------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn is_cuda(t: &Tensor) -> bool {
    matches!(t.device(), Device::Cuda(_))
}

/// (x + m rounded to the model dtype, then times `scale` rounded again when it is not 1; RMSNorm of that). Without
/// `m` the first is x itself. `w` None = unscaled.
pub fn add_norm(
    x: &Tensor,
    m: Option<&Tensor>,
    w: Option<&Tensor>,
    eps: f64,
    mode: NormMode,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let (n, h) = x.dims2()?;
    #[cfg(feature = "cuda")]
    if is_cuda(x) {
        return cu::add_norm(x, m, w, eps, mode, scale, n, h);
    }
    let dt = x.dtype();
    let xs = host(x)?;
    let ms = m.map(host).transpose()?;
    let ws = w.map(host).transpose()?;
    let r = rounder(dt);
    let mut xo = vec![0f32; n * h];
    let mut no = vec![0f32; n * h];
    for i in 0..n {
        let row = &mut xo[i * h..(i + 1) * h];
        for j in 0..h {
            row[j] = match &ms {
                Some(m) if scale != 1.0 => r(r(xs[i * h + j] + m[i * h + j]) * scale as f32),
                Some(m) => r(xs[i * h + j] + m[i * h + j]),
                None => xs[i * h + j],
            };
        }
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let s = 1.0 / (ss / h as f32 + eps as f32).sqrt();
        for j in 0..h {
            let wj = ws.as_ref().map_or(1.0, |w| w[j]);
            no[i * h + j] = match mode {
                NormMode::F32 => r(row[j] * s * wj),
                NormMode::Rounded => r(r(row[j] * s) * wj),
            };
        }
    }
    let xo = if m.is_some() {
        dev_tensor(xo, (n, h), dt, x.device())?
    } else {
        x.clone()
    };
    Ok((xo, dev_tensor(no, (n, h), dt, x.device())?))
}

/// act(gate) * up over gu = [gate | up] [N, 2I]. `round_act` rounds act(gate) to the model dtype first (as a
/// separate activation op does); otherwise one rounding at the end (a fused SwiGLU).
pub fn act_mul(gu: &Tensor, act: Act, round_act: bool) -> Result<Tensor> {
    let (n, i2) = gu.dims2()?;
    let i = i2 / 2;
    #[cfg(feature = "cuda")]
    if is_cuda(gu) {
        return cu::act_mul(gu, act, round_act, n, i);
    }
    let dt = gu.dtype();
    let g = host(gu)?;
    let r = rounder(dt);
    let mut out = vec![0f32; n * i];
    for t in 0..n {
        for j in 0..i {
            let (x, u) = (g[t * i2 + j], g[t * i2 + i + j]);
            let a = act_f(x, act);
            let a = if round_act { r(a) } else { a };
            out[t * i + j] = r(a * u);
        }
    }
    dev_tensor(out, (n, i), dt, gu.device())
}

fn act_f(x: f32, act: Act) -> f32 {
    match act {
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::GeluTanh => 0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715 * x * x * x)).tanh()),
    }
}

/// Queries, keys and values of an attention layer from its projection p [N, ld]: q/k RMSNorm, RoPE at each token's
/// position, optional value norm. -> q [N, nh*hd], k and v [N, nkv*hd] (model dtype). Keys and values of sequences
/// that build a state are also written to their caches (attention layer `la`).
pub fn qkv_prep(p: &Tensor, pack: &Pack, a: &AttnSpec) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if is_cuda(p) {
        return cu::qkv_prep(p, pack, a);
    }
    let dt = p.dtype();
    let r = rounder(dt);
    let ps = host(p)?;
    let (n, hd, half) = (pack.n, a.hd, a.half());
    let inv = host(&a.inv_freq)?;
    let (qw, kw) = (
        a.qn.as_ref().map(host).transpose()?,
        a.kn.as_ref().map(host).transpose()?,
    );
    let norm = |x: &[f32], w: Option<&Vec<f32>>| -> Vec<f32> {
        let Some(w) = w else { return x.to_vec() };
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let s = 1.0 / (ss / hd as f32 + a.eps as f32).sqrt();
        x.iter()
            .zip(w)
            .map(|(v, w)| match a.mode {
                NormMode::F32 => r(v * s * w),
                NormMode::Rounded => r(r(v * s) * w),
            })
            .collect()
    };
    let rope = |y: &[f32], pos: u32| -> Vec<f32> {
        (0..hd)
            .map(|d| {
                if d >= 2 * half {
                    return y[d];
                }
                let f = pos as f32 * inv[d % half];
                let (c, s) = (f.cos(), f.sin());
                r(if d < half {
                    y[d] * c - y[d + half] * s
                } else {
                    y[d] * c + y[d - half] * s
                })
            })
            .collect()
    };
    let (mut q, mut k, mut v) = (
        vec![0f32; n * a.nh * hd],
        vec![0f32; n * a.nkv * hd],
        vec![0f32; n * a.nkv * hd],
    );
    for t in 0..n {
        let row = &ps[t * a.ld..(t + 1) * a.ld];
        for h in 0..a.nh {
            let y = norm(&row[h * a.q_stride..h * a.q_stride + hd], qw.as_ref());
            q[(t * a.nh + h) * hd..(t * a.nh + h + 1) * hd].copy_from_slice(&rope(&y, pack.pos[t]));
        }
        for h in 0..a.nkv {
            let raw = &row[a.k_off + h * hd..a.k_off + (h + 1) * hd];
            let y = norm(raw, kw.as_ref());
            k[(t * a.nkv + h) * hd..(t * a.nkv + h + 1) * hd]
                .copy_from_slice(&rope(&y, pack.pos[t]));
            let mut vv: Vec<f32> = if a.kv_eq {
                raw.to_vec()
            } else {
                row[a.v_off + h * hd..a.v_off + (h + 1) * hd].to_vec()
            };
            if a.v_norm {
                let ss: f32 = vv.iter().map(|x| x * x).sum();
                let s = 1.0 / (ss / hd as f32 + a.eps as f32).sqrt();
                vv.iter_mut().for_each(|x| *x = r(*x * s));
            }
            v[(t * a.nkv + h) * hd..(t * a.nkv + h + 1) * hd].copy_from_slice(&vv);
        }
    }
    let w = a.nkv * hd;
    for s in &pack.seqs {
        if let Some(c) = &s.keep {
            for (src, dst) in [(&k, &c.k), (&v, &c.v)] {
                let dst = dst.as_ref().expect("attention cache");
                let rows = dev_tensor(
                    src[s.start * w..(s.start + s.len) * w].to_vec(),
                    s.len * w,
                    dt,
                    p.device(),
                )?;
                dst.slice_set(&rows, 0, s.len * a.kv_off)?;
            }
        }
    }
    let dev = p.device();
    Ok((
        dev_tensor(q, (n, a.nh * hd), dt, dev)?,
        dev_tensor(k, (n, w), dt, dev)?,
        dev_tensor(v, (n, w), dt, dev)?,
    ))
}

/// Causal attention of every packed sequence over its past keys (its state's cache) then its own keys, f32 math.
/// The output gate (sigmoid of the second half of each query head's projection) is applied when `a.gate`.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    p: &Tensor,
    pack: &Pack,
    a: &AttnSpec,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if is_cuda(q) {
        return cu::attention(q, k, v, p, pack, a);
    }
    let dt = q.dtype();
    let r = rounder(dt);
    let (qs, ks, vs, ps) = (host(q)?, host(k)?, host(v)?, host(p)?);
    let (hd, nh, nkv) = (a.hd, a.nh, a.nkv);
    let w = nkv * hd;
    let mut out = vec![0f32; pack.n * nh * hd];
    for s in &pack.seqs {
        let past = s.past.as_ref().filter(|c| c.len > 0);
        let (pk, pv, plen) = match past {
            Some(c) => (
                host(c.k.as_ref().expect("attention cache"))?,
                host(c.v.as_ref().expect("attention cache"))?,
                c.len,
            ),
            None => (Vec::new(), Vec::new(), 0),
        };
        let key = |j: usize, g: usize, which: usize| -> &[f32] {
            if j < plen {
                let src = if which == 0 { &pk } else { &pv };
                let o = plen * a.kv_off + j * w + g * hd;
                &src[o..o + hd]
            } else {
                let src = if which == 0 { &ks } else { &vs };
                let o = (s.start + j - plen) * w + g * hd;
                &src[o..o + hd]
            }
        };
        for i in 0..s.len {
            let t = s.start + i;
            let qpos = plen + i;
            let lo = if a.window > 0 {
                (qpos + 1).saturating_sub(a.window)
            } else {
                0
            };
            for h in 0..nh {
                let g = h / (nh / nkv);
                let qv = &qs[(t * nh + h) * hd..(t * nh + h + 1) * hd];
                let sc: Vec<f32> = (lo..=qpos)
                    .map(|j| {
                        let x = qv.iter().zip(key(j, g, 0)).map(|(a, b)| a * b).sum::<f32>()
                            * a.scale as f32;
                        if a.softcap > 0.0 {
                            a.softcap as f32 * (x / a.softcap as f32).tanh()
                        } else {
                            x
                        }
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for d in 0..hd {
                    let mut o = 0f32;
                    for (jj, j) in (lo..=qpos).enumerate() {
                        o += e[jj] * key(j, g, 1)[d];
                    }
                    let mut o = o / z;
                    if a.gate {
                        let gv = ps[t * a.ld + h * a.q_stride + hd + d];
                        o = r(o) * (1.0 / (1.0 + (-gv).exp()));
                    }
                    out[(t * nh + h) * hd + d] = r(o);
                }
            }
        }
    }
    dev_tensor(out, (pack.n, nh * hd), dt, q.device())
}

/// Depthwise causal conv1d + SiLU of a DeltaNet layer over each sequence, continuing its past conv state (zeros for
/// a new state). -> [N, C] in the model dtype (rounded after the conv and after the SiLU in bf16, as transformers'
/// bf16 conv1d and SiLU do). First writes the last K-1 conv inputs of each state-building sequence to its cache.
pub fn conv(p: &Tensor, w: &Tensor, pack: &Pack, g: &GdnSpec, lg: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if is_cuda(p) {
        return cu::conv(p, w, pack, g, lg);
    }
    let dt = p.dtype();
    let r = rounder(dt);
    let (ps, ws) = (host(p)?, host(w)?);
    let (c, kk, ld) = (g.c(), g.k, g.ld);
    let per = (kk - 1) * c;
    let mut out = vec![0f32; pack.n * c];
    for s in &pack.seqs {
        let prev = match &s.past {
            Some(st) => {
                let all = host(st.conv.as_ref().expect("conv cache"))?;
                all[lg * per..(lg + 1) * per].to_vec()
            }
            None => vec![0f32; per],
        };
        let x = |tau: isize, ci: usize| -> f32 {
            if tau >= 0 {
                ps[(s.start + tau as usize) * ld + ci]
            } else {
                prev[((kk as isize - 1) + tau) as usize * c + ci]
            }
        };
        if let Some(keep) = &s.keep {
            let tail: Vec<f32> = (0..kk - 1)
                .flat_map(|j| {
                    let tau = s.len as isize - (kk as isize - 1) + j as isize;
                    (0..c).map(move |ci| (tau, ci))
                })
                .map(|(tau, ci)| x(tau, ci))
                .collect();
            let t = dev_tensor(tail, per, dt, p.device())?;
            keep.conv
                .as_ref()
                .expect("conv cache")
                .slice_set(&t, 0, lg * per)?;
        }
        for t in 0..s.len {
            for ci in 0..c {
                let mut acc = 0f32;
                for k in 0..kk {
                    acc += ws[k * c + ci] * x(t as isize - (kk as isize - 1) + k as isize, ci);
                }
                let a = r(acc);
                out[(s.start + t) * c + ci] = r(a / (1.0 + (-a).exp()));
            }
        }
    }
    dev_tensor(out, (pack.n, c), dt, p.device())
}

/// The gated delta rule over each sequence (transformers' torch_recurrent_gated_delta_rule), with the gates
/// (beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)) and the q/k L2 norms computed in the kernel. Starts
/// from the past recurrent state (zeros for a new one); a state-building sequence writes its final state to its
/// cache. -> f32 [N, HV*DV].
pub fn gdn(
    qkv: &Tensor,
    p: &Tensor,
    a_neg: &Tensor,
    dt_bias: &Tensor,
    pack: &Pack,
    g: &GdnSpec,
    lg: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if is_cuda(qkv) {
        return cu::gdn(qkv, p, a_neg, dt_bias, pack, g, lg);
    }
    let (xs, ps, an, db) = (host(qkv)?, host(p)?, host(a_neg)?, host(dt_bias)?);
    let (hk, hv, c, ld) = (g.hk, g.hv, g.c(), g.ld);
    let per = hv * DK * DV;
    let mut out = vec![0f32; pack.n * hv * DV];
    for s in &pack.seqs {
        let mut st = match &s.past {
            Some(c) => host(c.rec.as_ref().expect("rec cache"))?[lg * per..(lg + 1) * per].to_vec(),
            None => vec![0f32; per],
        };
        for h in 0..hv {
            let kh = h / (hv / hk);
            let sm = &mut st[h * DK * DV..(h + 1) * DK * DV];
            for t in s.start..s.start + s.len {
                let row = &xs[t * c..(t + 1) * c];
                let q = &row[kh * DK..(kh + 1) * DK];
                let k = &row[hk * DK + kh * DK..hk * DK + (kh + 1) * DK];
                let v = &row[2 * hk * DK + h * DV..2 * hk * DK + (h + 1) * DV];
                let nq =
                    1.0 / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt() / (DK as f32).sqrt();
                let nk = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let x = ps[t * ld + g.a_off() + h] + db[h];
                let sp = x.max(0.0) + (1.0 + (-x.abs()).exp()).ln();
                let decay = (sp * an[h]).exp();
                let beta = 1.0 / (1.0 + (-ps[t * ld + g.b_off() + h]).exp());
                for j in 0..DV {
                    let mut mem = 0f32;
                    for i in 0..DK {
                        sm[i * DV + j] *= decay;
                        mem += sm[i * DV + j] * k[i] * nk;
                    }
                    let delta = (v[j] - mem) * beta;
                    let mut acc = 0f32;
                    for i in 0..DK {
                        sm[i * DV + j] += k[i] * nk * delta;
                        acc += sm[i * DV + j] * q[i] * nq;
                    }
                    out[(t * hv + h) * DV + j] = acc;
                }
            }
        }
        if let Some(keep) = &s.keep {
            let t = dev_tensor(st, per, DType::F32, qkv.device())?;
            keep.rec
                .as_ref()
                .expect("rec cache")
                .slice_set(&t, 0, lg * per)?;
        }
    }
    dev_tensor(out, (pack.n, hv * DV), DType::F32, qkv.device())
}

/// RMSNorm (weight w, f32) of each DeltaNet head's output o [N, HV*DV] f32, times silu(z) from the projection.
/// -> model dtype [N, HV*DV].
pub fn gated_norm(o: &Tensor, p: &Tensor, w: &Tensor, g: &GdnSpec) -> Result<Tensor> {
    let n = o.dim(0)?;
    #[cfg(feature = "cuda")]
    if is_cuda(o) {
        return cu::gated_norm(o, p, w, g, n);
    }
    let dt = p.dtype();
    let r = rounder(dt);
    let (os, ps, ws) = (host(o)?, host(p)?, host(w)?);
    let mut out = vec![0f32; n * g.hv * DV];
    for t in 0..n {
        for h in 0..g.hv {
            let x = &os[(t * g.hv + h) * DV..(t * g.hv + h + 1) * DV];
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let s = 1.0 / (ss / DV as f32 + g.eps as f32).sqrt();
            for d in 0..DV {
                let z = ps[t * g.ld + g.z_off() + h * DV + d];
                out[(t * g.hv + h) * DV + d] = r(x[d] * s * ws[d] * (z / (1.0 + (-z).exp())));
            }
        }
    }
    dev_tensor(out, (n, g.hv * DV), dt, o.device())
}

/// GGUF Q8_0 blocks (per 32 weights: an f16 scale then 32 int8) of a row-major [rows, cols] matrix, dequantized to
/// `dt` (each weight d * q rounded once).
pub fn dequant_q8(q: &Tensor, rows: usize, cols: usize, dt: DType) -> Result<Tensor> {
    if !cols.is_multiple_of(32) || q.elem_count() != rows * cols / 32 * 34 {
        candle_core::bail!("dequant_q8: {} bytes for [{rows}, {cols}]", q.elem_count());
    }
    #[cfg(feature = "cuda")]
    if is_cuda(q) {
        return cu::dequant_q8(q, rows, cols, dt);
    }
    let b: Vec<u8> = q.to_device(&Device::Cpu)?.to_vec1()?;
    let out: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let off = i / 32 * 34;
            let d = half::f16::from_le_bytes([b[off], b[off + 1]]).to_f32();
            d * (b[off + 2 + i % 32] as i8) as f32
        })
        .collect();
    dev_tensor(out, (rows, cols), dt, q.device())
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec1()
}

fn dev_tensor(
    v: Vec<f32>,
    shape: impl Into<candle_core::Shape>,
    dt: DType,
    dev: &Device,
) -> Result<Tensor> {
    Tensor::from_vec(v, shape, &Device::Cpu)?
        .to_dtype(dt)?
        .to_device(dev)
}

fn rounder(dt: DType) -> fn(f32) -> f32 {
    if dt == DType::BF16 {
        |x| half::bf16::from_f32(x).to_f32()
    } else {
        |x| x
    }
}

/// Serving setup for a CUDA device: keep freed memory in the stream-ordered pool across synchronisations (the
/// default release threshold of 0 hands it back to the driver at every sync, so each pass re-mapped all of its
/// activations), and stop recording two CUDA events per allocation (candle runs everything on one stream).
#[cfg(feature = "cuda")]
pub fn tune(dev: &Device) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    if let Device::Cuda(d) = dev {
        // SAFETY: `d` is an active, valid CudaDevice context; disabling driver event tracking
        // avoids redundant per-allocation events on candle's single-stream model runner.
        unsafe { d.disable_event_tracking() };
        let ctx = d.cuda_stream().context().clone();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut keep: u64 = u64::MAX;
        // SAFETY: `ctx` provides an active CUDA context; `&mut pool` and `&mut keep` are valid stack
        // pointers for FFI types (`sys::CUmemoryPool` and `u64`). Adjusts release threshold without freeing buffers.
        unsafe {
            if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device())
                .result()
                .is_ok()
            {
                let _ = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut keep as *mut u64 as *mut std::ffi::c_void,
                )
                .result();
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod cu {
    use super::{Act, AttnSpec, GdnSpec, NormMode, Pack, BQ, DK, DV};
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{
        CudaFunction, DevicePtr, LaunchConfig, PushKernelArg,
    };
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::{CudaDevice, DType, Result, Storage, Tensor};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A kernel argument.
    pub enum A {
        P(u64),
        I(i32),
        L(i64),
        F(f32),
    }

    /// The device address of a contiguous tensor's first element.
    pub fn ptr(t: &Tensor) -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        let (a, _) = l
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("kev kernel: non-contiguous input".into()))?;
        let Storage::Cuda(c) = &*st else {
            candle_core::bail!("kev kernel: not a CUDA tensor")
        };
        let base = match c.dtype() {
            DType::F32 => {
                let s = c.as_cuda_slice::<f32>()?;
                s.device_ptr(s.stream()).0
            }
            DType::BF16 => {
                let s = c.as_cuda_slice::<half::bf16>()?;
                s.device_ptr(s.stream()).0
            }
            DType::U32 => {
                let s = c.as_cuda_slice::<u32>()?;
                s.device_ptr(s.stream()).0
            }
            DType::I64 => {
                let s = c.as_cuda_slice::<i64>()?;
                s.device_ptr(s.stream()).0
            }
            DType::U8 => {
                let s = c.as_cuda_slice::<u8>()?;
                s.device_ptr(s.stream()).0
            }
            d => candle_core::bail!("kev kernel: unsupported dtype {d:?}"),
        };
        Ok(base + (a * t.dtype().size_in_bytes()) as u64)
    }

    fn opt(t: Option<&Tensor>) -> Result<u64> {
        t.map_or(Ok(0), ptr)
    }

    pub fn empty(d: &CudaDevice, n: usize, dt: DType) -> Result<Tensor> {
        let n1 = n.max(1);
        // SAFETY: `d` is an active CUDA device context. Allocates `n1` contiguous elements via the
        // CUDA stream-ordered allocator. The returned slice safely owns the allocation lifetime and is
        // populated by destination kernels before host reading.
        let s = match dt {
            DType::F32 => CudaStorage::wrap_cuda_slice(unsafe { d.alloc::<f32>(n1)? }, d.clone()),
            DType::BF16 => {
                CudaStorage::wrap_cuda_slice(unsafe { d.alloc::<half::bf16>(n1)? }, d.clone())
            }
            d => candle_core::bail!("kev: cannot allocate {d:?}"),
        };
        Tensor::from_storage(
            Storage::Cuda(s),
            n1,
            candle_core::op::BackpropOp::none(),
            false,
        )
        .narrow(0, 0, n)
    }

    fn out(t: &Tensor, shape: &[usize], dt: DType) -> Result<(Tensor, u64)> {
        let candle_core::Device::Cuda(d) = t.device() else {
            unreachable!()
        };
        let n: usize = shape.iter().product();
        let o = empty(d, n, dt)?.reshape(shape)?;
        let p = ptr(&o)?;
        Ok((o, p))
    }

    type Funcs = HashMap<(candle_core::cuda_backend::DeviceId, &'static str), CudaFunction>;
    type Modules = HashMap<
        candle_core::cuda_backend::DeviceId,
        std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaModule>,
    >;

    fn func(d: &CudaDevice, name: &'static str) -> Result<CudaFunction> {
        static FUNCS: std::sync::OnceLock<Mutex<(Funcs, Modules)>> = std::sync::OnceLock::new();
        let mut g = FUNCS
            .get_or_init(|| Mutex::new((HashMap::new(), HashMap::new())))
            .lock()
            .unwrap();
        if let Some(f) = g.0.get(&(d.id(), name)) {
            return Ok(f.clone());
        }
        let module = match g.1.get(&d.id()) {
            Some(m) => m.clone(),
            None => {
                let m = d
                    .cuda_stream()
                    .context()
                    .load_module(candle_core::cuda_backend::cudarc::nvrtc::Ptx::from_src(
                        ptx()?,
                    ))
                    .w()?;
                g.1.insert(d.id(), m.clone());
                m
            }
        };
        let f = module.load_function(name).w()?;
        g.0.insert((d.id(), name), f.clone());
        Ok(f)
    }

    pub fn launch(
        t: &Tensor,
        name: &'static str,
        grid: (u32, u32, u32),
        block: u32,
        smem: u32,
        args: &[A],
    ) -> Result<()> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        let candle_core::Device::Cuda(d) = t.device() else {
            unreachable!()
        };
        let f = func(d, name)?;
        if smem > 48 * 1024 {
            f.set_attribute(
                candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .w()?;
        }
        let stream = d.cuda_stream();
        let mut b = stream.launch_builder(&f);
        for a in args {
            match a {
                A::P(x) => b.arg(x),
                A::I(x) => b.arg(x),
                A::L(x) => b.arg(x),
                A::F(x) => b.arg(x),
            };
        }
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: `d` is an active CUDA context; function attributes and parameter bindings match
        // the compiled NVRTC kernel signature, and all referenced device buffers are valid for the launch.
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }

    fn sfx(dt: DType) -> Result<&'static str> {
        Ok(match dt {
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            d => candle_core::bail!("kev kernel: unsupported dtype {d:?}"),
        })
    }

    macro_rules! kname {
        ($base:literal, $dt:expr) => {
            match sfx($dt)? {
                "bf16" => concat!($base, "_bf16"),
                _ => concat!($base, "_f32"),
            }
        };
    }

    /// The pass metadata on the device: one u32 buffer and one pointer buffer (i64 bits).
    pub struct Meta {
        u32s: Tensor,
        ptrs: Tensor,
        o_pos: usize,
        o_cu: usize,
        o_plen: usize,
        o_tiles: usize,
        o_tiles64: usize,
        o_t0: usize,
        o_t1: usize,
        b: usize,
    }

    impl Meta {
        pub fn upload(p: &Pack, d: &CudaDevice) -> Result<Self> {
            let dev = candle_core::Device::Cuda(d.clone());
            let mut v = p.tok2seq.clone();
            let o_pos = v.len();
            v.extend(&p.pos);
            let o_cu = v.len();
            v.extend(&p.cu);
            let o_plen = v.len();
            v.extend(&p.plen);
            let o_tiles = v.len();
            v.extend(&p.tiles);
            let o_tiles64 = v.len();
            v.extend(&p.tiles64);
            let o_t0 = v.len();
            v.extend(&p.tier0);
            let o_t1 = v.len();
            v.extend(&p.tier1);
            v.push(0);
            let b = p.seqs.len();
            // rd conv, rd rec, rd k, rd v, wr conv, wr rec, wr k, wr v
            let mut ptrs = vec![0i64; 8 * b.max(1)];
            for (i, s) in p.seqs.iter().enumerate() {
                for (slot, c) in [(0, &s.past), (4, &s.keep)] {
                    if let Some(c) = c {
                        for (j, t) in [&c.conv, &c.rec, &c.k, &c.v].into_iter().enumerate() {
                            ptrs[(slot + j) * b + i] = opt(t.as_ref())? as i64;
                        }
                    }
                }
            }
            Ok(Self {
                u32s: Tensor::from_vec(v, (o_t1 + p.tier1.len() + 1,), &dev)?,
                ptrs: Tensor::from_vec(ptrs, (8 * b.max(1),), &dev)?,
                o_pos,
                o_cu,
                o_plen,
                o_tiles,
                o_tiles64,
                o_t0,
                o_t1,
                b,
            })
        }
        fn u(&self, off: usize) -> Result<u64> {
            Ok(ptr(&self.u32s)? + 4 * off as u64)
        }
        fn p(&self, slot: usize) -> Result<u64> {
            Ok(ptr(&self.ptrs)? + 8 * (slot * self.b) as u64)
        }
    }

    fn meta(p: &Pack) -> &Meta {
        p.meta.as_ref().expect("pack uploaded for CUDA")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_norm(
        x: &Tensor,
        m: Option<&Tensor>,
        w: Option<&Tensor>,
        eps: f64,
        mode: NormMode,
        scale: f64,
        n: usize,
        h: usize,
    ) -> Result<(Tensor, Tensor)> {
        let dt = x.dtype();
        let (no, pno) = out(x, &[n, h], dt)?;
        let (xo, pxo) = match m {
            Some(_) => {
                let (t, p) = out(x, &[n, h], dt)?;
                (t, p)
            }
            None => (x.clone(), 0),
        };
        launch(
            x,
            kname!("add_norm", dt),
            (n as u32, 1, 1),
            256,
            0,
            &[
                A::P(ptr(x)?),
                A::P(opt(m)?),
                A::P(opt(w)?),
                A::P(pxo),
                A::P(pno),
                A::I(h as i32),
                A::F(eps as f32),
                A::I((mode == NormMode::Rounded) as i32),
                A::F(scale as f32),
            ],
        )?;
        Ok((xo, no))
    }

    pub fn act_mul(gu: &Tensor, act: Act, round_act: bool, n: usize, i: usize) -> Result<Tensor> {
        let dt = gu.dtype();
        let (o, po) = out(gu, &[n, i], dt)?;
        let total = n * i;
        launch(
            gu,
            kname!("act_mul", dt),
            (total.div_ceil(256) as u32, 1, 1),
            256,
            0,
            &[
                A::P(ptr(gu)?),
                A::P(po),
                A::L(total as i64),
                A::I(i as i32),
                A::I((act == Act::GeluTanh) as i32),
                A::I(round_act as i32),
            ],
        )?;
        Ok(o)
    }

    pub fn qkv_prep(p: &Tensor, pack: &Pack, a: &AttnSpec) -> Result<(Tensor, Tensor, Tensor)> {
        let dt = p.dtype();
        let (n, hd) = (pack.n, a.hd);
        let (q, pq) = out(p, &[n, a.nh * hd], dt)?;
        let (k, pk) = out(p, &[n, a.nkv * hd], dt)?;
        let (v, pv) = out(p, &[n, a.nkv * hd], dt)?;
        let m = meta(pack);
        if hd > 1024 || hd % 32 != 0 {
            candle_core::bail!("qkv_prep: head_dim {hd} unsupported");
        }
        launch(
            p,
            kname!("qkv_prep", dt),
            (n as u32, (a.nh + a.nkv) as u32, 1),
            hd as u32,
            0,
            &[
                A::P(ptr(p)?),
                A::I(a.ld as i32),
                A::P(opt(a.qn.as_ref())?),
                A::P(opt(a.kn.as_ref())?),
                A::I(a.v_norm as i32),
                A::I((a.mode == NormMode::Rounded) as i32),
                A::F(a.eps as f32),
                A::P(ptr(&a.inv_freq)?),
                A::I(a.half() as i32),
                A::P(pq),
                A::P(pk),
                A::P(pv),
                A::I(a.nh as i32),
                A::I(a.nkv as i32),
                A::I(a.q_stride as i32),
                A::I(a.k_off as i32),
                A::I(a.v_off as i32),
                A::I(a.kv_eq as i32),
                A::P(m.u(0)?),
                A::P(m.u(m.o_pos)?),
                A::P(m.u(m.o_cu)?),
                A::P(m.p(6)?),
                A::P(m.p(7)?),
                A::L(a.kv_off as i64),
            ],
        )?;
        Ok((q, k, v))
    }

    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        p: &Tensor,
        pack: &Pack,
        a: &AttnSpec,
    ) -> Result<Tensor> {
        let dt = q.dtype();
        let hd = a.hd;
        let (o, po) = out(q, &[pack.n, a.nh * hd], dt)?;
        let m = meta(pack);
        // bf16: tensor cores; f32 (and head_dim 512): the f32 CUDA-core kernel
        let mma = dt == DType::BF16 && matches!(hd, 64 | 128 | 256);
        let name = match (hd, mma, sfx(dt)?) {
            (64, true, _) => "attn_mma64",
            (128, true, _) => "attn_mma128",
            (256, true, _) => "attn_mma256",
            (64, _, "bf16") => "attn64_bf16",
            (64, _, _) => "attn64_f32",
            (128, _, "bf16") => "attn128_bf16",
            (128, _, _) => "attn128_f32",
            (256, _, "bf16") => "attn256_bf16",
            (256, _, _) => "attn256_f32",
            (512, _, "bf16") => "attn512_bf16",
            (512, _, _) => "attn512_f32",
            _ => candle_core::bail!("attention: head_dim {hd} unsupported"),
        };
        let (smem, tiles, o_tiles) = if mma {
            ((128 * (hd + 8) * 2) as u32, &pack.tiles64, m.o_tiles64)
        } else {
            (((BQ + 32) * hd * 4) as u32, &pack.tiles, m.o_tiles)
        };
        let ntiles = tiles.len() / 2;
        let gate = if a.gate {
            ptr(p)? + (hd * dt.size_in_bytes()) as u64
        } else {
            0
        };
        launch(
            q,
            name,
            (ntiles as u32, a.nh as u32, 1),
            128,
            smem,
            &[
                A::P(ptr(q)?),
                A::P(ptr(k)?),
                A::P(ptr(v)?),
                A::P(po),
                A::P(gate),
                A::I(a.ld as i32),
                A::I(a.q_stride as i32),
                A::P(m.u(o_tiles)?),
                A::P(m.u(m.o_cu)?),
                A::P(m.u(m.o_plen)?),
                A::P(m.p(2)?),
                A::P(m.p(3)?),
                A::L(a.kv_off as i64),
                A::I(a.nh as i32),
                A::I(a.nkv as i32),
                A::F(a.scale as f32),
                A::F(a.softcap as f32),
                A::I(a.window as i32),
            ],
        )?;
        Ok(o)
    }

    pub fn conv(p: &Tensor, w: &Tensor, pack: &Pack, g: &GdnSpec, lg: usize) -> Result<Tensor> {
        let dt = p.dtype();
        let c = g.c();
        let m = meta(pack);
        let tail = pack.tier0.len() * (g.k - 1) * c;
        launch(
            p,
            kname!("conv_tail", dt),
            (tail.div_ceil(256) as u32, 1, 1),
            256,
            0,
            &[
                A::P(ptr(p)?),
                A::I(g.ld as i32),
                A::I(c as i32),
                A::I(g.k as i32),
                A::P(m.u(m.o_cu)?),
                A::P(m.u(m.o_t0)?),
                A::L(tail as i64),
                A::P(m.p(0)?),
                A::P(m.p(4)?),
                A::I(lg as i32),
            ],
        )?;
        let (o, po) = out(p, &[pack.n, c], dt)?;
        let total = pack.n * c;
        launch(
            p,
            kname!("conv", dt),
            (total.div_ceil(256) as u32, 1, 1),
            256,
            0,
            &[
                A::P(ptr(p)?),
                A::I(g.ld as i32),
                A::P(ptr(w)?),
                A::P(po),
                A::L(total as i64),
                A::I(c as i32),
                A::I(g.k as i32),
                A::P(m.u(0)?),
                A::P(m.u(m.o_cu)?),
                A::P(m.p(0)?),
                A::I(lg as i32),
            ],
        )?;
        Ok(o)
    }

    pub fn gdn(
        qkv: &Tensor,
        p: &Tensor,
        a_neg: &Tensor,
        dt_bias: &Tensor,
        pack: &Pack,
        g: &GdnSpec,
        lg: usize,
    ) -> Result<Tensor> {
        let dt = p.dtype();
        let m = meta(pack);
        let (o, po) = out(qkv, &[pack.n, g.hv * DV], DType::F32)?;
        for (list, off) in [(&pack.tier0, m.o_t0), (&pack.tier1, m.o_t1)] {
            launch(
                qkv,
                kname!("gdn", dt),
                ((g.hv * DV / 8) as u32, list.len() as u32, 1),
                32,
                0,
                &[
                    A::P(ptr(qkv)?),
                    A::P(ptr(p)?),
                    A::I(g.ld as i32),
                    A::I(g.a_off() as i32),
                    A::I(g.b_off() as i32),
                    A::P(ptr(a_neg)?),
                    A::P(ptr(dt_bias)?),
                    A::P(po),
                    A::P(m.u(m.o_cu)?),
                    A::P(m.u(off)?),
                    A::P(m.p(1)?),
                    A::P(m.p(5)?),
                    A::I(g.hk as i32),
                    A::I(g.hv as i32),
                    A::I(lg as i32),
                ],
            )?;
        }
        let _ = DK;
        Ok(o)
    }

    pub fn gated_norm(o: &Tensor, p: &Tensor, w: &Tensor, g: &GdnSpec, n: usize) -> Result<Tensor> {
        let dt = p.dtype();
        let (y, py) = out(p, &[n, g.hv * DV], dt)?;
        let warps = n * g.hv;
        launch(
            p,
            kname!("gated_norm", dt),
            (warps.div_ceil(8) as u32, 1, 1),
            256,
            0,
            &[
                A::P(ptr(o)?),
                A::P(ptr(p)?),
                A::I(g.ld as i32),
                A::I(g.z_off() as i32),
                A::P(ptr(w)?),
                A::P(py),
                A::L(warps as i64),
                A::I(g.hv as i32),
                A::F(g.eps as f32),
            ],
        )?;
        Ok(y)
    }

    pub fn dequant_q8(q: &Tensor, rows: usize, cols: usize, dt: DType) -> Result<Tensor> {
        let (o, po) = out(q, &[rows, cols], dt)?;
        let n = rows * cols;
        launch(
            q,
            kname!("dequant_q8", dt),
            (n.div_ceil(256) as u32, 1, 1),
            256,
            0,
            &[A::P(ptr(q)?), A::P(po), A::L(n as i64)],
        )?;
        Ok(o)
    }

    /// The kernels' PTX, compiled once.
    pub fn ptx() -> Result<&'static str> {
        static PTX: std::sync::OnceLock<std::result::Result<String, String>> =
            std::sync::OnceLock::new();
        PTX.get_or_init(|| {
            let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
                arch: Some("sm_80"),
                ..Default::default()
            };
            candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts(SRC, opts)
                .map(|p| p.to_src())
                .map_err(|e| format!("{e:?}"))
        })
        .as_deref()
        .map_err(|e| candle_core::Error::Msg(format!("nvrtc: {e}")))
    }

    const SRC: &str = include_str!("kernels.cu");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Result<Device> {
        if cfg!(feature = "cuda") {
            Device::new_cuda(0)
        } else {
            Ok(Device::Cpu)
        }
    }

    /// max |a - b| / (1 + |a|)
    fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        let (a, b) = (host(a)?, host(b)?);
        assert_eq!(a.len(), b.len());
        Ok(a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs() / (1.0 + x.abs()))
            .fold(0.0, f32::max))
    }

    fn randn(shape: &[usize], dt: DType, sd: f32) -> Result<Tensor> {
        Tensor::randn(0f32, sd, shape, &Device::Cpu)?.to_dtype(dt)
    }

    /// Inputs for one run, on the CPU; `run` copies them to its device.
    struct Case {
        dt: DType,
        cs: CacheShape,
        g: GdnSpec,
        old: [Tensor; 4],
        p_gdn: Tensor,
        conv_w: Tensor,
        a_neg: Tensor,
        dt_bias: Tensor,
        nw: Tensor,
        attn: Vec<(AttnSpec, Tensor)>,
        x: Tensor,
        m: Tensor,
        w: Tensor,
    }

    const LENS: [usize; 5] = [37, 5, 2, 19, 1];

    fn case(dt: DType) -> Result<Case> {
        let g0 = GdnSpec {
            hk: 2,
            hv: 4,
            k: 4,
            ld: 0,
            eps: 1e-6,
        };
        let g = GdnSpec {
            ld: g0.c() + g0.hv * DV + 2 * g0.hv,
            ..g0
        };
        let cs = CacheShape {
            gdn_layers: 2,
            conv: 3 * g.c(),
            rec: g.hv * DK * DV,
            kv: 2 * 64 + 3 * 128 + 2 * 256,
        };
        let n: usize = LENS.iter().sum();
        let mut attn = Vec::new();
        for (hd, gate, kv_off, window) in [
            (64usize, true, 0usize, 0usize),
            (128, false, 128, 9),
            (256, true, 512, 0),
        ] {
            let (nh, nkv) = match hd {
                128 => (6usize, 3usize),
                _ => (4, 2),
            };
            let qs = if gate { 2 * hd } else { hd };
            let ld = nh * qs + 2 * nkv * hd;
            let norm_w = || -> Result<Tensor> { randn(&[hd], DType::F32, 0.1)? + 1.0 };
            attn.push((
                AttnSpec {
                    nh,
                    nkv,
                    hd,
                    q_stride: qs,
                    k_off: nh * qs,
                    v_off: nh * qs + nkv * hd,
                    gate,
                    kv_eq: !gate,
                    v_norm: !gate,
                    qn: Some(norm_w()?),
                    kn: if gate { Some(norm_w()?) } else { None },
                    mode: if gate {
                        NormMode::F32
                    } else {
                        NormMode::Rounded
                    },
                    eps: 1e-6,
                    inv_freq: Tensor::new(&[1.0f32, 0.3, 0.01, 0.0][..], &Device::Cpu)?,
                    scale: 1.0 / (hd as f64).sqrt(),
                    softcap: if gate { 0.0 } else { 5.0 },
                    window,
                    kv_off,
                    ld,
                },
                randn(&[n, ld], dt, 1.0)?,
            ));
        }
        Ok(Case {
            dt,
            old: [
                randn(&[cs.gdn_layers * cs.conv], dt, 1.0)?,
                randn(&[cs.gdn_layers * cs.rec], DType::F32, 0.1)?,
                randn(&[7 * cs.kv], dt, 1.0)?,
                randn(&[7 * cs.kv], dt, 1.0)?,
            ],
            p_gdn: randn(&[n, g.ld], dt, 1.0)?,
            conv_w: randn(&[g.k, g.c()], DType::F32, 0.5)?,
            a_neg: (Tensor::rand(0f32, 1.0, g.hv, &Device::Cpu)? * -1.0)?,
            dt_bias: randn(&[g.hv], DType::F32, 1.0)?,
            nw: randn(&[DV], DType::F32, 1.0)?,
            attn,
            x: randn(&[n, 96], dt, 1.0)?,
            m: randn(&[n, 96], dt, 1.0)?,
            w: (randn(&[96], DType::F32, 0.1)? + 1.0)?,
            cs,
            g,
        })
    }

    /// Two states built in one pass (tier 0), rows continuing them and an older state (tier 1): every op and every
    /// cache write, on `d`.
    fn run(c: &Case, d: &Device) -> Result<Vec<Tensor>> {
        let mv = |t: &Tensor| t.to_device(d);
        let dt = c.dt;
        let old = Arc::new(StateCache {
            len: 7,
            conv: Some(mv(&c.old[0])?),
            rec: Some(mv(&c.old[1])?),
            k: Some(mv(&c.old[2])?),
            v: Some(mv(&c.old[3])?),
        });
        let s1 = Arc::new(StateCache::alloc(LENS[0], &c.cs, dt, d)?);
        let s2 = Arc::new(StateCache::alloc(LENS[2], &c.cs, dt, d)?);
        let pack = Pack::new(
            vec![
                (LENS[0], None, Some(s1.clone())),
                (LENS[1], Some(s1.clone()), None),
                (LENS[2], None, Some(s2.clone())),
                (LENS[3], Some(old.clone()), None),
                (LENS[4], Some(s2.clone()), None),
            ],
            d,
        )?;
        let mut outs = Vec::new();
        let g = &c.g;
        let p = mv(&c.p_gdn)?;
        let x = conv(&p, &mv(&c.conv_w)?, &pack, g, 1)?;
        let o = gdn(&x, &p, &mv(&c.a_neg)?, &mv(&c.dt_bias)?, &pack, g, 1)?;
        outs.extend([x, o.clone(), gated_norm(&o, &p, &mv(&c.nw)?, g)?]);
        for (a, p) in &c.attn {
            let a = AttnSpec {
                qn: a.qn.as_ref().map(mv).transpose()?,
                kn: a.kn.as_ref().map(mv).transpose()?,
                inv_freq: mv(&a.inv_freq)?,
                ..a.clone()
            };
            let p = mv(p)?;
            let (q, k, v) = qkv_prep(&p, &pack, &a)?;
            let o = attention(&q, &k, &v, &p, &pack, &a)?;
            outs.extend([q, k, v, o]);
        }
        let (x, m, w) = (mv(&c.x)?, mv(&c.m)?, mv(&c.w)?);
        let (xo, no) = add_norm(&x, Some(&m), Some(&w), 1e-6, NormMode::F32, 1.0)?;
        let (xs, ns) = add_norm(&x, Some(&m), Some(&w), 1e-6, NormMode::F32, 0.7)?;
        let (_, n2) = add_norm(&x, None, None, 1e-6, NormMode::Rounded, 1.0)?;
        outs.extend([xo, no, xs, ns, n2]);
        for act in [Act::Silu, Act::GeluTanh] {
            outs.push(act_mul(&x, act, act == Act::GeluTanh)?);
        }
        // Q8_0: 3 rows x 64 columns, scales 0.5, -0.25, 1e-3 (f16) and ramps of int8
        let mut q8 = Vec::new();
        for r in 0..3 {
            for blk in 0..2 {
                let d = [0.5f32, -0.25, 1e-3][r] * (blk + 1) as f32;
                q8.extend(half::f16::from_f32(d).to_le_bytes());
                q8.extend((0..32).map(|i| ((i * 7 - 100 + r as i32) as i8) as u8));
            }
        }
        outs.push(dequant_q8(
            &Tensor::from_vec(q8, 3 * 2 * 34, d)?,
            3,
            64,
            dt,
        )?);
        // the caches the pass built: DeltaNet layer 1 (layer 0 was not run) and both attention layers
        for s in [&s1, &s2] {
            let conv = s.conv.as_ref().unwrap();
            let rec = s.rec.as_ref().unwrap();
            outs.push(conv.narrow(0, c.cs.conv, c.cs.conv)?);
            outs.push(rec.narrow(0, c.cs.rec, c.cs.rec)?);
            outs.extend([s.k.clone().unwrap(), s.v.clone().unwrap()]);
        }
        Ok(outs)
    }

    #[test]
    fn kernels_match_reference() -> Result<()> {
        let d = dev()?;
        for dt in [DType::F32, DType::BF16] {
            let c = case(dt)?;
            let (want, got) = (run(&c, &Device::Cpu)?, run(&c, &d)?);
            // f32: summation order only; bf16: one rounding step of the output apart
            let tol = if dt == DType::F32 { 1e-4 } else { 1.6e-2 };
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                let e = rel_err(a, b)?;
                eprintln!(
                    "output {i} ({dt:?}): rel err {e:.2e}, |max| {:.3}",
                    host(a)?.iter().fold(0f32, |m, x| m.max(x.abs()))
                );
                assert!(e < tol, "output {i} ({dt:?}): rel err {e}");
            }
        }
        Ok(())
    }
}
