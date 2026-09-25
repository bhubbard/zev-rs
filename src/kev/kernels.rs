//! Fused CUDA kernels for the Qwen3.5 layers (compiled with NVRTC on first use), each a Candle custom op with a CPU
//! reference that the tests check the kernels against:
//! - GdnOp: the gated delta rule (DeltaNet recurrence) over right-padded rows, with the q/k L2 norms in-kernel;
//! - ConvSilu: the depthwise causal conv1d + SiLU of a DeltaNet layer, continuing a cached conv state;
//! - MaskedSoftmax: scaled softmax over attention scores with the row-form mask (a cached, right-padded state prefix,
//!   then causal) applied in-kernel, so no [rows, T, S] mask is ever built.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Layout, Result, Shape};

// The DeltaNet kernel is written for these head sizes (Qwen3.5-4B and -9B); Model::load checks the config.
pub const DK: usize = 128;
pub const DV: usize = 128;

fn contiguous(l: &Layout) -> Result<(usize, usize)> {
    l.contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("kev kernel: non-contiguous input".into()))
}

/// A CPU tensor as f32 (f32 or bf16 storage).
fn cpu_f32(s: &CpuStorage, l: &Layout) -> Result<Vec<f32>> {
    let (a, b) = contiguous(l)?;
    Ok(match s.dtype() {
        DType::F32 => s.as_slice::<f32>()?[a..b].to_vec(),
        DType::BF16 => s.as_slice::<half::bf16>()?[a..b]
            .iter()
            .map(|x| x.to_f32())
            .collect(),
        d => candle_core::bail!("kev kernel: unsupported dtype {d:?}"),
    })
}

fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------------------------------------------
// Gated delta rule
// ---------------------------------------------------------------------------------------------------------------

/// Inputs: the post-conv q|k|v [B, T, 2*HK*DK + HV*DV] f32 (q and k raw: the L2 norm and the 1/sqrt(DK) query scale
/// happen here), [decay | beta] [B, T, 2*HV] f32 (decay in log space), the initial state [B, HV, DK, DV] f32. Output,
/// flat f32: the core output [B, T, HV, DV] (zero past each row's length) then the state after each row's last real
/// token [B, HV, DK, DV]. Value head h reads key head h / (HV/HK), as transformers' repeat_interleave does.
pub struct GdnOp {
    pub lens: Vec<u32>,
    pub kheads: usize,
    pub vheads: usize,
}

/// The same recurrence as transformers' torch_recurrent_gated_delta_rule.
fn gdn_cpu(
    qkv: &[f32],
    gb: &[f32],
    s0: &[f32],
    lens: &[u32],
    t: usize,
    hk: usize,
    hv: usize,
) -> Vec<f32> {
    let b = lens.len();
    let c = 2 * hk * DK + hv * DV;
    let mut out = vec![0f32; b * t * hv * DV + b * hv * DK * DV];
    let (o, st) = out.split_at_mut(b * t * hv * DV);
    for bi in 0..b {
        for h in 0..hv {
            let kh = h / (hv / hk);
            let base = (bi * hv + h) * DK * DV;
            let s = &mut st[base..base + DK * DV];
            s.copy_from_slice(&s0[base..base + DK * DV]);
            for ti in 0..lens[bi] as usize {
                let row = &qkv[(bi * t + ti) * c..(bi * t + ti + 1) * c];
                let q = &row[kh * DK..(kh + 1) * DK];
                let k = &row[hk * DK + kh * DK..hk * DK + (kh + 1) * DK];
                let v = &row[2 * hk * DK + h * DV..2 * hk * DK + (h + 1) * DV];
                let nq =
                    1.0 / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt() / (DK as f32).sqrt();
                let nk = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let g = &gb[(bi * t + ti) * 2 * hv..];
                let (decay, beta) = (g[h].exp(), g[hv + h]);
                for j in 0..DV {
                    let mut mem = 0f32;
                    for i in 0..DK {
                        s[i * DV + j] *= decay;
                        mem += s[i * DV + j] * k[i] * nk;
                    }
                    let delta = (v[j] - mem) * beta;
                    let mut acc = 0f32;
                    for i in 0..DK {
                        s[i * DV + j] += k[i] * nk * delta;
                        acc += s[i * DV + j] * q[i] * nq;
                    }
                    o[((bi * t + ti) * hv + h) * DV + j] = acc;
                }
            }
        }
    }
    out
}

impl GdnOp {
    fn dims(&self, l1: &Layout) -> Result<(usize, usize)> {
        let (b, t, c) = l1.shape().dims3()?;
        if c != 2 * self.kheads * DK + self.vheads * DV
            || self.lens.len() != b
            || self.lens.iter().any(|&n| n as usize > t)
        {
            candle_core::bail!("gdn: bad shapes {:?} lens {:?}", l1.shape(), self.lens);
        }
        Ok((b, t))
    }
}

impl CustomOp3 for GdnOp {
    fn name(&self) -> &'static str {
        "gated-delta-rule"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (_, t) = self.dims(l1)?;
        let out = gdn_cpu(
            &cpu_f32(s1, l1)?,
            &cpu_f32(s2, l2)?,
            &cpu_f32(s3, l3)?,
            &self.lens,
            t,
            self.kheads,
            self.vheads,
        );
        let n = out.len();
        Ok((CpuStorage::F32(out), Shape::from(n)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use cuda::*;
        let (b, t) = self.dims(l1)?;
        let dev = s1.device().clone();
        let (qkv, gb, s0) = (
            view::<f32>(s1, l1)?,
            view::<f32>(s2, l2)?,
            view::<f32>(s3, l3)?,
        );
        let n = b * t * self.vheads * DV + b * self.vheads * DK * DV;
        let dst = dev.alloc_zeros::<f32>(n)?;
        let lens = upload(&dev, &self.lens)?;
        let cfg = LaunchConfig {
            grid_dim: ((self.vheads * DV / 8) as u32, b as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_func("gdn_fwd", MODULE, ptx()?)?;
        let mut lb = func.builder();
        let (ti, hk, hv) = (t as i32, self.kheads as i32, self.vheads as i32);
        lb.arg(&qkv)
            .arg(&gb)
            .arg(&s0)
            .arg(&dst)
            .arg(&lens)
            .arg(&ti)
            .arg(&hk)
            .arg(&hv);
        unsafe { lb.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(dst, dev),
            Shape::from(n),
        ))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Causal depthwise conv1d + SiLU
// ---------------------------------------------------------------------------------------------------------------

/// Inputs: the in-projection output p [B, T, ld] (model dtype; the conv input is its first `c` columns), the conv
/// state [B, K-1, c] (the K-1 inputs before the row, model dtype) and the taps w [K, c] f32. Output f32 [B, T, c]:
/// silu(sum_k w[k] x[t-K+1+k]), rounded to bf16 after the conv and after the SiLU when the model runs in bf16 (as
/// transformers' bf16 conv1d and SiLU do).
pub struct ConvSilu {
    pub c: usize,
}

impl CustomOp3 for ConvSilu {
    fn name(&self) -> &'static str {
        "conv-silu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (b, t, ld) = l1.shape().dims3()?;
        let kk = l3.shape().dims2()?.0;
        let round = s1.dtype() == DType::BF16;
        let r = |x: f32| if round { round_bf16(x) } else { x };
        let (p, st, w) = (cpu_f32(s1, l1)?, cpu_f32(s2, l2)?, cpu_f32(s3, l3)?);
        let c = self.c;
        let mut out = vec![0f32; b * t * c];
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..c {
                    let mut acc = 0f32;
                    for k in 0..kk {
                        let tau = ti as isize - (kk as isize - 1) + k as isize;
                        let x = if tau >= 0 {
                            p[(bi * t + tau as usize) * ld + ci]
                        } else {
                            st[(bi * (kk - 1) + (kk as isize - 1 + tau) as usize) * c + ci]
                        };
                        acc += w[k * c + ci] * x;
                    }
                    let a = r(acc);
                    out[(bi * t + ti) * c + ci] = r(a / (1.0 + (-a).exp()));
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((b, t, c))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use cuda::*;
        let (b, t, ld) = l1.shape().dims3()?;
        let kk = l3.shape().dims2()?.0;
        let dev = s1.device().clone();
        let n = b * t * self.c;
        let dst = unsafe { dev.alloc::<f32>(n)? };
        let w = view::<f32>(s3, l3)?;
        let args = [t as i32, ld as i32, self.c as i32, kk as i32];
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n64 = n as i64;
        match s1.dtype() {
            DType::BF16 => {
                let (p, st) = (view::<half::bf16>(s1, l1)?, view::<half::bf16>(s2, l2)?);
                let func = dev.get_or_load_custom_func("conv_silu_bf16", MODULE, ptx()?)?;
                let mut lb = func.builder();
                lb.arg(&p)
                    .arg(&st)
                    .arg(&w)
                    .arg(&dst)
                    .arg(&n64)
                    .arg(&args[0])
                    .arg(&args[1])
                    .arg(&args[2])
                    .arg(&args[3]);
                unsafe { lb.launch(cfg) }.w()?;
            }
            DType::F32 => {
                let (p, st) = (view::<f32>(s1, l1)?, view::<f32>(s2, l2)?);
                let func = dev.get_or_load_custom_func("conv_silu_f32", MODULE, ptx()?)?;
                let mut lb = func.builder();
                lb.arg(&p)
                    .arg(&st)
                    .arg(&w)
                    .arg(&dst)
                    .arg(&n64)
                    .arg(&args[0])
                    .arg(&args[1])
                    .arg(&args[2])
                    .arg(&args[3]);
                unsafe { lb.launch(cfg) }.w()?;
            }
            d => candle_core::bail!("conv-silu: unsupported dtype {d:?}"),
        }
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(dst, dev),
            Shape::from((b, t, self.c)),
        ))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Row-form masked softmax
// ---------------------------------------------------------------------------------------------------------------

/// softmax(scale * s) over the last dim of scores s [B, H, R, Sk] f32, R = rep * T query rows (query i = r % T), keys
/// = `smax` cached state positions (row b sees the first plen[b]) then the T new tokens (causal). Masked keys get 0.
pub struct MaskedSoftmax {
    pub plen: Vec<u32>,
    pub smax: usize,
    pub t: usize,
    pub scale: f32,
}

impl CustomOp1 for MaskedSoftmax {
    fn name(&self) -> &'static str {
        "masked-softmax"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let (b, h, r, sk) = l1.shape().dims4()?;
        let s = cpu_f32(s1, l1)?;
        let mut out = vec![0f32; s.len()];
        for row in 0..b * h * r {
            let (bi, i) = (row / (h * r), (row % r) % self.t);
            let ok = |j: usize| {
                if j < self.smax {
                    j < self.plen[bi] as usize
                } else {
                    j - self.smax <= i
                }
            };
            let x = &s[row * sk..(row + 1) * sk];
            let m = (0..sk)
                .filter(|&j| ok(j))
                .map(|j| x[j] * self.scale)
                .fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = (0..sk)
                .map(|j| {
                    if ok(j) {
                        (x[j] * self.scale - m).exp()
                    } else {
                        0.0
                    }
                })
                .collect();
            let z: f32 = e.iter().sum();
            for j in 0..sk {
                out[row * sk + j] = e[j] / z;
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use cuda::*;
        let (b, h, r, sk) = l1.shape().dims4()?;
        let dev = s1.device().clone();
        let s = view::<f32>(s1, l1)?;
        let dst = unsafe { dev.alloc::<f32>(b * h * r * sk)? };
        let plen = upload(&dev, &self.plen)?;
        let args = [
            h as i32,
            r as i32,
            sk as i32,
            self.smax as i32,
            self.t as i32,
        ];
        let cfg = LaunchConfig {
            grid_dim: ((b * h * r) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_func("masked_softmax", MODULE, ptx()?)?;
        let mut lb = func.builder();
        lb.arg(&s).arg(&dst).arg(&plen);
        for a in &args {
            lb.arg(a);
        }
        lb.arg(&self.scale);
        unsafe { lb.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(dst, dev),
            l1.shape().clone(),
        ))
    }
}

/// Serving setup for a CUDA device: keep freed memory in the stream-ordered pool across synchronisations (the
/// default release threshold of 0 hands it back to the driver at every sync, so each pass re-mapped all of its
/// activations), and stop recording two CUDA events per allocation (candle runs everything on one stream).
#[cfg(feature = "cuda")]
pub fn tune(dev: &candle_core::Device) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    if let candle_core::Device::Cuda(d) = dev {
        unsafe { d.disable_event_tracking() };
        let ctx = d.cuda_stream().context().clone();
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut keep: u64 = u64::MAX;
        unsafe {
            sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device())
                .result()
                .map_err(candle_core::Error::wrap)?;
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut keep as *mut u64 as *mut std::ffi::c_void,
            )
            .result()
            .map_err(candle_core::Error::wrap)?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda {
    pub use candle_core::cuda_backend::cudarc::driver::{
        CudaSlice, CudaView, LaunchConfig, PushKernelArg,
    };
    pub use candle_core::cuda_backend::WrapErr;
    use candle_core::{CudaDevice, Layout, Result};

    pub fn view<'a, T: candle_core::cuda_backend::CudaDType>(
        s: &'a candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<CudaView<'a, T>> {
        let (a, b) = super::contiguous(l)?;
        Ok(s.as_cuda_slice::<T>()?.slice(a..b))
    }

    pub fn upload(dev: &CudaDevice, v: &[u32]) -> Result<CudaSlice<u32>> {
        let mut d = unsafe { dev.alloc::<u32>(v.len().max(1))? };
        dev.memcpy_htod(v, &mut d)?;
        Ok(d)
    }

    /// The kernels' PTX, compiled once. Load a function with dev.get_or_load_custom_func(name, MODULE, ptx()?).
    pub fn ptx() -> Result<&'static str> {
        static PTX: std::sync::OnceLock<std::result::Result<String, String>> =
            std::sync::OnceLock::new();
        PTX.get_or_init(|| {
            candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(SRC)
                .map(|p| p.to_src())
                .map_err(|e| format!("{e:?}"))
        })
        .as_deref()
        .map_err(|e| candle_core::Error::Msg(format!("nvrtc: {e}")))
    }

    pub const MODULE: &str = "kev_kernels";

    const SRC: &str = r#"
#define DK 128
#define DV 128
#define NEG_INF __int_as_float(0xff800000)

// One warp per (row, value head, 8 value columns). Lane = column (lane / 4) x key slice (lane % 4); a lane keeps the
// 32 state entries (4 r + slice, column) in registers, so the delta-rule reductions are two shuffles. The next
// token's inputs are loaded while the current one is processed.
extern "C" __global__ void gdn_fwd(const float* __restrict__ qkv, const float* __restrict__ gb, const float* __restrict__ s0,
                                   float* __restrict__ out, const unsigned int* __restrict__ lens, int T, int HK, int HV) {
    const int tiles = DV / 8;
    const int h = blockIdx.x / tiles, tile = blockIdx.x % tiles, b = blockIdx.y, lane = threadIdx.x;
    const int col = lane >> 2, ks = lane & 3;
    const int j = tile * 8 + col, kh = h / (HV / HK);
    const int C = 2 * HK * DK + HV * DV;
    const long B = gridDim.y;
    __shared__ float qs[DK];
    __shared__ float kv[DK];
    float S[32];
    const long sbase = ((long)(b * HV + h)) * DK * DV + j;
    #pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0[sbase + (long)(4 * r + ks) * DV];
    const int len = lens[b];
    const float qscale = rsqrtf((float)DK);
    float nq[4], nk[4], nv = 0.f, ng = 0.f, nb = 0.f;
    if (len > 0) {
        const float* row = qkv + ((long)b * T) * C;
        #pragma unroll
        for (int r = 0; r < 4; r++) { nq[r] = row[kh * DK + lane + 32 * r]; nk[r] = row[HK * DK + kh * DK + lane + 32 * r]; }
        nv = row[2 * HK * DK + h * DV + j];
        const float* g = gb + ((long)b * T) * 2 * HV;
        ng = g[h]; nb = g[HV + h];
    }
    for (int t = 0; t < len; t++) {
        float q[4], k[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) { q[r] = nq[r]; k[r] = nk[r]; }
        const float v = nv, gl = ng, beta = nb;
        if (t + 1 < len) {
            const float* row = qkv + ((long)b * T + t + 1) * C;
            #pragma unroll
            for (int r = 0; r < 4; r++) { nq[r] = row[kh * DK + lane + 32 * r]; nk[r] = row[HK * DK + kh * DK + lane + 32 * r]; }
            nv = row[2 * HK * DK + h * DV + j];
            const float* g = gb + ((long)b * T + t + 1) * 2 * HV;
            ng = g[h]; nb = g[HV + h];
        }
        float sq = 0.f, sk = 0.f;
        #pragma unroll
        for (int r = 0; r < 4; r++) { sq += q[r] * q[r]; sk += k[r] * k[r]; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { sq += __shfl_xor_sync(0xffffffff, sq, o); sk += __shfl_xor_sync(0xffffffff, sk, o); }
        const float iq = rsqrtf(sq + 1e-6f) * qscale, ik = rsqrtf(sk + 1e-6f);
        __syncwarp();
        #pragma unroll
        for (int r = 0; r < 4; r++) { qs[lane + 32 * r] = q[r] * iq; kv[lane + 32 * r] = k[r] * ik; }
        __syncwarp();
        const float decay = expf(gl);
        float mem = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; r++) { S[r] *= decay; mem += S[r] * kv[4 * r + ks]; }
        mem += __shfl_xor_sync(0xffffffff, mem, 1);
        mem += __shfl_xor_sync(0xffffffff, mem, 2);
        const float delta = (v - mem) * beta;
        float acc = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; r++) { S[r] += kv[4 * r + ks] * delta; acc += S[r] * qs[4 * r + ks]; }
        acc += __shfl_xor_sync(0xffffffff, acc, 1);
        acc += __shfl_xor_sync(0xffffffff, acc, 2);
        if (ks == 0) out[(((long)b * T + t) * HV + h) * DV + j] = acc;
    }
    float* st = out + B * T * HV * DV + sbase;
    #pragma unroll
    for (int r = 0; r < 32; r++) st[(long)(4 * r + ks) * DV] = S[r];
}

__device__ __forceinline__ float bf2f(unsigned short x) { return __uint_as_float(((unsigned)x) << 16); }
__device__ __forceinline__ float rbf(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

#define CONV_BODY(LOAD, ROUND)                                                                             \
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (idx >= n) return;                                                                                  \
    const int c = idx % C; const long bt = idx / C; const int t = bt % T; const long b = bt / T;          \
    float acc = 0.f;                                                                                       \
    for (int k = 0; k < K; k++) {                                                                          \
        const int tau = t - (K - 1) + k;                                                                   \
        const float x = tau >= 0 ? LOAD(p[(b * T + tau) * ld + c]) : LOAD(st[(b * (K - 1) + (K - 1) + tau) * C + c]); \
        acc += w[k * C + c] * x;                                                                           \
    }                                                                                                      \
    acc = ROUND(acc);                                                                                      \
    out[idx] = ROUND(acc / (1.f + expf(-acc)));

#define ID(x) (x)
extern "C" __global__ void conv_silu_bf16(const unsigned short* __restrict__ p, const unsigned short* __restrict__ st, const float* __restrict__ w,
                                          float* __restrict__ out, long n, int T, int ld, int C, int K) { CONV_BODY(bf2f, rbf) }
extern "C" __global__ void conv_silu_f32(const float* __restrict__ p, const float* __restrict__ st, const float* __restrict__ w,
                                         float* __restrict__ out, long n, int T, int ld, int C, int K) { CONV_BODY(ID, ID) }

__device__ float block_reduce(float v, bool is_max) {
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) { float x = __shfl_xor_sync(0xffffffff, v, o); v = is_max ? fmaxf(v, x) : v + x; }
    const int w = threadIdx.x / 32, l = threadIdx.x % 32, nw = blockDim.x / 32;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < nw ? part[l] : (is_max ? NEG_INF : 0.f);
    for (int o = 16; o > 0; o >>= 1) { float x = __shfl_xor_sync(0xffffffff, v, o); v = is_max ? fmaxf(v, x) : v + x; }
    return v;
}

extern "C" __global__ void masked_softmax(const float* __restrict__ s, float* __restrict__ out, const unsigned int* __restrict__ plen,
                                          int H, int R, int Sk, int smax, int T, float scale) {
    const long row = blockIdx.x;
    const int b = row / ((long)H * R), i = (row % R) % T, pl = plen[b];
    const float* x = s + row * Sk;
    float* y = out + row * Sk;
    float m = NEG_INF;
    for (int j = threadIdx.x; j < Sk; j += blockDim.x) {
        const bool ok = j < smax ? j < pl : (j - smax) <= i;
        if (ok) m = fmaxf(m, x[j] * scale);
    }
    m = block_reduce(m, true);
    float z = 0.f;
    for (int j = threadIdx.x; j < Sk; j += blockDim.x) {
        const bool ok = j < smax ? j < pl : (j - smax) <= i;
        if (ok) z += expf(x[j] * scale - m);
    }
    z = block_reduce(z, false);
    for (int j = threadIdx.x; j < Sk; j += blockDim.x) {
        const bool ok = j < smax ? j < pl : (j - smax) <= i;
        y[j] = ok ? expf(x[j] * scale - m) / z : 0.f;
    }
}
"#;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn dev() -> Result<Device> {
        if cfg!(feature = "cuda") {
            Device::new_cuda(0)
        } else {
            Ok(Device::Cpu)
        }
    }

    fn max_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_device(&Device::Cpu)?.flatten_all()? - b.to_device(&Device::Cpu)?.flatten_all()?)?
            .abs()?
            .max(0)?
            .to_scalar::<f32>()
    }

    /// Each op on the current device against its CPU reference, with padded rows and cached state.
    #[test]
    fn kernels_match_reference() -> Result<()> {
        let (d, cpu) = (dev()?, Device::Cpu);
        // gated delta rule: row 1 padded, row 2 empty (its final state is the initial one)
        let (b, t, hk, hv) = (3usize, 37usize, 16usize, 32usize);
        let c = 2 * hk * DK + hv * DV;
        let lens = vec![37u32, 5, 0];
        let qkv = Tensor::randn(0f32, 1.0, (b, t, c), &cpu)?;
        let gb = Tensor::cat(
            &[
                (Tensor::rand(0f32, 1.0, (b, t, hv), &cpu)? * -0.5)?,
                Tensor::rand(0f32, 1.0, (b, t, hv), &cpu)?,
            ],
            2,
        )?;
        let s0 = (Tensor::randn(0f32, 1.0, (b, hv, DK, DV), &cpu)? * 0.1)?;
        let op = GdnOp {
            lens,
            kheads: hk,
            vheads: hv,
        };
        let want = qkv.apply_op3_no_bwd(&gb, &s0, &op)?;
        let got =
            qkv.to_device(&d)?
                .apply_op3_no_bwd(&gb.to_device(&d)?, &s0.to_device(&d)?, &op)?;
        assert!(
            max_err(&want, &got)? < 1e-3,
            "gdn err {}",
            max_err(&want, &got)?
        );
        let n1 = b * t * hv * DV;
        let last = want.narrow(0, n1 + 2 * hv * DK * DV, hv * DK * DV)?;
        assert_eq!(max_err(&last, &s0.narrow(0, 2, 1)?)?, 0.0);
        // conv + silu, bf16 and f32
        for dt in [DType::BF16, DType::F32] {
            let (cc, ld, kk) = (64usize, 80usize, 4usize);
            let p = Tensor::randn(0f32, 1.0, (2, 9, ld), &cpu)?.to_dtype(dt)?;
            let st = Tensor::randn(0f32, 1.0, (2, kk - 1, cc), &cpu)?.to_dtype(dt)?;
            let w = Tensor::randn(0f32, 1.0, (kk, cc), &cpu)?;
            let op = ConvSilu { c: cc };
            let want = p.apply_op3_no_bwd(&st, &w, &op)?;
            let got =
                p.to_device(&d)?
                    .apply_op3_no_bwd(&st.to_device(&d)?, &w.to_device(&d)?, &op)?;
            let tol = if dt == DType::BF16 { 2e-2 } else { 1e-5 };
            assert!(
                max_err(&want, &got)? < tol,
                "conv {dt:?} err {}",
                max_err(&want, &got)?
            );
        }
        // masked softmax: 2 rows, state 5 of 7 cached positions for row 0, 7 for row 1, then 4 causal tokens, rep 2
        let s = Tensor::randn(0f32, 3.0, (2, 3, 8, 11), &cpu)?;
        let op = MaskedSoftmax {
            plen: vec![5, 7],
            smax: 7,
            t: 4,
            scale: 0.7,
        };
        let want = s.apply_op1_no_bwd(&op)?;
        let got = s.to_device(&d)?.apply_op1_no_bwd(&op)?;
        assert!(
            max_err(&want, &got)? < 1e-5,
            "softmax err {}",
            max_err(&want, &got)?
        );
        let row: Vec<f32> = want.get(0)?.get(0)?.get(1)?.to_vec1()?; // row 0, query 1: keys 0..5 and 7..=8
        assert!(row[5] == 0.0 && row[6] == 0.0 && row[9] == 0.0 && row[8] > 0.0);
        Ok(())
    }
}
