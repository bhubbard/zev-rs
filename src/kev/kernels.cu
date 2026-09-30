// Packed-varlen kernels for the decoder layers (see kernels.rs). Compiled with NVRTC.
// T is float or bf16 (the model dtype); math is f32. Per-pass metadata:
//   tok2seq[N], pos[N], cu[B+1] (sequence starts), plen[B] (past length), tiles[2*T] (sequence, first query),
//   tier lists (sequence ids), and per-sequence cache pointers rd[4][B] (past: conv, rec, k, v) and wr[4][B] (keep).

#define DK 128
#define DV 128
#define NEG_INF __int_as_float(0xff800000)
#define FULL 0xffffffffu

typedef unsigned short bf16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float bf2f(bf16 x) { return __uint_as_float(((unsigned)x) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
__device__ __forceinline__ float ldf(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float ldf(const bf16* p, long i) { return bf2f(p[i]); }
__device__ __forceinline__ void stf(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void stf(bf16* p, long i, float v) { p[i] = f2bf(v); }
template <typename T> __device__ __forceinline__ float rnd(float v);
template <> __device__ __forceinline__ float rnd<float>(float v) { return v; }
template <> __device__ __forceinline__ float rnd<bf16>(float v) { return bf2f(f2bf(v)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
// Sum over the block (blockDim.x a multiple of 32). Every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < nw ? part[l] : 0.f;
    return warp_sum(v);
}

// ---------------------------------------------------------------------------------------------------------------
// Residual add + RMSNorm. One block per token: xo = rnd(x + m) (when m), no = norm(xo) * w.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float residual(const T* x, const T* m, long i, float scale) {
    float s = ldf(x, i);
    if (m) {
        s = rnd<T>(s + ldf(m, i));
        if (scale != 1.f) s = rnd<T>(s * scale);
    }
    return s;
}
template <typename T>
__device__ void add_norm_body(const T* x, const T* m, const float* w, T* xo, T* no, int H, float eps, int rounded, float scale) {
    const long row = (long)blockIdx.x * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float s = residual(x, m, row + i, scale);
        ss += s * s;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float s = residual(x, m, row + i, scale);
        if (m) stf(xo, row + i, s);
        const float wi = w ? w[i] : 1.f;
        stf(no, row + i, rounded ? rnd<T>(s * r) * wi : s * r * wi);
    }
}
extern "C" __global__ void add_norm_bf16(const bf16* x, const bf16* m, const float* w, bf16* xo, bf16* no, int H, float eps, int rounded,
                                         float scale) {
    add_norm_body(x, m, w, xo, no, H, eps, rounded, scale);
}
extern "C" __global__ void add_norm_f32(const float* x, const float* m, const float* w, float* xo, float* no, int H, float eps, int rounded,
                                        float scale) {
    add_norm_body(x, m, w, xo, no, H, eps, rounded, scale);
}

// ---------------------------------------------------------------------------------------------------------------
// act(gate) * up over gu = [gate | up] per token.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void act_body(const T* gu, T* out, long total, int I, int gelu, int round_act) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const long t = idx / I;
    const int i = idx % I;
    const float g = ldf(gu, t * 2 * I + i), u = ldf(gu, t * 2 * I + I + i);
    float a = gelu ? 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) : g / (1.f + expf(-g));
    if (round_act) a = rnd<T>(a);
    stf(out, idx, a * u);
}
extern "C" __global__ void act_mul_bf16(const bf16* gu, bf16* out, long total, int I, int gelu, int round_act) {
    act_body(gu, out, total, I, gelu, round_act);
}
extern "C" __global__ void act_mul_f32(const float* gu, float* out, long total, int I, int gelu, int round_act) {
    act_body(gu, out, total, I, gelu, round_act);
}

// ---------------------------------------------------------------------------------------------------------------
// q/k RMSNorm + RoPE (rotate-half over pairs (i, i + half), i < half) + optional value norm, one block of hd threads
// per (token, head); key blocks also write the value and, for state-building sequences, the cache.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void prep_body(const T* p, int ld, const float* qw, const float* kw, int vnorm, int rounded, float eps, const float* inv,
                          int half, T* q, T* k, T* v, int nh, int nkv, int qs, int koff, int voff, int kv_eq, const u32* tok2seq,
                          const u32* pos, const u32* cu, const u64* wk, const u64* wv, long kvoff) {
    __shared__ float y[1024];
    const int hd = blockDim.x, d = threadIdx.x;
    const long n = blockIdx.x;
    const bool isq = blockIdx.y < nh;
    const int h = isq ? blockIdx.y : blockIdx.y - nh;
    const long base = n * ld + (isq ? (long)h * qs : koff + (long)h * hd);
    const float x = ldf(p, base + d);
    const float* w = isq ? qw : kw;
    float val = x;
    if (w) {
        const float ss = block_sum(x * x);
        const float xn = x * rsqrtf(ss / hd + eps);
        val = rounded ? rnd<T>(rnd<T>(xn) * w[d]) : rnd<T>(xn * w[d]);
    }
    y[d] = val;
    __syncthreads();
    float o = val;
    if (d < 2 * half) {
        const float f = (float)pos[n] * inv[d % half];
        const float c = cosf(f), s = sinf(f);
        o = d < half ? y[d] * c - y[d + half] * s : y[d] * c + y[d - half] * s;
    }
    if (isq) {
        stf(q, (n * nh + h) * hd + d, o);
        return;
    }
    const long kvw = (long)nkv * hd;
    stf(k, n * kvw + (long)h * hd + d, o);
    float vv = kv_eq ? x : ldf(p, n * ld + voff + (long)h * hd + d);
    if (vnorm) {
        const float ss = block_sum(vv * vv);
        vv = rnd<T>(vv * rsqrtf(ss / hd + eps));
    }
    stf(v, n * kvw + (long)h * hd + d, vv);
    const u32 b = tok2seq[n];
    if (wk[b]) {
        const long S = cu[b + 1] - cu[b], j = n - cu[b];
        const long at = S * kvoff + j * kvw + (long)h * hd + d;
        stf((T*)wk[b], at, o);
        stf((T*)wv[b], at, vv);
    }
}
#define PREP_ARGS(T)                                                                                                         \
    const T *p, int ld, const float *qw, const float *kw, int vnorm, int rounded, float eps, const float *inv, int half, T *q, \
        T *k, T *v, int nh, int nkv, int qs, int koff, int voff, int kv_eq, const u32 *tok2seq, const u32 *pos, const u32 *cu,  \
        const u64 *wk, const u64 *wv, long kvoff
#define PREP_CALL prep_body(p, ld, qw, kw, vnorm, rounded, eps, inv, half, q, k, v, nh, nkv, qs, koff, voff, kv_eq, tok2seq, pos, cu, wk, wv, kvoff)
extern "C" __global__ void qkv_prep_bf16(PREP_ARGS(bf16)) { PREP_CALL; }
extern "C" __global__ void qkv_prep_f32(PREP_ARGS(float)) { PREP_CALL; }

// ---------------------------------------------------------------------------------------------------------------
// Attention over packed sequences, f32 math (online softmax, no score matrix). One block (4 warps) per (tile of 16
// queries of one sequence, query head). Keys: the sequence's past (its state's cache, plen[b] positions) then its own
// tokens, causal; optional sliding window and softcap. Warp w owns queries 4w..4w+3; in the score phase lane l owns
// key l of the 32-key tile, in the value phase dims l, l+32, ... of each of its queries.
// Dynamic shared memory: Q [16][HD] f32, then K^T [HD][32] / V [32][HD] f32 (one reused buffer).
// ---------------------------------------------------------------------------------------------------------------
template <int HD, typename T>
__device__ void attn_body(const T* q, const T* kn, const T* vn, T* out, const T* gate, int gld, int gs, const u32* tiles, const u32* cu,
                          const u32* plen, const u64* rk, const u64* rv, long kvoff, int nh, int nkv, float scale, float softcap,
                          int window) {
    constexpr int BQ = 16, BK = 32, RW = 4, DPL = HD / 32;
    extern __shared__ float sm[];
    float* Qs = sm;
    float* KV = sm + BQ * HD;
    const int tile = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    const int start = cu[b], n = cu[b + 1] - start, P = plen[b];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const long kvw = (long)nkv * HD;
    const T* pk = P ? (const T*)rk[b] + (long)P * kvoff : nullptr;
    const T* pv = P ? (const T*)rv[b] + (long)P * kvoff : nullptr;
    for (int e = tid; e < BQ * HD; e += 128) {
        const int r = e / HD, d = e % HD;
        Qs[e] = q0 + r < n ? ldf(q, ((long)(start + q0 + r) * nh + h) * HD + d) : 0.f;
    }
    float m[RW], l[RW], acc[RW][DPL];
#pragma unroll
    for (int r = 0; r < RW; r++) {
        m[r] = NEG_INF;
        l[r] = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[r][i] = 0.f;
    }
    const int qlast = min(q0 + BQ, n) - 1;
    const int kend = P + qlast + 1;
    int kbeg = 0;
    if (window > 0) {
        kbeg = max(0, P + q0 - window + 1);
        kbeg -= kbeg % BK;
    }
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();
        for (int e = tid; e < BK * HD; e += 128) {  // K^T: consecutive threads take consecutive keys
            const int kk = e % BK, d = e / BK, j = k0 + kk;
            float x = 0.f;
            if (j < kend) x = j < P ? ldf(pk, (long)j * kvw + (long)g * HD + d) : ldf(kn, (long)(start + j - P) * kvw + (long)g * HD + d);
            KV[d * BK + kk] = x;
        }
        __syncthreads();
        float s[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) s[r] = 0.f;
#pragma unroll 4
        for (int d = 0; d < HD; d += 4) {
            const float k0v = KV[d * BK + lane], k1v = KV[(d + 1) * BK + lane], k2v = KV[(d + 2) * BK + lane], k3v = KV[(d + 3) * BK + lane];
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const float4 qv = *(const float4*)&Qs[(w * RW + r) * HD + d];
                s[r] += qv.x * k0v + qv.y * k1v + qv.z * k2v + qv.w * k3v;
            }
        }
        const int j = k0 + lane;
        float pr[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const int i = q0 + w * RW + r;  // query index in the sequence
            const int qp = P + (i < n ? i : 0);
            const bool ok = j <= qp && j < kend && (window <= 0 || qp - j < window);
            float x = s[r] * scale;
            if (softcap > 0.f) x = softcap * tanhf(x / softcap);
            x = ok ? x : NEG_INF;
            const float mn = fmaxf(m[r], warp_max(x));
            const float alpha = mn == NEG_INF ? 1.f : expf(m[r] - mn);
            pr[r] = ok ? expf(x - mn) : 0.f;
            l[r] = l[r] * alpha + pr[r];
            m[r] = mn;
#pragma unroll
            for (int i2 = 0; i2 < DPL; i2++) acc[r][i2] *= alpha;
        }
        __syncthreads();
        for (int e = tid; e < BK * HD; e += 128) {  // V: coalesced rows
            const int kk = e / HD, d = e % HD, jj = k0 + kk;
            float x = 0.f;
            if (jj < kend) x = jj < P ? ldf(pv, (long)jj * kvw + (long)g * HD + d) : ldf(vn, (long)(start + jj - P) * kvw + (long)g * HD + d);
            KV[kk * HD + d] = x;
        }
        __syncthreads();
        for (int jj = 0; jj < BK; jj++) {
            float pj[RW];
#pragma unroll
            for (int r = 0; r < RW; r++) pj[r] = __shfl_sync(FULL, pr[r], jj);
#pragma unroll
            for (int i2 = 0; i2 < DPL; i2++) {
                const float vv = KV[jj * HD + lane + 32 * i2];
#pragma unroll
                for (int r = 0; r < RW; r++) acc[r][i2] += pj[r] * vv;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = q0 + w * RW + r;
        const float z = warp_sum(l[r]);
        if (i >= n) continue;
        const long tok = start + i;
#pragma unroll
        for (int i2 = 0; i2 < DPL; i2++) {
            const int d = lane + 32 * i2;
            float o = acc[r][i2] / z;
            if (gate) {
                const float gv = ldf(gate, tok * gld + (long)h * gs + d);
                o = rnd<T>(o) * (1.f / (1.f + expf(-gv)));
            }
            stf(out, (tok * nh + h) * HD + d, o);
        }
    }
}
#define ATTN_ARGS(T)                                                                                                             \
    const T *q, const T *kn, const T *vn, T *out, const T *gate, int gld, int gs, const u32 *tiles, const u32 *cu, const u32 *plen, \
        const u64 *rk, const u64 *rv, long kvoff, int nh, int nkv, float scale, float softcap, int window
#define ATTN_CALL(HD) attn_body<HD>(q, kn, vn, out, gate, gld, gs, tiles, cu, plen, rk, rv, kvoff, nh, nkv, scale, softcap, window)
extern "C" __global__ void __launch_bounds__(128) attn64_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn64_f32(ATTN_ARGS(float)) { ATTN_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn128_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn128_f32(ATTN_ARGS(float)) { ATTN_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn256_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(256); }
extern "C" __global__ void __launch_bounds__(128) attn256_f32(ATTN_ARGS(float)) { ATTN_CALL(256); }
extern "C" __global__ void __launch_bounds__(128) attn512_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(512); }
extern "C" __global__ void __launch_bounds__(128) attn512_f32(ATTN_ARGS(float)) { ATTN_CALL(512); }

// ---------------------------------------------------------------------------------------------------------------
// DeltaNet causal conv1d + SiLU (thread per token and channel), continuing each sequence's past conv state.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void conv_body(const T* p, int ld, const float* w, T* out, long total, int C, int K, const u32* tok2seq, const u32* cu,
                          const u64* rd, int lg) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    const long tok = idx / C;
    const u32 b = tok2seq[tok];
    const long start = cu[b], t = tok - start;
    const T* prev = rd[b] ? (const T*)rd[b] + (long)lg * (K - 1) * C : nullptr;
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        const long tau = t - (K - 1) + k;
        const float x = tau >= 0 ? ldf(p, (start + tau) * ld + c) : (prev ? ldf(prev, (K - 1 + tau) * C + c) : 0.f);
        acc += w[k * C + c] * x;
    }
    acc = rnd<T>(acc);
    stf(out, idx, acc / (1.f + expf(-acc)));
}
extern "C" __global__ void conv_bf16(const bf16* p, int ld, const float* w, bf16* out, long total, int C, int K, const u32* tok2seq,
                                     const u32* cu, const u64* rd, int lg) {
    conv_body(p, ld, w, out, total, C, K, tok2seq, cu, rd, lg);
}
extern "C" __global__ void conv_f32(const float* p, int ld, const float* w, float* out, long total, int C, int K, const u32* tok2seq,
                                    const u32* cu, const u64* rd, int lg) {
    conv_body(p, ld, w, out, total, C, K, tok2seq, cu, rd, lg);
}

// The last K-1 conv inputs of each listed (state-building) sequence -> its conv cache, bits copied.
template <typename T>
__device__ void tail_body(const T* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd, const u64* wr, int lg) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    const int jj = (idx / C) % (K - 1);
    const u32 b = list[idx / ((long)C * (K - 1))];
    const long start = cu[b], n = cu[b + 1] - start, tau = n - (K - 1) + jj;
    const long lay = (long)lg * (K - 1) * C;
    T v;
    if (tau >= 0) v = p[(start + tau) * ld + c];
    else if (rd[b]) v = ((const T*)rd[b])[lay + (K - 1 + tau) * C + c];
    else v = (T)0;
    ((T*)wr[b])[lay + (long)jj * C + c] = v;
}
extern "C" __global__ void conv_tail_bf16(const bf16* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd,
                                          const u64* wr, int lg) {
    tail_body(p, ld, C, K, cu, list, total, rd, wr, lg);
}
extern "C" __global__ void conv_tail_f32(const float* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd,
                                         const u64* wr, int lg) {
    tail_body(p, ld, C, K, cu, list, total, rd, wr, lg);
}

// ---------------------------------------------------------------------------------------------------------------
// Gated delta rule. One warp per (listed sequence, value head, 8 value columns). Lane = column (lane / 4) x key slice
// (lane % 4); a lane keeps its 32 state entries in registers, so the delta-rule reductions are two shuffles. Gates and
// q/k L2 norms are computed here; the next token's inputs load while the current one is processed.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void gdn_body(const T* __restrict__ qkv, const T* __restrict__ proj, int ld, int a_off, int b_off, const float* a_neg,
                         const float* dt_bias, float* __restrict__ out, const u32* cu, const u32* list, const u64* rd, const u64* wr, int HK,
                         int HV, int lg) {
    const int tiles = DV / 8;
    const int h = blockIdx.x / tiles, tile = blockIdx.x % tiles, lane = threadIdx.x;
    const u32 b = list[blockIdx.y];
    const int col = lane >> 2, ks = lane & 3;
    const int j = tile * 8 + col, kh = h / (HV / HK);
    const int C = 2 * HK * DK + HV * DV;
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ float qs[DK];
    __shared__ float kv[DK];
    float S[32];
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(4 * r + ks) * DV] : 0.f;
    const float qscale = rsqrtf((float)DK), an = a_neg[h], db = dt_bias[h];
    float nq[4], nk[4], nv = 0.f, na = 0.f, nb = 0.f;
    if (len > 0) {
        const T* row = qkv + start * C;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            nq[r] = ldf(row, kh * DK + lane + 32 * r);
            nk[r] = ldf(row, HK * DK + kh * DK + lane + 32 * r);
        }
        nv = ldf(row, 2 * HK * DK + h * DV + j);
        na = ldf(proj, start * ld + a_off + h);
        nb = ldf(proj, start * ld + b_off + h);
    }
    for (int t = 0; t < len; t++) {
        float q[4], k[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = nq[r];
            k[r] = nk[r];
        }
        const float v = nv, av = na, bv = nb;
        if (t + 1 < len) {
            const long tok = start + t + 1;
            const T* row = qkv + tok * C;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                nq[r] = ldf(row, kh * DK + lane + 32 * r);
                nk[r] = ldf(row, HK * DK + kh * DK + lane + 32 * r);
            }
            nv = ldf(row, 2 * HK * DK + h * DV + j);
            na = ldf(proj, tok * ld + a_off + h);
            nb = ldf(proj, tok * ld + b_off + h);
        }
        const float x = av + db;
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        const float decay = expf(sp * an);
        const float beta = 1.f / (1.f + expf(-bv));
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            sq += q[r] * q[r];
            sk += k[r] * k[r];
        }
        sq = warp_sum(sq);
        sk = warp_sum(sk);
        const float iq = rsqrtf(sq + 1e-6f) * qscale, ik = rsqrtf(sk + 1e-6f);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qs[lane + 32 * r] = q[r] * iq;
            kv[lane + 32 * r] = k[r] * ik;
        }
        __syncwarp();
        float mem = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] *= decay;
            mem += S[r] * kv[4 * r + ks];
        }
        mem += __shfl_xor_sync(FULL, mem, 1);
        mem += __shfl_xor_sync(FULL, mem, 2);
        const float delta = (v - mem) * beta;
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] += kv[4 * r + ks] * delta;
            acc += S[r] * qs[4 * r + ks];
        }
        acc += __shfl_xor_sync(FULL, acc, 1);
        acc += __shfl_xor_sync(FULL, acc, 2);
        if (ks == 0) out[((start + t) * HV + h) * DV + j] = acc;
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(4 * r + ks) * DV] = S[r];
    }
}
#define GDN_ARGS(T)                                                                                                                 \
    const T *qkv, const T *proj, int ld, int a_off, int b_off, const float *a_neg, const float *dt_bias, float *out, const u32 *cu, \
        const u32 *list, const u64 *rd, const u64 *wr, int HK, int HV, int lg
#define GDN_CALL gdn_body(qkv, proj, ld, a_off, b_off, a_neg, dt_bias, out, cu, list, rd, wr, HK, HV, lg)
extern "C" __global__ void gdn_bf16(GDN_ARGS(bf16)) { GDN_CALL; }
extern "C" __global__ void gdn_f32(GDN_ARGS(float)) { GDN_CALL; }

// ---------------------------------------------------------------------------------------------------------------
// DeltaNet output norm: RMSNorm(o) * w * silu(z), one warp per (token, value head).
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void gnorm_body(const float* o, const T* p, int ld, int z_off, const float* w, T* out, long warps, int HV, float eps) {
    const long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gw >= warps) return;
    const long tok = gw / HV;
    const int h = gw % HV;
    const float* x = o + gw * DV;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = x[lane + 32 * i];
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(warp_sum(ss) / DV + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = lane + 32 * i;
        const float z = ldf(p, tok * ld + z_off + (long)h * DV + d);
        stf(out, gw * DV + d, v[i] * r * w[d] * (z / (1.f + expf(-z))));
    }
}
extern "C" __global__ void gated_norm_bf16(const float* o, const bf16* p, int ld, int z_off, const float* w, bf16* out, long warps, int HV,
                                           float eps) {
    gnorm_body(o, p, ld, z_off, w, out, warps, HV, eps);
}
extern "C" __global__ void gated_norm_f32(const float* o, const float* p, int ld, int z_off, const float* w, float* out, long warps, int HV,
                                          float eps) {
    gnorm_body(o, p, ld, z_off, w, out, warps, HV, eps);
}

// ---------------------------------------------------------------------------------------------------------------
// Tensor-core attention (bf16 in, f32 accumulate, P rounded to bf16 for P.V as flash-attention does). One block of
// 4 warps per (tile of 64 queries of one sequence, query head); warp w owns queries 16w..16w+15. Per 32-key tile:
// S = Q K^T with mma.m16n8k16, online softmax in registers, O += P V. Same keys, mask, window and gate as attn_body.
// Dynamic shared memory: Q [64][HD+8], K [32][HD+8], V [32][HD+8] bf16 (rows padded 16 bytes against bank
// conflicts in ldmatrix).
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) { return (unsigned)f2bf(lo) | ((unsigned)f2bf(hi) << 16); }

template <int HD>
__device__ void attn_mma_body(const bf16* q, const bf16* kn, const bf16* vn, bf16* out, const bf16* gate, int gld, int gs, const u32* tiles,
                              const u32* cu, const u32* plen, const u64* rk, const u64* rv, long kvoff, int nh, int nkv, float scale,
                              float softcap, int window) {
    constexpr int BQ = 64, BK = 32, LD = HD + 8, NB = HD / 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    bf16* Qs = (bf16*)smraw;
    bf16* Ks = Qs + BQ * LD;
    bf16* Vs = Ks + BK * LD;
    const int tile = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    const int start = cu[b], n = cu[b + 1] - start, P = plen[b];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const long kvw = (long)nkv * HD;
    const bf16* pk = P ? (const bf16*)rk[b] + (long)P * kvoff : nullptr;
    const bf16* pv = P ? (const bf16*)rv[b] + (long)P * kvoff : nullptr;
    constexpr int CH = HD / 8;  // 16-byte chunks per row
    for (int e = tid; e < BQ * CH; e += 128) {
        const int r = e / CH, c = e % CH;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) x = *(const uint4*)(q + ((long)(start + q0 + r) * nh + h) * HD + 8 * c);
        *(uint4*)(Qs + r * LD + 8 * c) = x;
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    const int qa = q0 + 16 * w + gid, qb = qa + 8;  // this thread's two query rows
    const int pa = P + min(qa, n - 1), pb = P + min(qb, n - 1);
    const int qlast = min(q0 + BQ, n) - 1;
    const int kend = P + qlast + 1;
    int kbeg = 0;
    if (window > 0) {
        kbeg = max(0, P + q0 - window + 1);
        kbeg -= kbeg % BK;
    }
    const bool warp_live = q0 + 16 * w < n;
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();
        for (int e = tid; e < 2 * BK * CH; e += 128) {
            const int which = e / (BK * CH), r = (e / CH) % BK, c = e % CH, j = k0 + r;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (j < kend) {
                const bf16* src = which == 0 ? (j < P ? pk + (long)j * kvw : kn + (long)(start + j - P) * kvw)
                                             : (j < P ? pv + (long)j * kvw : vn + (long)(start + j - P) * kvw);
                x = *(const uint4*)(src + (long)g * HD + 8 * c);
            }
            *(uint4*)((which == 0 ? Ks : Vs) + r * LD + 8 * c) = x;
        }
        __syncthreads();
        if (!warp_live) continue;
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kc = 0; kc < HD / 16; kc++) {
            unsigned a[4];
            ldsm4(a, Qs + (16 * w + (lane & 15)) * LD + 16 * kc + 8 * (lane >> 4));
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bb[4];
                ldsm4(bb, Ks + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(s[nb], a, bb[0], bb[1]);
                mma16816(s[nb + 1], a, bb[2], bb[3]);
            }
        }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int j = k0 + 8 * nb + 2 * tig + (e & 1);
                const int qp = e < 2 ? pa : pb;
                const bool ok = j <= qp && j < kend && (window <= 0 || qp - j < window);
                float x = s[nb][e] * scale;
                if (softcap > 0.f) x = softcap * tanhf(x / softcap);
                x = ok ? x : NEG_INF;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x);
                else mx1 = fmaxf(mx1, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float al0 = n0 == NEG_INF ? 1.f : expf(m0 - n0), al1 = n1 == NEG_INF ? 1.f : expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
#pragma unroll
        for (int i = 0; i < NB; i++) {
            o[i][0] *= al0;
            o[i][1] *= al0;
            o[i][2] *= al1;
            o[i][3] *= al1;
        }
        unsigned pa_[2][4];
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            float p[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float mm = e < 2 ? n0 : n1;
                p[e] = s[nb][e] == NEG_INF ? 0.f : expf(s[nb][e] - mm);
            }
            l0 += p[0] + p[1];
            l1 += p[2] + p[3];
            const int kc = nb >> 1, hi = nb & 1;
            pa_[kc][2 * hi] = pack_bf16(p[0], p[1]);
            pa_[kc][2 * hi + 1] = pack_bf16(p[2], p[3]);
        }
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {
#pragma unroll
            for (int nb = 0; nb < NB; nb += 2) {
                unsigned bb[4];
                ldsm4t(bb, Vs + (16 * kc + (lane & 15)) * LD + 8 * nb + 8 * (lane >> 4));
                mma16816(o[nb], pa_[kc], bb[0], bb[1]);
                mma16816(o[nb + 1], pa_[kc], bb[2], bb[3]);
            }
        }
    }
    if (!warp_live) return;
    l0 += __shfl_xor_sync(FULL, l0, 1);
    l0 += __shfl_xor_sync(FULL, l0, 2);
    l1 += __shfl_xor_sync(FULL, l1, 1);
    l1 += __shfl_xor_sync(FULL, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int qi = half ? qb : qa;
        if (qi >= n) continue;
        const float z = half ? l1 : l0;
        const long tok = start + qi;
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int d = 8 * nb + 2 * tig + e;
                float v = o[nb][2 * half + e] / z;
                if (gate) {
                    const float gv = bf2f(gate[tok * gld + (long)h * gs + d]);
                    v = rnd<bf16>(v) * (1.f / (1.f + expf(-gv)));
                }
                out[(tok * nh + h) * HD + d] = f2bf(v);
            }
        }
    }
}
#define ATTN_MMA_ARGS                                                                                                                     \
    const bf16 *q, const bf16 *kn, const bf16 *vn, bf16 *out, const bf16 *gate, int gld, int gs, const u32 *tiles, const u32 *cu, \
        const u32 *plen, const u64 *rk, const u64 *rv, long kvoff, int nh, int nkv, float scale, float softcap, int window
#define ATTN_MMA_CALL(HD) attn_mma_body<HD>(q, kn, vn, out, gate, gld, gs, tiles, cu, plen, rk, rv, kvoff, nh, nkv, scale, softcap, window)
extern "C" __global__ void __launch_bounds__(128) attn_mma64(ATTN_MMA_ARGS) { ATTN_MMA_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn_mma128(ATTN_MMA_ARGS) { ATTN_MMA_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn_mma256(ATTN_MMA_ARGS) { ATTN_MMA_CALL(256); }

// ---------------------------------------------------------------------------------------------------------------
// GGUF Q8_0 -> the model dtype: per 32 weights an f16 scale d then 32 int8 q; w = d * q.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ float h2f(unsigned short h) {
    const unsigned s = h >> 15, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    else v = ldexpf((float)(m | 1024u), (int)e - 25);
    return s ? -v : v;
}
template <typename T>
__device__ void dq8_body(const unsigned char* q, T* out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* blk = q + (i / 32) * 34;
    const float d = h2f((unsigned short)(blk[0] | (blk[1] << 8)));
    stf(out, i, d * (float)(signed char)blk[2 + i % 32]);
}
extern "C" __global__ void dequant_q8_bf16(const unsigned char* q, bf16* out, long n) { dq8_body(q, out, n); }
extern "C" __global__ void dequant_q8_f32(const unsigned char* q, float* out, long n) { dq8_body(q, out, n); }
