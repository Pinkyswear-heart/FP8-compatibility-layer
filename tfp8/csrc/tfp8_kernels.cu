// TFP8 W8A16 kernels for Turing (sm_75): FP8 weights stay resident as uint8, are decoded
// inside the tile, multiplied on FP16 tensor cores (or FP32 FMA for GEMV) with FP32
// accumulation. No full FP16 weight is ever written to global memory.
//
//   y[m,n] = xs[m] * sum_g ws[n,g] * sum_{k in g} x16[m,k] * dec(wq[n,k])  + bias[n]
//
//   x16  : fp16 [M,K] row-major       (activation, already divided by a power-of-2 row scale xs)
//   wq   : uint8 [N,K] row-major      (FP8 codes, E4M3FN or E5M2 selected by fmt)
//   ws   : f32 [N, K/G]               (per-output-channel when G == K, per-K-group otherwise)
//   xs   : f32 [M]                    (activation row scale, exact power of two)
//   bias : f32 [N] (ignored when has_bias == 0)
//   y    : fp16 or bf16 [M,N], written straight from the FP32 accumulator.
//
// Decode: bit arithmetic, not a shared-memory LUT. A random 32-lane gather over a 256-entry
// table serialises on bank conflicts (measured ~1.6x slower GEMV on the 2080 Ti);
//   e5m2  -> fp16 is a pure shift: (b << 8) (same bias 15).
//   e4m3fn-> fp16 normals: ((b & 0x87) << 8) | ((b & 0x78) << 7) | (8 << 10);
//            subnormals (e == 0, m != 0, m*2^-9 is normal in fp16) from an 8-entry table.
// The LUT is kept as the fallback for unknown formats.
// Compiled at runtime with NVRTC (no host compiler needed on Windows).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
typedef unsigned char u8;
typedef unsigned long long u64;

constexpr int FMT_E4M3FN = 0, FMT_E5M2 = 1, FMT_LUT = 2;  // FMT_LUT: shared-memory table fallback

__device__ __forceinline__ void store_out(__half* y, u64 i, float v) { y[i] = __float2half_rn(v); }
__device__ __forceinline__ void store_out(__nv_bfloat16* y, u64 i, float v) { y[i] = __float2bfloat16_rn(v); }
__device__ __forceinline__ void store_out(float* y, u64 i, float v) { y[i] = v; }

__device__ __forceinline__ __half e4m3_sub(int m) {
    // fp16 bit patterns of m * 2^-9, m in 1..7 (all normal fp16 numbers)
    const unsigned short t[8] = {0x0000, 0x1800, 0x1C00, 0x1E00, 0x2000, 0x2100, 0x2200, 0x2300};
    return __ushort_as_half(t[m & 7]);
}

__device__ __forceinline__ __half dec1(u8 b, int fmt, const __half* lut) {
    if (fmt == FMT_E4M3FN) {
        // fp16 exp = e4m3 exp + 8 (bias 7 -> 15), fp16 mantissa = e4m3 mantissa << 7
        if ((b & 0x78u) != 0) {
            unsigned expmnt = (((unsigned)b & 0x7Fu) << 7) + 0x2000u;  // (exp+8)<<10 | mant<<7 (carry!)
            return __ushort_as_half((unsigned short)(expmnt | (((unsigned)b & 0x80u) << 8)));
        }
        __half s = e4m3_sub(b & 7);
        return (b & 0x80u) ? __hneg(s) : s;
    }
    if (fmt == FMT_E5M2) return __ushort_as_half((unsigned short)((unsigned)b << 8));
    return lut[b];
}

__device__ __forceinline__ void dec16(uint4 raw, int fmt, const __half* lut, __half h[16]) {
    const u8* b = reinterpret_cast<const u8*>(&raw);
#pragma unroll
    for (int i = 0; i < 16; ++i) h[i] = dec1(b[i], fmt, lut);
}

// ---------------------------------------------------------------- vector decode (fast path)
// Four FP8 codes in one 32-bit word -> four fp16 values in two packed words.
// A byte cannot be shifted into its fp16 lane in place: bits shifted by 7 (or 8) cross the
// 16-bit lane, and __vadd2 does not carry between lanes. So the even bytes (0,2) and the odd
// bytes (1,3) are processed in separate lane-aligned words, then re-interleaved with
// __byte_perm -- 2 vadd2 + 2 perm per four values, versus 4 branchy per-byte decodes.
//   e4m3fn fp16 = sign<<15 | (exp+8)<<10 | mant<<7   for every normal value (exp != 0)
//   e5m2   fp16 = the byte << 8                      (same bias 15)
__device__ __forceinline__ bool e4m3_all_normal(unsigned w) {
    return (((w & 0x78787878u) + 0x08080808u) & 0x80808080u) == 0x80808080u;  // no exp == 0 byte
}
__device__ __forceinline__ bool e4m3_has_nan(unsigned w) {
    unsigned t = w ^ 0x7F7F7F7Fu;                      // zero byte <=> 0x7F or 0xFF (NaN / -NaN)
    return (((t - 0x01010101u) & ~t & 0x80808080u) != 0);
}

// 4 codes -> out[0] = (v0,v1), out[1] = (v2,v3) as packed fp16 pairs.
__device__ __forceinline__ void dec4(unsigned w, int fmt, const __half* lut, unsigned out[2]) {
    if (fmt == FMT_E4M3FN) {
        unsigned ev = w & 0x00FF00FFu;                 // bytes 0,2 in 16-bit lanes
        unsigned od = (w >> 8) & 0x00FF00FFu;          // bytes 1,3
        unsigned lo = __vadd2((ev & 0x007F007Fu) << 7, 0x20002000u) | ((ev & 0x00800080u) << 8);
        unsigned hi = __vadd2((od & 0x007F007Fu) << 7, 0x20002000u) | ((od & 0x00800080u) << 8);
        out[0] = __byte_perm(lo, hi, 0x5410);          // lo.lo16 (v0) | hi.lo16 (v1)
        out[1] = __byte_perm(lo, hi, 0x7632);          // lo.hi16 (v2) | hi.hi16 (v3)
        return;
    }
    if (fmt == FMT_E5M2) {
        unsigned ev = (w & 0x00FF00FFu) << 8;
        unsigned od = ((w >> 8) & 0x00FF00FFu) << 8;
        out[0] = __byte_perm(ev, od, 0x5410);
        out[1] = __byte_perm(ev, od, 0x7632);
        return;
    }
    const u8* b = reinterpret_cast<const u8*>(&w);
    out[0] = __half_as_ushort(dec1(b[0], fmt, lut)) | ((unsigned)__half_as_ushort(dec1(b[1], fmt, lut)) << 16);
    out[1] = __half_as_ushort(dec1(b[2], fmt, lut)) | ((unsigned)__half_as_ushort(dec1(b[3], fmt, lut)) << 16);
}

// One uint4 (16 codes) -> 8 packed fp16 pairs. Rare e4m3fn codes (zero, subnormal, NaN) fall
// back per word, so the fast path never has to be correct for them.
__device__ __forceinline__ void dec4x4(uint4 raw, int fmt, const __half* lut, unsigned out[8]) {
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    const bool fast = (fmt != FMT_E4M3FN) || true;     // per-word check below
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (fmt == FMT_E4M3FN && !(e4m3_all_normal(w[i]) && !e4m3_has_nan(w[i]))) {
            const u8* b = reinterpret_cast<const u8*>(&w[i]);
#pragma unroll
            for (int t = 0; t < 2; ++t)
                out[2 * i + t] = __half_as_ushort(dec1(b[2 * t], fmt, lut)) |
                                 ((unsigned)__half_as_ushort(dec1(b[2 * t + 1], fmt, lut)) << 16);
        } else {
            dec4(w[i], fmt, lut, &out[2 * i]);
        }
    }
    (void)fast;
}

// ---------------------------------------------------------------- GEMV / small M (M <= 8)
// One warp per output channel; the activation tile is staged in shared memory once per block
// and shared by all warps, so the activation is read from HBM once per block, not per warp.
// Requires K % 16 == 0 and G % 16 == 0 so a lane's 16 weights never straddle a scale group.
// Widest row count the GEMV serves. Mirrored by _ext.GEMV_MAXM, which decides routing; the kernel
// itself is templated on the row width and never uses this.
constexpr int GEMV_MAXM = 8;
constexpr int GEMV_BK = 512;   // activation elements of K staged per iteration
constexpr int GEMV_WARPS = 4;

constexpr int DT_F16 = 0, DT_BF16 = 1, DT_F32 = 2;

// The GEMV accumulates in FP32, so it consumes the activation in its native dtype: no FP16
// staging pass, no row scale (that trick only exists to feed FP16 tensor cores). Decode therefore
// costs one launch on the activation side, which matters at M=1.
template <int DT>
__device__ __forceinline__ void load16(const void* base, int off, float out[16]) {
    if (DT == DT_F16) {
        const __half2* h = reinterpret_cast<const __half2*>((const __half*)base + off);
#pragma unroll
        for (int t = 0; t < 8; ++t) { float2 f = __half22float2(h[t]); out[2*t] = f.x; out[2*t+1] = f.y; }
    } else if (DT == DT_BF16) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>((const __nv_bfloat16*)base + off);
#pragma unroll
        for (int t = 0; t < 8; ++t) { float2 f = __bfloat1622float2(h[t]); out[2*t] = f.x; out[2*t+1] = f.y; }
    } else {
        const float4* f = reinterpret_cast<const float4*>((const float*)base + off);
#pragma unroll
        for (int t = 0; t < 4; ++t) { float4 v = f[t]; out[4*t] = v.x; out[4*t+1] = v.y; out[4*t+2] = v.z; out[4*t+3] = v.w; }
    }
}

// Four output channels per warp: four independent 128-bit weight loads in flight per lane per
// step, which is what decode is actually short of. One load per warp (the obvious layout) leaves
// the memory pipeline idle and measured only ~100 GB/s of the card's ~600 GB/s.
constexpr int GEMV_ROWS = 1;

// Row widths that get their own instantiation. MT is the height of the shared-memory tile and the
// extent of the accumulator, so it decides how much of the inner loop exists at all: at MT=8 the
// FMA block and the reduction are unrolled eight ways and seven of them are dead at M=1, yet they
// still issue. Measured at M=1 (min of 31 rounds, two builds each, production's own noise floor
// 1.00-1.02x): MT=1 1.16-1.24x, MT=4 1.07-1.16x, MT=8 1.00x. M=2 lands at 1.17x on MT=4, so the
// ladder is (1, 4, 8) and serves M=1, M<=4 and M<=8. It must match _ext.GEMV_MT, which exports and
// picks these; the GEMV_TAG expansion below is what instantiates them.
//
// A runtime branch to a narrow version was tried first and is much worse (0.45-0.60x): the two
// copies do not inline into one straight-line path, so the kernel keeps a stack frame and twice
// the code. __launch_bounds__(128, 7) is also no help (1.00x) -- it reaches MT=1's 72 registers
// and 7 blocks/SM but not its instruction count, which is where the win actually is.
//
// Shipped A/B against the pre-change kernel (interleaved, min of 31 rounds, noise 1.00-1.02):
// M=1 1.29-1.42x, M=2 1.19-1.33x, M=4 1.08-1.20x, M=8 0.99-1.01x.
template <typename XT, int DT, int MT, typename OutT>
__device__ void gemv_w8a16(const void* __restrict__ x, const u8* __restrict__ w,
                           const __half* __restrict__ lut, const float* __restrict__ ws,
                           const float* __restrict__ xs, const float* __restrict__ bias,
                           OutT* __restrict__ y, int M, int N, int K, int G, int has_bias,
                           int fmt) {
    __shared__ XT sX[MT][GEMV_BK];
    __shared__ __half sl[256];
    for (int i = threadIdx.x; i < 256; i += GEMV_WARPS * 32) sl[i] = lut[i];

    const int lane = threadIdx.x & 31;
    const int n0 = blockIdx.x * (GEMV_WARPS * GEMV_ROWS) + (threadIdx.x >> 5) * GEMV_ROWS;

    float acc[GEMV_ROWS][MT];
#pragma unroll
    for (int r = 0; r < GEMV_ROWS; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

    // Elements per 128-bit staging load: 8 for fp16/bf16, 4 for fp32. (This used to be clamped to
    // >= 8, so an fp32 activation only staged every other group of 4 and the GEMV read
    // uninitialised shared memory -- fp32-input outputs were wrong and nondeterministic.)
    constexpr int VEC = 16 / (int)sizeof(XT);
    constexpr int PER = GEMV_BK / VEC;

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        // Tail: K is only guaranteed to be a multiple of 16, not of GEMV_BK. Everything past
        // `rem` would read the next row (or run off the buffer) with 128-bit loads.
        const int rem = min(GEMV_BK, K - k0);
        // Stage only the rows this launch reads, capped at the instantiation's width. Staging a
        // full tile unconditionally made M=1 write seven eighths of this loop as zeros at MT=8:
        // 2048 versus 256 shared-memory stores per K-window, plus the same number of
        // predicated-off global loads.
        const int mr = min(M, MT);
        for (int i = threadIdx.x; i < mr * PER; i += GEMV_WARPS * 32) {
            int m = i / PER, c = (i % PER) * VEC;
            uint4 v = c < rem
                          ? *reinterpret_cast<const uint4*>(reinterpret_cast<const XT*>(x) + (u64)m * K + k0 + c)
                          : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sX[m][c]) = v;
        }
        __syncthreads();

        const int j = lane * 16;              // GEMV_BK / 16 == 32: exactly one chunk per lane
        const bool live = j < rem;            // this lane's 16-element chunk is inside the row
        // issue all weight loads first, then decode+accumulate: keeps them in flight together
        uint4 raw[GEMV_ROWS];
#pragma unroll
        for (int r = 0; r < GEMV_ROWS; ++r) {
            int n = n0 + r;
            raw[r] = (n < N && live) ? *reinterpret_cast<const uint4*>(w + (u64)n * K + k0 + j)
                                     : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int r = 0; r < GEMV_ROWS; ++r) {
            int n = n0 + r;
            if (n >= N || !live) continue;
            unsigned h2[8];
            dec4x4(raw[r], fmt, sl, h2);
            float wf[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2[i]));
                wf[2 * i] = f.x;
                wf[2 * i + 1] = f.y;
            }
            const float s = ws[(u64)n * (K / G) + (k0 + j) / G];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m < M) {
                    float xv[16];
                    load16<DT>(sX[m], j, xv);
                    float p = 0.f;
#pragma unroll
                    for (int t = 0; t < 16; ++t) p = fmaf(xv[t], wf[t], p);
                    acc[r][m] = fmaf(p, s, acc[r][m]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < GEMV_ROWS; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[r][m] += __shfl_xor_sync(0xffffffff, acc[r][m], o);

    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < GEMV_ROWS; ++r) {
            int n = n0 + r;
            if (n >= N) continue;
            const float bv = has_bias ? bias[n] : 0.f;
            for (int m = 0; m < M; ++m) store_out(y, (u64)m * N + n, acc[r][m] + bv);
        }
    }
}

// ---------------------------------------------------------------- fused GEMM (tensor cores)
// Block tile 64x64x32, 4 warps, each warp owns a 32x32 output tile (2x2 WMMA 16x16x16).
// FP8 weight tile: 64 rows x 32 bytes, one 128-bit load per thread, decoded into shared memory.
// Requires K % 32 == 0, G % 32 == 0. M and N tails are zero-padded.
//
// Scales, without spilling partial sums: ws[n, g] = smax[n] * r[n, g] with smax[n] = max_g ws[n, g]
// and r <= 1. r is folded into the decoded B tile (fp32 multiply, one fp16 rounding; exact for the
// usual power-of-two scales as long as the group is within ~2^9 of the row max), smax[n], the
// activation row scale and the bias are applied once in the epilogue. Per-channel weights (one
// group) skip the fold entirely and are bit-identical to the old per-group-spill kernel.
// (The spill -- 4 fragment stores + 32 smem reads + 32 FMAs per warp every G/32 K steps, plus 32
// extra accumulator registers -- made g128 ~25% slower than per-channel.)
//
// Software pipeline (Turing has no cp.async, so the prefetch goes through registers):
//   step k: issue the global loads of K-block k+1 into registers -> WMMA on smem stage k&1 ->
//           decode + store the prefetched block into stage (k+1)&1 -> one __syncthreads.
// Stage (k+1)&1 was last read in step k-1, which ended with a barrier, so the store is safe.
constexpr int BM = 64, BN = 64, BK = 32, LDS = BK + 8;

__device__ __forceinline__ float row_scale_max(const float* __restrict__ wsr, int ng) {
    float m = 0.f;
    for (int g = 0; g < ng; ++g) m = fmaxf(m, wsr[g]);
    return m;
}

template <typename OutT>
__device__ void gemm_w8a16(const __half* __restrict__ x, const u8* __restrict__ w,
                           const __half* __restrict__ lut, const float* __restrict__ ws,
                           const float* __restrict__ xs, const float* __restrict__ bias,
                           OutT* __restrict__ y, int M, int N, int K, int G, int has_bias,
                           int fmt) {
    __shared__ __half sl[256];
    __shared__ __align__(32) __half sA[2][BM][LDS];
    __shared__ __align__(32) __half sB[2][BN][LDS];  // [n][k] == col-major K x N
    __shared__ __align__(32) float sT[4][16 * 32];   // epilogue only: one 16x32 half-tile per warp

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 256; i += 128) sl[i] = lut[i];

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp >> 1) * 32, wc = (warp & 1) * 32;
    const int ng = K / G;

    // Per-thread load assignment: two 8-half A chunks (rows ar, ar+32) and one 16-byte B chunk.
    const int ar = tid >> 2, ac = (tid & 3) * 8;
    const int br = tid >> 1, bc = (tid & 1) * 16, gn = col0 + br;
    const bool a0 = row0 + ar < M, a1 = row0 + ar + 32 < M, bl = gn < N;
    const __half* xa0 = x + (u64)(row0 + ar) * K + ac;
    const __half* xa1 = x + (u64)(row0 + ar + 32) * K + ac;
    const u8* wb = w + (u64)gn * K + bc;
    const float* wsb = ws + (u64)(bl ? gn : 0) * ng;
    const bool fold = bl && ng > 1;
    const float bmax = fold ? row_scale_max(wsb, ng) : 1.f;
    // 1/smax once per block: exact for power-of-two scales, so r = ws * binv is exact too; for
    // other scales it adds one fp32 rounding, far below the fp16 rounding of the folded weight.
    const float binv = bmax > 0.f ? 1.f / bmax : 0.f;

    uint4 pa0, pa1, pb;
    float pr = 1.f;      // prefetched group ratio for the B chunk in flight
    auto fetch = [&](int k0) {
        const uint4 z = make_uint4(0, 0, 0, 0);
        pa0 = a0 ? *reinterpret_cast<const uint4*>(xa0 + k0) : z;
        pa1 = a1 ? *reinterpret_cast<const uint4*>(xa1 + k0) : z;
        pb = bl ? *reinterpret_cast<const uint4*>(wb + k0) : z;
        if (fold) pr = wsb[k0 / G];   // scale load rides with the weight load, not after the MMAs
    };
    auto stash = [&](int st) {
        *reinterpret_cast<uint4*>(&sA[st][ar][ac]) = pa0;
        *reinterpret_cast<uint4*>(&sA[st][ar + 32][ac]) = pa1;
        unsigned h2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (bl) {
            dec4x4(pb, fmt, sl, h2);
            if (fold) {        // fold r[n, g] = ws[n, g] / smax[n] (<= 1) into the decoded weights
                const float r = pr * binv;
                if (r != 1.f) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2[i]));
                        __half2 o = __floats2half2_rn(f.x * r, f.y * r);
                        h2[i] = *reinterpret_cast<unsigned*>(&o);
                    }
                }
            }
        }
        *reinterpret_cast<uint4*>(&sB[st][br][bc]) = *reinterpret_cast<uint4*>(&h2[0]);
        *reinterpret_cast<uint4*>(&sB[st][br][bc + 8]) = *reinterpret_cast<uint4*>(&h2[4]);
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    fetch(0);
    __syncthreads();          // sl (decode table) visible before the first stash
    stash(0);
    __syncthreads();

    for (int k0 = 0, st = 0; k0 < K; k0 += BK, st ^= 1) {
        const bool more = k0 + BK < K;
        if (more) fetch(k0 + BK);                 // in flight during the MMAs below
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &sA[st][wr + 16 * i][kk], LDS);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], &sB[st][wc + 16 * j][kk], LDS);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        if (more) stash(st ^ 1);
        __syncthreads();
    }

    // Epilogue: y = (acc * smax[n]) * xs[m] + bias[n]. For one group smax == ws[n] and this is the
    // exact expression the per-group-spill kernel evaluated, so per-channel output is unchanged.
    const int gc = col0 + wc + lane;
    const float s = gc < N ? (ng > 1 ? row_scale_max(ws + (u64)gc * ng, ng) : ws[gc]) : 0.f;
    const float bv = (gc < N && has_bias) ? bias[gc] : 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&sT[warp][16 * j], acc[i][j], 32, wmma::mem_row_major);
        __syncwarp();
        if (gc < N) {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int gr = row0 + wr + 16 * i + e;
                const float v = fmaf(sT[warp][e * 32 + lane], s, 0.f);
                if (gr < M) store_out(y, (u64)gr * N + gc, v * xs[gr] + bv);
            }
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------- C entry points
#define ARGS void* x, const u8* w, const __half* lut, const float* ws, const float* xs, \
             const float* bias, void* y, int M, int N, int K, int G, int has_bias, int fmt
#define FWD (const __half*)x, w, lut, ws, xs, bias
// GEMV: native activation dtype in, chosen output dtype out. xs is unused (FP32 accumulate).
// One exported kernel per row width, because MT is a compile-time property of the unrolled body;
// see GEMV_MT. NAME carries the width as _m<MT> so the launcher can pick it from M.
#define GEMV_ENTRY(IN_T, IN_DT, OUT_T, NAME, MT_)                                             extern "C" __global__ void NAME(const IN_T* x, const u8* w, const __half* lut, const float* ws,                                 const float* xs, const float* bias, OUT_T* y,                                                  int M, int N, int K, int G, int has_bias, int fmt)                 { gemv_w8a16<IN_T, IN_DT, MT_>(x, w, lut, ws, xs, bias, y, M, N, K, G, has_bias, fmt); }
#define GEMV_OUT(IN_T, IN_DT, TAG, MT_)     GEMV_ENTRY(IN_T, IN_DT, __half, tfp8_gemv_##TAG##_m##MT_##_f16, MT_)     GEMV_ENTRY(IN_T, IN_DT, __nv_bfloat16, tfp8_gemv_##TAG##_m##MT_##_bf16, MT_)     GEMV_ENTRY(IN_T, IN_DT, float, tfp8_gemv_##TAG##_m##MT_##_f32, MT_)
#define GEMV_TAG(IN_T, IN_DT, TAG)     GEMV_OUT(IN_T, IN_DT, TAG, 1)     GEMV_OUT(IN_T, IN_DT, TAG, 4)     GEMV_OUT(IN_T, IN_DT, TAG, 8)
GEMV_TAG(__half, DT_F16, in_f16)
GEMV_TAG(__nv_bfloat16, DT_BF16, in_bf16)
GEMV_TAG(float, DT_F32, in_f32)

// GEMM: FP16 activations (tensor cores) + per-row power-of-two scale xs in the epilogue.
extern "C" __global__ void __launch_bounds__(128) tfp8_gemm_f16(ARGS)  { gemm_w8a16(FWD, (__half*)y, M, N, K, G, has_bias, fmt); }
extern "C" __global__ void __launch_bounds__(128) tfp8_gemm_bf16(ARGS) { gemm_w8a16(FWD, (__nv_bfloat16*)y, M, N, K, G, has_bias, fmt); }
extern "C" __global__ void __launch_bounds__(128) tfp8_gemm_f32(ARGS)  { gemm_w8a16(FWD, (float*)y, M, N, K, G, has_bias, fmt); }

// Fused decode+scale into a caller-provided FP16 buffer (the opt-in large-M "dequant then
// cuBLAS" backend, and the test reference). out[n,k] = dec(wq[n,k]) * ws[n, k/G].
// One 16-byte weight chunk per thread, one row per block-y; no 64-bit div/mod on the hot path.
extern "C" __global__ void tfp8_dequant_f16(const u8* __restrict__ w, const __half* __restrict__ lut,
                                            const float* __restrict__ ws, __half* __restrict__ out,
                                            int N, int K, int G, int fmt) {
    const int n = blockIdx.y;
    const int nchunk = K >> 4;
    const float* wsr = ws + (u64)n * (K / G);
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < nchunk; c += gridDim.x * blockDim.x) {
        uint4 raw = *reinterpret_cast<const uint4*>(w + (u64)n * K + (c << 4));
        unsigned h2[8];
        dec4x4(raw, fmt, lut, h2);
        __half ho[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2[i]));
            ho[2 * i] = __float2half_rn(f.x * wsr[((c << 4) | (2 * i)) / G]);
            ho[2 * i + 1] = __float2half_rn(f.y * wsr[((c << 4) | (2 * i + 1)) / G]);
        }
        __half* o = out + (u64)n * K + (c << 4);
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<uint4*>(&ho[0]);
        *reinterpret_cast<uint4*>(o + 8) = *reinterpret_cast<uint4*>(&ho[8]);
    }
}

// Chunked-backend epilogue: y[m, n] = acc[m, n] * xs[m] + bias[n], cast to the output dtype, in
// one pass. acc is the contiguous FP32 [M, n] cuBLAS result for one N-slice; y points at that
// slice's column window inside the [M, ldy] output, so no full-size FP32 intermediate, no slice
// copy and no separate mul/add/cast passes (those three cost ~28% of the chunked path at M=2048).
// vec: four columns per thread with 128-bit loads (n % 4 == 0, ldy % 4 == 0, 8-byte aligned y).
template <typename OutT>
__device__ __forceinline__ void store4(OutT* y, float a, float b, float c, float d) {
    store_out(y, 0, a); store_out(y, 1, b); store_out(y, 2, c); store_out(y, 3, d);
}
__device__ __forceinline__ void store4(__half* y, float a, float b, float c, float d) {
    __half2 lo = __floats2half2_rn(a, b), hi = __floats2half2_rn(c, d);
    uint2 v = make_uint2(*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi));
    *reinterpret_cast<uint2*>(y) = v;
}
__device__ __forceinline__ void store4(__nv_bfloat16* y, float a, float b, float c, float d) {
    __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
    uint2 v = make_uint2(*reinterpret_cast<unsigned*>(&lo), *reinterpret_cast<unsigned*>(&hi));
    *reinterpret_cast<uint2*>(y) = v;
}

template <typename OutT>
__device__ void epilogue_rows(const float* __restrict__ acc, const float* __restrict__ xs,
                              const float* __restrict__ bias, OutT* __restrict__ y,
                              int M, int n, int ldy, int has_bias, int vec) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int m = blockIdx.y; m < M; m += gridDim.y) {
        const float s = xs[m];
        const float* a = acc + (u64)m * n;
        OutT* o = y + (u64)m * ldy;
        if (vec) {
            const int c = t * 4;
            if (c >= n) continue;
            float4 v = *reinterpret_cast<const float4*>(a + c);
            float4 b = has_bias ? *reinterpret_cast<const float4*>(bias + c) : make_float4(0.f, 0.f, 0.f, 0.f);
            store4(o + c, fmaf(v.x, s, b.x), fmaf(v.y, s, b.y), fmaf(v.z, s, b.z), fmaf(v.w, s, b.w));
        } else {
            for (int c = t; c < n; c += gridDim.x * blockDim.x)
                store_out(o, c, fmaf(a[c], s, has_bias ? bias[c] : 0.f));
        }
    }
}

#define EPI_ARGS const float* acc, const float* xs, const float* bias, void* y, int M, int n, int ldy,                  int has_bias, int vec
extern "C" __global__ void tfp8_epilogue_f16(EPI_ARGS)  { epilogue_rows(acc, xs, bias, (__half*)y, M, n, ldy, has_bias, vec); }
extern "C" __global__ void tfp8_epilogue_bf16(EPI_ARGS) { epilogue_rows(acc, xs, bias, (__nv_bfloat16*)y, M, n, ldy, has_bias, vec); }
extern "C" __global__ void tfp8_epilogue_f32(EPI_ARGS)  { epilogue_rows(acc, xs, bias, (float*)y, M, n, ldy, has_bias, vec); }

// Activation row prep for the GEMM path, one block per row: x16 = x / 2^e, xs = 2^e, where the
// row amax is mapped into [2^(target-1), 2^target). Exact (power of two), so BF16/FP32 -> FP16
// is lossless above around 2^-14. Fused so the GEMM preamble costs one launch, not six.
template <typename XT>
__device__ __forceinline__ float as_f32(XT v) { return static_cast<float>(v); }
__device__ __forceinline__ float as_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float as_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename XT>
__global__ void prep_rows_kernel(const XT* __restrict__ x, __half* __restrict__ x16,
                                 float* __restrict__ xs, int M, int K, float target) {
    const int m = blockIdx.x;
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(as_f32(x[(u64)m * K + k])));
    // Block-wide max: every thread only saw K / blockDim.x elements of the row.
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    __shared__ float sW[32];
    if ((threadIdx.x & 31) == 0) sW[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) amax = fmaxf(amax, sW[i]);
    const float e = amax > 0.f ? ceilf(log2f(amax)) - target : 0.f;
    const float scale = exp2f(e);
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        x16[(u64)m * K + k] = __float2half_rn(as_f32(x[(u64)m * K + k]) / scale);
    if (threadIdx.x == 0) xs[m] = scale;
}

extern "C" __global__ void tfp8_prep_bf16(const __nv_bfloat16* x, __half* x16, float* xs, int M, int K, float t)
    { prep_rows_kernel<__nv_bfloat16>(x, x16, xs, M, K, t); }
extern "C" __global__ void tfp8_prep_f32(const float* x, __half* x16, float* xs, int M, int K, float t)
    { prep_rows_kernel<float>(x, x16, xs, M, K, t); }
extern "C" __global__ void tfp8_prep_f16(const __half* x, __half* x16, float* xs, int M, int K, float t)
    { prep_rows_kernel<__half>(x, x16, xs, M, K, t); }
