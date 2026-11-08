// FP8 (LUT-decoded) x FP8 GEMM with dequant fused into the tile loop.
// Targets sm_75+ (FP16 WMMA, FP32 accumulate); no native FP8 hardware needed.
//
//   C[M,N] = A[M,K] * B[K,N] + bias[N]
//   A: fp8 row-major MxK     B: fp8 row-major KxN
//   SCALE_BLOCK_K : sa[M][K/32], sb[K/32][N]  -> must be applied per K block, inside the loop
//   SCALE_ROW_COL : sa[M],       sb[N]        -> applied once in the epilogue
//   Scales are FP32 or UE8M0 (uint8 exponent, value = 2^(e-127), 255 = NaN).
//
// Compiled at runtime with NVRTC (see tests/fp8_gemm_host.cpp): the installed MSVC is newer than
// CUDA 12.9 supports, and nvcc/cudafe++ crashes on it. NVRTC needs no host compiler.
#include <cuda_fp16.h>
#include <mma.h>
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long size_t_;

using namespace nvcuda;

enum ScaleMode { SCALE_BLOCK_K = 0, SCALE_ROW_COL = 1 };
constexpr int BM = 32, BN = 32, BK = 32, PAD = 8, SCALE_BLOCK = 32;
template <bool UE8M0>
__device__ __forceinline__ float load_scale(const void* p, int i) {
    if constexpr (UE8M0) {
        unsigned e = ((const uint8_t*)p)[i];
        return e == 255 ? __int_as_float(0x7FC00000) : ldexpf(1.0f, (int)e - 127);
    } else {
        return ((const float*)p)[i];
    }
}

// Decode 8 FP8 bytes through the shared-memory LUT, store as fp16.
__device__ __forceinline__ void decode8(const uint8_t* src, const __half* lut, __half* dst) {
    uint2 raw = *reinterpret_cast<const uint2*>(src);
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = lut[b[i]];
}

// 4 warps / block, each owns a 16x16 output tile of a 32x32 block tile.
// Requires M%32==0, N%32==0, K%32==0 (demo).
template <int MODE, bool UE8M0>
__global__ void fp8_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                const __half* __restrict__ lutA, const __half* __restrict__ lutB,
                                const void* __restrict__ sa, const void* __restrict__ sb,
                                const float* __restrict__ bias, float* __restrict__ C,
                                int M, int N, int K) {
    __shared__ __half sLut[2][256];
    __shared__ __align__(32) __half sA[BM][BK + PAD];
    __shared__ __align__(32) __half sB[BK][BN + PAD];
    __shared__ __align__(32) float sTmp[4][16 * 16];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < 256; i += 128) { sLut[0][i] = lutA[i]; sLut[1][i] = lutB[i]; }
    __syncthreads();

    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int wr = (warp >> 1) * 16, wc = (warp & 1) * 16;
    const int ld_r = tid >> 2, ld_c = (tid & 3) * 8;  // 32 rows x 32 bytes per tile

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc, tmp;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
    wmma::fill_fragment(acc, 0.0f);
    float racc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) racc[e] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        decode8(A + (unsigned long long)(row0 + ld_r) * K + k0 + ld_c, sLut[0], &sA[ld_r][ld_c]);
        decode8(B + (unsigned long long)(k0 + ld_r) * N + col0 + ld_c, sLut[1], &sB[ld_r][ld_c]);
        __syncthreads();

        if constexpr (MODE == SCALE_BLOCK_K) {
            // Per-block partial product, then scale and accumulate in FP32 registers.
            wmma::fill_fragment(tmp, 0.0f);
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::load_matrix_sync(fa, &sA[wr][kk], BK + PAD);
                wmma::load_matrix_sync(fb, &sB[kk][wc], BN + PAD);
                wmma::mma_sync(tmp, fa, fb, tmp);
            }
            wmma::store_matrix_sync(sTmp[warp], tmp, 16, wmma::mem_row_major);
            __syncwarp();
            const int kb = k0 / SCALE_BLOCK;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                int idx = lane + 32 * e, gr = row0 + wr + idx / 16, gc = col0 + wc + idx % 16;
                float s = load_scale<UE8M0>(sa, gr * (K / SCALE_BLOCK) + kb) *
                          load_scale<UE8M0>(sb, kb * N + gc);
                racc[e] += sTmp[warp][idx] * s;
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma::load_matrix_sync(fa, &sA[wr][kk], BK + PAD);
                wmma::load_matrix_sync(fb, &sB[kk][wc], BN + PAD);
                wmma::mma_sync(acc, fa, fb, acc);
            }
        }
        __syncthreads();
    }

    if constexpr (MODE == SCALE_ROW_COL) {
        wmma::store_matrix_sync(sTmp[warp], acc, 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            int idx = lane + 32 * e, gr = row0 + wr + idx / 16, gc = col0 + wc + idx % 16;
            racc[e] = sTmp[warp][idx] * load_scale<UE8M0>(sa, gr) * load_scale<UE8M0>(sb, gc);
        }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        int idx = lane + 32 * e, gr = row0 + wr + idx / 16, gc = col0 + wc + idx % 16;
        C[(unsigned long long)gr * N + gc] = racc[e] + bias[gc];
    }
}

// Plain-C entry points, one per (scale mode, scale format) instantiation.
#define ENTRY(NAME, MODE, UE)     extern "C" __global__ void NAME(const uint8_t* A, const uint8_t* B, const __half* lutA,                                     const __half* lutB, const void* sa, const void* sb,                                     const float* bias, float* C, int M, int N, int K) {         fp8_gemm_kernel<MODE, UE>(A, B, lutA, lutB, sa, sb, bias, C, M, N, K);     }
ENTRY(gemm_block_fp32,  SCALE_BLOCK_K, false)
ENTRY(gemm_block_ue8m0, SCALE_BLOCK_K, true)
ENTRY(gemm_rowcol_fp32, SCALE_ROW_COL, false)
ENTRY(gemm_rowcol_ue8m0, SCALE_ROW_COL, true)
