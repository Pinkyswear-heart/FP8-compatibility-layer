// Host test driver: builds LUTs, quantizes data, runs the fused kernel, checks against a double reference.
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "fp8.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

enum ScaleMode { SCALE_BLOCK_K = 0, SCALE_ROW_COL = 1 };
constexpr int SCALE_BLOCK = 32, BM = 32, BN = 32;
#define NV(x) do { nvrtcResult r_ = (x); if (r_ != NVRTC_SUCCESS) { \
    printf("NVRTC error %s at %s:%d\n", nvrtcGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    printf("CUDA driver error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

struct Case {
    const char* name;
    Fp8Format fa, fb;
    ScaleMode mode;
    bool ue8m0;
};

static CUmodule g_mod;

// Compile the kernel source with NVRTC straight to a native cubin and load it.
static void load_kernels(const char* path, int major, int minor) {
    std::ifstream in(path);
    if (!in) { printf("cannot open %s (run from the demo dir or pass the path)\n", path); exit(1); }
    std::stringstream ss; ss << in.rdbuf();
    std::string src = ss.str();
    nvrtcProgram prog;
    NV(nvrtcCreateProgram(&prog, src.c_str(), "fp8_gemm_kernel.cu", 0, nullptr, nullptr));
    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    std::string inc = std::string("-I") + CUDA_INCLUDE_DIR;
    const char* opts[] = {arch.c_str(), "--std=c++17", "-default-device", inc.c_str()};
    nvrtcResult r = nvrtcCompileProgram(prog, 4, opts);
    size_t n; nvrtcGetProgramLogSize(prog, &n);
    if (n > 1) { std::string log(n, 0); nvrtcGetProgramLog(prog, &log[0]); printf("%s\n", log.c_str()); }
    NV(r);
    NV(nvrtcGetCUBINSize(prog, &n));
    std::string cubin(n, 0);
    NV(nvrtcGetCUBIN(prog, &cubin[0]));
    nvrtcDestroyProgram(&prog);
    CU(cuModuleLoadData(&g_mod, cubin.data()));
}

static void launch_fp8_gemm(int mode, bool ue8m0, const void* A, const void* B, const void* lutA,
                            const void* lutB, const void* sa, const void* sb, const float* bias,
                            float* C, int M, int N, int K) {
    static const char* names[2][2] = {{"gemm_block_fp32", "gemm_block_ue8m0"},
                                      {"gemm_rowcol_fp32", "gemm_rowcol_ue8m0"}};
    if (M % BM || N % BN || K % 32) { fprintf(stderr, "launch_fp8_gemm: M/N/K must be multiples of %d/%d/32\n", BM, BN); exit(1); }
    CUfunction fn; CU(cuModuleGetFunction(&fn, g_mod, names[mode][ue8m0]));
    void* args[] = {&A, &B, &lutA, &lutB, &sa, &sb, &bias, &C, &M, &N, &K};
    CU(cuLaunchKernel(fn, N / BN, M / BM, 1, 128, 1, 1, 0, nullptr, args, nullptr));
}

template <typename T> static T* to_dev(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Scale so that amax maps to max_finite. UE8M0 rounds the exponent up (no overflow).
static float make_scale(float amax, const Fp8Format& f, bool ue8m0, uint8_t* e_out) {
    float s = amax > 0 ? amax / f.max_finite : 1.0f;
    if (ue8m0) {
        int e = (int)std::ceil(std::log2((double)s));
        e = std::max(-127, std::min(127, e));
        *e_out = (uint8_t)(e + 127);
        return std::ldexp(1.0f, e);
    }
    return s;
}

static bool run_case(const Case& c, int M, int N, int K) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> ex(-4, 4);
    const int KB = K / SCALE_BLOCK;

    // Original FP32 data; per-(row,K-block) magnitude varies so per-block scaling matters.
    std::vector<float> Af((size_t)M * K), Bf((size_t)K * N), bias(N);
    for (int m = 0; m < M; ++m)
        for (int kb = 0; kb < KB; ++kb) {
            float g = std::ldexp(1.0f, ex(rng));
            for (int k = 0; k < SCALE_BLOCK; ++k) Af[(size_t)m * K + kb * SCALE_BLOCK + k] = nd(rng) * g;
        }
    for (int n = 0; n < N; ++n)
        for (int kb = 0; kb < KB; ++kb) {
            float g = std::ldexp(1.0f, ex(rng));
            for (int k = 0; k < SCALE_BLOCK; ++k) Bf[(size_t)(kb * SCALE_BLOCK + k) * N + n] = nd(rng) * g;
        }
    for (auto& b : bias) b = nd(rng);

    // Quantize and keep the dequantized values for the reference.
    std::vector<uint8_t> Aq(Af.size()), Bq(Bf.size());
    std::vector<float> saF, sbF;  std::vector<uint8_t> saE, sbE;
    std::vector<double> Ad(Af.size()), Bd(Bf.size());
    uint8_t eb = 0;
    auto quant = [&](float x, float s, const Fp8Format& f) {
        uint8_t q = fp8_encode(x / s, f);
        return std::make_pair(q, (double)fp8_decode(q, f) * (double)s);
    };
    // A scales (per row x K block, or per row)
    int nsa = c.mode == SCALE_BLOCK_K ? M * KB : M, nsb = c.mode == SCALE_BLOCK_K ? KB * N : N;
    saF.resize(nsa); saE.resize(nsa); sbF.resize(nsb); sbE.resize(nsb);
    for (int m = 0; m < M; ++m) {
        int nblk = c.mode == SCALE_BLOCK_K ? KB : 1, len = c.mode == SCALE_BLOCK_K ? SCALE_BLOCK : K;
        for (int b = 0; b < nblk; ++b) {
            float amax = 0;
            for (int k = 0; k < len; ++k) amax = std::max(amax, std::fabs(Af[(size_t)m * K + b * len + k]));
            float s = make_scale(amax, c.fa, c.ue8m0, &eb);
            saF[m * nblk + b] = s; saE[m * nblk + b] = eb;
            for (int k = 0; k < len; ++k) {
                size_t i = (size_t)m * K + b * len + k;
                auto qd = quant(Af[i], s, c.fa); Aq[i] = qd.first; Ad[i] = qd.second;
            }
        }
    }
    for (int n = 0; n < N; ++n) {
        int nblk = c.mode == SCALE_BLOCK_K ? KB : 1, len = c.mode == SCALE_BLOCK_K ? SCALE_BLOCK : K;
        for (int b = 0; b < nblk; ++b) {
            float amax = 0;
            for (int k = 0; k < len; ++k) amax = std::max(amax, std::fabs(Bf[(size_t)(b * len + k) * N + n]));
            float s = make_scale(amax, c.fb, c.ue8m0, &eb);
            sbF[b * N + n] = s; sbE[b * N + n] = eb;
            for (int k = 0; k < len; ++k) {
                size_t i = (size_t)(b * len + k) * N + n;
                auto qd = quant(Bf[i], s, c.fb); Bq[i] = qd.first; Bd[i] = qd.second;
            }
        }
    }

    // Reference in double: dequantized operands, exact products.
    std::vector<double> ref((size_t)M * N), truth((size_t)M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double r = bias[n], t = bias[n];
            for (int k = 0; k < K; ++k) {
                r += Ad[(size_t)m * K + k] * Bd[(size_t)k * N + n];
                t += (double)Af[(size_t)m * K + k] * Bf[(size_t)k * N + n];
            }
            ref[(size_t)m * N + n] = r; truth[(size_t)m * N + n] = t;
        }

    // Two LUTs (A and B formats may differ).
    uint16_t lutA_h[256], lutB_h[256];
    fp8_build_lut(c.fa, lutA_h); fp8_build_lut(c.fb, lutB_h);
    uint16_t *lutA, *lutB;
    CK(cudaMalloc(&lutA, 512)); CK(cudaMalloc(&lutB, 512));
    CK(cudaMemcpy(lutA, lutA_h, 512, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(lutB, lutB_h, 512, cudaMemcpyHostToDevice));

    uint8_t *dA = to_dev(Aq), *dB = to_dev(Bq);
    float* dbias = to_dev(bias);
    void *dsa, *dsb;
    if (c.ue8m0) { dsa = to_dev(saE); dsb = to_dev(sbE); } else { dsa = to_dev(saF); dsb = to_dev(sbF); }
    float* dC; CK(cudaMalloc(&dC, (size_t)M * N * 4));

    launch_fp8_gemm(c.mode, c.ue8m0, dA, dB, lutA, lutB, dsa, dsb, dbias, dC, M, N, K);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<float> C((size_t)M * N);
    CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
    double max_ref = 0, max_err = 0, max_qerr = 0;
    for (size_t i = 0; i < C.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        max_err = std::max(max_err, std::fabs((double)C[i] - ref[i]));
        max_qerr = std::max(max_qerr, std::fabs(ref[i] - truth[i]));
    }
    double rel = max_err / max_ref;
    bool ok = rel < 1e-5;
    printf("%-34s A=%-6s B=%-6s  kernel-vs-ref rel.err %.2e  (FP8 quantization err vs FP32: %.2e)  %s\n",
           c.name, c.fa.name, c.fb.name, rel, max_qerr / max_ref, ok ? "PASS" : "FAIL");

    cudaFree(lutA); cudaFree(lutB); cudaFree(dA); cudaFree(dB); cudaFree(dbias);
    cudaFree(dsa); cudaFree(dsb); cudaFree(dC);
    return ok;
}

static void dump_lut(const char* path, const Fp8Format& f) {
    uint16_t lut[256];
    fp8_build_lut(f, lut);
    FILE* fp = fopen(path, "wb");
    fwrite(lut, 2, 256, fp);
    fclose(fp);
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    CU(cuInit(0)); CK(cudaFree(0));  // runtime creates the primary context the driver API reuses
    printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    load_kernels(argc > 1 ? argv[1] : "fp8_gemm_kernel.cu", p.major, p.minor);
    dump_lut("lut_e4m3fn.bin", FMT_E4M3FN);
    dump_lut("lut_e5m2.bin", FMT_E5M2);

    const int M = 128, N = 96, K = 512;
    Case cases[] = {
        {"E4M3 block32 FP32-scale",        FMT_E4M3FN, FMT_E4M3FN, SCALE_BLOCK_K, false},
        {"E4M3 block32 UE8M0-scale",       FMT_E4M3FN, FMT_E4M3FN, SCALE_BLOCK_K, true},
        {"E5M2 block32 FP32-scale",        FMT_E5M2,   FMT_E5M2,   SCALE_BLOCK_K, false},
        {"A=E4M3 / B=E5M2 block32 UE8M0",  FMT_E4M3FN, FMT_E5M2,   SCALE_BLOCK_K, true},
        {"E4M3 per-row/col epilogue FP32", FMT_E4M3FN, FMT_E4M3FN, SCALE_ROW_COL, false},
        {"E4M3 per-row/col epilogue UE8M0",FMT_E4M3FN, FMT_E4M3FN, SCALE_ROW_COL, true},
    };
    bool all = true;
    for (auto& c : cases) all &= run_case(c, M, N, K);
    printf(all ? "ALL PASS\n" : "SOME FAILED\n");
    return all ? 0 : 1;
}
