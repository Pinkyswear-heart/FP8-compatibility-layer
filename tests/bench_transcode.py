"""FP8 <-> FP16 transcode throughput ("转译算力") benchmark for tfp8 on Turing.

Measures, on the RTX 2080 Ti (sm_75, no native FP8):
  * micro-kernels: the pure transcode ceiling (decode / encode / copy / fill bandwidth),
    NVRTC-compiled from the source below, launched with the same ctypes path as tfp8/_ext.py;
  * production paths: tfp8 kernels (fused GEMV, fused WMMA GEMM, chunked dequant+cuBLAS) and
    the torch reference paths (LUT gather decode, float8 casts, quantize_weight);
  * effective rates: Gelem/s, GB/s of traffic and TFLOPS, so the transcode cost can be
    separated from the GEMM cost.

The card is shared with desktop processes, so every number is an interleaved/median estimate:
  * gpu_time()        - CUDA events around a loop of `iters` calls, median of `rounds` rounds.
  * gpu_time_single() - CUDA events around one call (kernel duration, launch overhead excluded
                        if the GPU is idle).
  * wall/us           - CPU wall clock per call, the Python/ctypes launch floor.
All timings are reported in microseconds per call.

    python -I tests/bench_transcode.py                     # everything
    python -I tests/bench_transcode.py --only bw,decode    # selected sections
    python -I tests/bench_transcode.py --json out.json
"""
import argparse
import ctypes
import json
import os
import statistics
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tfp8 import _ext
from tfp8.formats import lut
from tfp8.linear import FP8Linear
from tfp8.quantize import dequantize_weight, quantize_weight

DEV = "cuda"
SEED = 0
DEFAULT_JSON = os.path.join(_REPO_ROOT, ".verify", "bench", "gpu", "transcode_gpu.json")


# --------------------------------------------------------------------------- timing helpers
def _stats(samples):
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": statistics.fmean(samples),
        "rounds_us": [round(s, 4) for s in samples],
    }


def gpu_time(fn, iters, rounds=7, warmup=2):
    """CUDA-event time over a loop of `iters` calls; median of `rounds` rounds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) / iters * 1e3)   # us per call
    return _stats(samples)


def gpu_time_single(fn, reps=30, warmup=3):
    """One refreshable measurement per call: excludes launch-queue batching effects."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1e3)
    return _stats(samples)


def gpu_time_auto(fn, target_ms=80.0, rounds=7, max_iters=500):
    """Pick iters so one round costs ~target_ms, then time it."""
    fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    est = max(a.elapsed_time(b), 0.01)
    iters = int(max(1, min(max_iters, target_ms / est)))
    return gpu_time(fn, iters=iters, rounds=rounds)


def wall_us(fn, iters=200, rounds=5):
    """CPU wall time per call with async launches (the Python/ctypes launch floor)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e6)
    return _stats(samples)


def rate(stat_us, elems=None, bytes_=None):
    us = stat_us["median_us"]
    out = {}
    if elems:
        out["Gelem_s"] = elems / (us * 1e-6) / 1e9
    if bytes_:
        out["GB_s"] = bytes_ / (us * 1e-6) / 1e9
    return out


# --------------------------------------------------------------------------- micro kernels
_MICRO_SRC = r"""
#include <cuda_fp16.h>
typedef unsigned char u8;
typedef unsigned long long u64;

// ---- bit-arithmetic decode, identical to tfp8/csrc/tfp8_kernels.cu -------------------------
__device__ __forceinline__ __half e4m3_sub(int m) {
    const unsigned short t[8] = {0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u};
    return __ushort_as_half(t[m & 7]);
}

__device__ __forceinline__ __half dec1(u8 b, int fmt, const __half* lut) {
    if (fmt == 0) {                                   // E4M3FN
        if ((b & 0x78u) != 0) {
            unsigned expmnt = (((unsigned)b & 0x7Fu) << 7) + 0x2000u;
            return __ushort_as_half((unsigned short)(expmnt | (((unsigned)b & 0x80u) << 8)));
        }
        __half s = e4m3_sub((int)(b & 7u));
        return (b & 0x80u) ? __hneg(s) : s;
    }
    if (fmt == 1) return __ushort_as_half((unsigned short)((unsigned)b << 8));   // E5M2
    return lut[b];
}

__device__ __forceinline__ bool e4m3_all_normal(unsigned w) {
    return (((w & 0x78787878u) + 0x08080808u) & 0x80808080u) == 0x80808080u;
}
__device__ __forceinline__ bool e4m3_has_nan(unsigned w) {
    unsigned t = w ^ 0x7F7F7F7Fu;
    return (((t - 0x01010101u) & ~t & 0x80808080u) != 0);
}

__device__ __forceinline__ void dec4(unsigned w, int fmt, const __half* lut, unsigned out[2]) {
    if (fmt == 0) {
        unsigned ev = w & 0x00FF00FFu;
        unsigned od = (w >> 8) & 0x00FF00FFu;
        unsigned lo = __vadd2((ev & 0x007F007Fu) << 7, 0x20002000u) | ((ev & 0x00800080u) << 8);
        unsigned hi = __vadd2((od & 0x007F007Fu) << 7, 0x20002000u) | ((od & 0x00800080u) << 8);
        out[0] = __byte_perm(lo, hi, 0x5410);
        out[1] = __byte_perm(lo, hi, 0x7632);
        return;
    }
    if (fmt == 1) {
        unsigned ev = (w & 0x00FF00FFu) << 8;
        unsigned od = ((w >> 8) & 0x00FF00FFu) << 8;
        out[0] = __byte_perm(ev, od, 0x5410);
        out[1] = __byte_perm(ev, od, 0x7632);
        return;
    }
    const u8* b = reinterpret_cast<const u8*>(&w);
    out[0] = (unsigned)__half_as_ushort(dec1(b[0], fmt, lut)) |
             ((unsigned)__half_as_ushort(dec1(b[1], fmt, lut)) << 16);
    out[1] = (unsigned)__half_as_ushort(dec1(b[2], fmt, lut)) |
             ((unsigned)__half_as_ushort(dec1(b[3], fmt, lut)) << 16);
}

__device__ __forceinline__ void dec4x4(uint4 raw, int fmt, const __half* lut, unsigned out[8]) {
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (fmt == 0 && !(e4m3_all_normal(w[i]) && !e4m3_has_nan(w[i]))) {
            const u8* b = reinterpret_cast<const u8*>(&w[i]);
#pragma unroll
            for (int t = 0; t < 2; ++t)
                out[2 * i + t] = (unsigned)__half_as_ushort(dec1(b[2 * t], fmt, lut)) |
                                 ((unsigned)__half_as_ushort(dec1(b[2 * t + 1], fmt, lut)) << 16);
        } else {
            dec4(w[i], fmt, lut, &out[2 * i]);
        }
    }
}

// ---- exact E4M3FN RNE + saturating encoder: fp16/fp32 in, 8-bit codes out ----------------
__device__ __forceinline__ u8 enc_e4m3(float x) {
    unsigned u = __float_as_uint(x);
    unsigned sign = (u >> 24) & 0x80u;
    unsigned ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return 0x7Fu;                 // NaN
    float f = __uint_as_float(ax);
    if (!(f < 448.0f)) return (u8)(sign | 0x7Eu);       // >= 448 or inf -> max finite
    if (f < 0.015625f) {                                // < 2^-6: subnormal, step 2^-9
        int r = (int)rintf(f * 512.0f);                 // RNE (rintf ties-to-even)
        return (u8)(sign | (unsigned)r);
    }
    unsigned m = (u >> 20) & 7u;
    unsigned rb = (u >> 19) & 1u;
    unsigned sticky = (u & 0x7FFFFu) != 0u;
    unsigned r = m + (rb & (sticky | (m & 1u)));
    int e = (int)((u >> 23) & 0xFFu) - 127;
    if (r == 8u) { r = 0u; e += 1; }
    if (e > 8) return (u8)(sign | 0x7Eu);
    return (u8)(sign | (unsigned)((e + 7) << 3) | r);
}

// ---- bandwidth reference kernels --------------------------------------------------------
extern "C" __global__ void m_copy_u8(const u8* __restrict__ in, u8* __restrict__ out, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride)
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(in + i);
}

extern "C" __global__ void m_copy_f16(const __half* __restrict__ in, __half* __restrict__ out,
                                      int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(in + i);
        *reinterpret_cast<uint4*>(out + i + 8) = *reinterpret_cast<const uint4*>(in + i + 8);
    }
}

extern "C" __global__ void m_fill_f16(__half* __restrict__ out, int n, float v) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    __half h = __float2half_rn(v);
    unsigned bits = (unsigned)__half_as_ushort(h) | ((unsigned)__half_as_ushort(h) << 16);
    uint4 u0 = make_uint4(bits, bits, bits, bits);
    for (; i < n; i += stride) {
        *reinterpret_cast<uint4*>(out + i) = u0;
        *reinterpret_cast<uint4*>(out + i + 8) = u0;
    }
}

// ---- transcode kernels ------------------------------------------------------------------
extern "C" __global__ void m_dec16_f16(const u8* __restrict__ w, const __half* __restrict__ lut,
                                       __half* __restrict__ out, int n, int fmt) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<const uint4*>(w + i);
        unsigned h2[8];
        dec4x4(raw, fmt, lut, h2);
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(&h2[0]);
        *reinterpret_cast<uint4*>(out + i + 8) = *reinterpret_cast<uint4*>(&h2[4]);
    }
}

extern "C" __global__ void m_dec16_f32(const u8* __restrict__ w, const __half* __restrict__ lut,
                                       float* __restrict__ out, int n, int fmt) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<const uint4*>(w + i);
        unsigned h2[8];
        dec4x4(raw, fmt, lut, h2);
        float fo[16];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2[t]));
            fo[2 * t] = f.x;
            fo[2 * t + 1] = f.y;
        }
        float* o = out + i;
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<uint4*>(&fo[0]);
        *reinterpret_cast<uint4*>(o + 4) = *reinterpret_cast<uint4*>(&fo[4]);
        *reinterpret_cast<uint4*>(o + 8) = *reinterpret_cast<uint4*>(&fo[8]);
        *reinterpret_cast<uint4*>(o + 12) = *reinterpret_cast<uint4*>(&fo[12]);
    }
}

// Decode + scale, one scale lookup per 16 elements (valid because G % 16 == 0 and i % 16 == 0;
// the production dequant kernel divides per element instead).
extern "C" __global__ void m_dec16_scale_f16(const u8* __restrict__ w,
                                             const __half* __restrict__ lut,
                                             const float* __restrict__ ws,
                                             __half* __restrict__ out, int n, int G) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        uint4 raw = *reinterpret_cast<const uint4*>(w + i);
        unsigned h2[8];
        dec4x4(raw, 0, lut, h2);
        const float s = ws[i / G];
        __half ho[16];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2[t]));
            ho[2 * t] = __float2half_rn(f.x * s);
            ho[2 * t + 1] = __float2half_rn(f.y * s);
        }
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(&ho[0]);
        *reinterpret_cast<uint4*>(out + i + 8) = *reinterpret_cast<uint4*>(&ho[8]);
    }
}

extern "C" __global__ void m_enc16_f16(const __half* __restrict__ x, u8* __restrict__ out,
                                       int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        const __half* hp = x + i;
        u8 o[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) o[t] = enc_e4m3(__half2float(hp[t]));
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
    }
}

extern "C" __global__ void m_enc16_f32(const float* __restrict__ x, u8* __restrict__ out,
                                       int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int stride = gridDim.x * blockDim.x * 16;
    for (; i < n; i += stride) {
        const float* xp = x + i;
        u8 o[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) o[t] = enc_e4m3(xp[t]);
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
    }
}
"""

_MICRO_NAMES = [
    "m_copy_u8", "m_copy_f16", "m_fill_f16",
    "m_dec16_f16", "m_dec16_f32", "m_dec16_scale_f16",
    "m_enc16_f16", "m_enc16_f32",
]


class Micro:
    """Compile _MICRO_SRC with NVRTC and launch it through cuLaunchKernel (same route as _ext)."""

    def __init__(self):
        from torch.cuda._utils import _cuda_load_module, _nvrtc_compile, _get_gpu_runtime_library
        self.dev = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(self.dev)
        binary, _log = _nvrtc_compile(_MICRO_SRC, _MICRO_NAMES[0], f"{major}{minor}", None,
                                      ["--std=c++17"])
        self.mod = _cuda_load_module(binary, _MICRO_NAMES)
        launch = _get_gpu_runtime_library().cuLaunchKernel
        launch.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 3 + [ctypes.c_uint] * 3 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        launch.restype = ctypes.c_int
        self._launch = launch
        self._stream = torch._C._cuda_getCurrentRawStream
        self._fns = {}
        self._slots = {}

    def call(self, name, grid, block, args):
        if name not in self._fns:
            self._fns[name] = self.mod[name].func
        slot = self._slots.get(name)
        if slot is None or len(slot[0]) != len(args):
            holders = [ctypes.c_void_p(0) if isinstance(a, torch.Tensor)
                       else (ctypes.c_double(0.0) if isinstance(a, float) else ctypes.c_int(0))
                       for a in args]
            arr = (ctypes.c_void_p * len(holders))(
                *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
            slot = self._slots[name] = (holders, arr)
        holders = slot[0]
        for h, a in zip(holders, args):
            h.value = a.data_ptr() if isinstance(a, torch.Tensor) else a
        rc = self._launch(self._fns[name], grid[0], grid[1], grid[2],
                          block[0], block[1], block[2], 0,
                          self._stream(self.dev), slot[1], None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel({name}) failed: {rc}")


def _grid(n, per_thread=16, block=256, cap=2048):
    return (min((n // per_thread + block - 1) // block, cap), 1, 1)


BLOCK = (256, 1, 1)


def _micro_time(micro, name, n, args, per_thread=16, cap=2048):
    grid = _grid(n, per_thread, cap=cap)
    return gpu_time_auto(lambda: micro.call(name, grid, BLOCK, args))


# --------------------------------------------------------------------------- helpers
def _clean(o):
    """JSON-safe copy: inf/nan -> strings."""
    if isinstance(o, float):
        if o != o:
            return "nan"
        if o == float("inf"):
            return "inf"
        if o == float("-inf"):
            return "-inf"
        return o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _mkq(N, K, fmt="e4m3fn", G=128):
    """Realistic quantized weight from N(0,1) BF16 weights."""
    w = torch.randn(N, K, device=DEV).bfloat16()
    q, s, G = quantize_weight(w, fmt, G)
    del w
    return q.contiguous(), s.contiguous(), G


def _mm_f32(a, b, flag):
    if flag:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a, b).float()


def _print_rate(label, st, extra="", elems=None, bytes_=None):
    r = rate(st, elems, bytes_)
    parts = [f"{st['median_us']:>10.2f} us"]
    if "Gelem_s" in r:
        parts.append(f"{r['Gelem_s']:>7.2f} Gelem/s")
    if "GB_s" in r:
        parts.append(f"{r['GB_s']:>7.1f} GB/s")
    print(f"  {label:34s} " + "  ".join(parts) + (f"   {extra}" if extra else ""))
    return r


# --------------------------------------------------------------------------- sections
def section_bw(micro):
    """Achievable memory bandwidth of the card (the hard ceiling for a bandwidth-bound transcode)."""
    print("\n== [bw] memory bandwidth reference (micro kernels vs torch) ==")
    res = {"micro": {}, "torch": {}}
    # micro copy: u8 and f16 at 256 MB of traffic
    n_u8 = 256 << 20
    src8 = torch.randint(0, 256, (n_u8,), dtype=torch.uint8, device=DEV)
    dst8 = torch.empty_like(src8)
    st = _micro_time(micro, "m_copy_u8", n_u8, [src8, dst8, n_u8])
    res["micro"]["copy_u8_256MB"] = {**st, **rate(st, bytes_=2 * n_u8)}
    _print_rate("copy u8->u8 (1B+1B)", st, bytes_=2 * n_u8, elems=n_u8)

    n_f16 = 64 << 20
    src16 = torch.randn(n_f16, dtype=torch.float16, device=DEV)
    dst16 = torch.empty_like(src16)
    st = _micro_time(micro, "m_copy_f16", n_f16, [src16, dst16, n_f16])
    res["micro"]["copy_f16_128MB"] = {**st, **rate(st, bytes_=4 * n_f16)}
    _print_rate("copy f16->f16 (2B+2B)", st, bytes_=4 * n_f16, elems=n_f16)

    st = _micro_time(micro, "m_fill_f16", n_f16, [dst16, n_f16, 0.0])
    res["micro"]["fill_f16_128MB"] = {**st, **rate(st, bytes_=2 * n_f16)}
    _print_rate("fill f16 (2B write only)", st, bytes_=2 * n_f16, elems=n_f16)

    # torch memcpy / fill / read for comparison
    st = gpu_time_auto(lambda: dst8.copy_(src8))
    res["torch"]["copy_u8_256MB"] = {**st, **rate(st, bytes_=2 * n_u8)}
    _print_rate("torch copy_ u8 (1B+1B)", st, bytes_=2 * n_u8, elems=n_u8)

    st = gpu_time_auto(lambda: dst16.copy_(src16))
    res["torch"]["copy_f16_128MB"] = {**st, **rate(st, bytes_=4 * n_f16)}
    _print_rate("torch copy_ f16 (2B+2B)", st, bytes_=4 * n_f16, elems=n_f16)

    st = gpu_time_auto(lambda: dst16.fill_(0.0))
    res["torch"]["fill_f16"] = {**st, **rate(st, bytes_=2 * n_f16)}
    _print_rate("torch fill_ f16 (write only)", st, bytes_=2 * n_f16, elems=n_f16)

    st = gpu_time_auto(lambda: src16.sum())
    res["torch"]["sum_f16"] = {**st, **rate(st, bytes_=2 * n_f16)}
    _print_rate("torch sum f16 (read only)", st, bytes_=2 * n_f16, elems=n_f16)

    # cuBLAS / CUDA-core compute ceilings (for the GEMM comparison later)
    a = torch.randn(4096, 4096, dtype=torch.float16, device=DEV)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=DEV)
    flops = 2 * 4096 ** 3
    st = gpu_time_auto(lambda: torch.mm(a, b))
    res["cublas_f16_4096"] = {**st, "TFLOPS": flops / (st["median_us"] * 1e-6) / 1e12}
    print(f"  {'cuBLAS fp16 4096^3':34s} {st['median_us'] / 1e3:>10.3f} ms  "
          f"{res['cublas_f16_4096']['TFLOPS']:>7.1f} TFLOPS")
    del a, b

    af = torch.randn(4096, 4096, device=DEV)
    bf = torch.randn(4096, 4096, device=DEV)
    st = gpu_time_auto(lambda: torch.mm(af, bf))
    res["cublas_f32_4096"] = {**st, "TFLOPS": flops / (st["median_us"] * 1e-6) / 1e12}
    print(f"  {'cuBLAS fp32 4096^3':34s} {st['median_us'] / 1e3:>10.3f} ms  "
          f"{res['cublas_f32_4096']['TFLOPS']:>7.1f} TFLOPS")
    del af, bf, src8, dst8, src16, dst16
    return res


def section_decode(micro):
    """FP8 -> FP16/FP32 decode: micro ceiling, production kernel, torch reference."""
    print("\n== [decode] FP8 -> FP16/FP32 transcode ==")
    res = {"micro": {}, "production": {}, "torch": {}, "checks": {}}

    N, K = 11008, 4096                     # a real Llama gate_proj shape: 45.1 M weights
    q, s, G = _mkq(N, K, "e4m3fn", 128)
    n = q.numel()
    q5, s5, _ = _mkq(N, K, "e5m2", 128)
    lut16 = lut("e4m3fn", DEV)
    out16 = torch.empty(n, dtype=torch.float16, device=DEV)
    out32 = torch.empty(n, dtype=torch.float32, device=DEV)
    qf = q.view(-1)
    q5f = q5.view(-1)
    wsflat = s.reshape(-1).contiguous()

    # -- correctness: bit-exact against the LUT reference
    m = 1 << 20
    qs = qf[:m]
    os16 = torch.empty(m, dtype=torch.float16, device=DEV)
    micro.call("m_dec16_f16", _grid(m), BLOCK, [qs, lut16, os16, m, 0])
    torch.cuda.synchronize()
    ref = lut16[qs.long()]
    res["checks"]["micro_dec_e4m3_bit_exact"] = bool(torch.equal(os16, ref))

    oss = torch.empty(m, dtype=torch.float16, device=DEV)
    micro.call("m_dec16_scale_f16", _grid(m), BLOCK, [qs, lut16, wsflat, oss, m, G])
    torch.cuda.synchronize()
    refs = (lut16[qs.long()].float() * wsflat[torch.arange(m, device=DEV) // G]).half()
    res["checks"]["micro_dec_scale_bit_exact"] = bool(torch.equal(oss, refs))

    oss5 = torch.empty(m, dtype=torch.float16, device=DEV)
    micro.call("m_dec16_f16", _grid(m), BLOCK, [q5f[:m], lut16, oss5, m, 1])
    torch.cuda.synchronize()
    res["checks"]["micro_dec_e5m2_bit_exact"] = bool(torch.equal(oss5, lut("e5m2", DEV)[q5f[:m].long()]))
    print(f"  bit-exact decode checks: e4m3={res['checks']['micro_dec_e4m3_bit_exact']} "
          f"e4m3+scale={res['checks']['micro_dec_scale_bit_exact']} "
          f"e5m2={res['checks']['micro_dec_e5m2_bit_exact']}")
    del qs, os16, oss, oss5, ref, refs

    # -- micro ceiling at several sizes (scaling: when does it saturate the memory system?)
    for n_i in (1 << 20, 4 << 20, 16 << 20, n):
        st = _micro_time(micro, "m_dec16_f16", n_i, [qf[:n_i], lut16, out16[:n_i], n_i, 0])
        res["micro"][f"dec_e4m3_f16_n{n_i}"] = {**st,
                                                **rate(st, bytes_=3 * n_i, elems=n_i)}
        tag = "45.1M" if n_i == n else f"{n_i / 2 ** 20:.0f}M"
        _print_rate(f"micro dec e4m3->f16  n={tag}", st, bytes_=3 * n_i, elems=n_i)

    st = _micro_time(micro, "m_dec16_f16", n, [q5f, lut16, out16, n, 1])
    res["micro"]["dec_e5m2_f16_45M"] = {**st, **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("micro dec e5m2->f16  n=45.1M", st, bytes_=3 * n, elems=n)

    st = _micro_time(micro, "m_dec16_f16", n, [qf, lut16, out16, n, 2])
    res["micro"]["dec_lut_f16_45M"] = {**st, **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("micro dec LUT->f16  n=45.1M", st, bytes_=3 * n, elems=n)

    st = _micro_time(micro, "m_dec16_f32", n, [qf, lut16, out32, n, 0])
    res["micro"]["dec_e4m3_f32_45M"] = {**st, **rate(st, bytes_=5 * n, elems=n)}
    _print_rate("micro dec e4m3->f32 n=45.1M", st, bytes_=5 * n, elems=n)

    st = _micro_time(micro, "m_dec16_scale_f16", n, [qf, lut16, wsflat, out16, n, G])
    res["micro"]["dec_e4m3_scale128_f16_45M"] = {**st,
                                                 **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("micro dec e4m3+scale128  n=45.1M", st, bytes_=3 * n, elems=n)

    st = _micro_time(micro, "m_dec16_scale_f16", n, [qf, lut16, s.reshape(-1), out16, n, K])
    res["micro"]["dec_e4m3_perchan_f16_45M"] = {**st,
                                                **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("micro dec e4m3+perchan   n=45.1M", st, bytes_=3 * n, elems=n)

    # -- production dequant kernel (tfp8_dequant_f16) on real shapes / formats / scale layouts
    configs = [
        (11008, 4096, "e4m3fn", 128, "qkv 11008x4096 e4m3 g128"),
        (11008, 4096, "e4m3fn", None, "qkv 11008x4096 e4m3 perchan"),
        (11008, 4096, "e5m2", 128, "qkv 11008x4096 e5m2 g128"),
        (11008, 4096, "e5m2", None, "qkv 11008x4096 e5m2 perchan"),
        (4096, 4096, "e4m3fn", 128, "o   4096x4096 e4m3 g128"),
        (4096, 4096, "e4m3fn", None, "o   4096x4096 e4m3 perchan"),
        (5632, 2048, "e4m3fn", 128, "1b  5632x2048 e4m3 g128"),
        (4608, 1152, "e4m3fn", 128, "dit 4608x1152 e4m3 g128"),
        (256, 512, "e4m3fn", 128, "tiny 256x512 e4m3 g128"),
    ]
    for (NN, KK, fmt, Gg, label) in configs:
        qq, ss, GG = _mkq(NN, KK, fmt, Gg)
        lutx = lut(fmt, DEV)
        oo = torch.empty(NN, KK, dtype=torch.float16, device=DEV)
        st = gpu_time_auto(lambda: _ext.dequant_into(oo, qq, lutx, ss, GG, fmt))
        nn = NN * KK
        rec = {**st, **rate(st, bytes_=3 * nn, elems=nn),
               "N": NN, "K": KK, "fmt": fmt, "group": GG, "MB": nn / 2 ** 20}
        # bit-exact against the torch reference on the first 8 rows
        if NN >= 8:
            rr = dequantize_weight(qq[:8], ss[:8], fmt, GG).half()
            rec["bit_exact_8rows"] = bool(torch.equal(oo[:8], rr))
            del rr
        res["production"][label] = rec
        _print_rate(f"prod dequant {label:29s}", st, bytes_=3 * nn, elems=nn)
        del qq, ss, oo, lutx

    # -- tfp8 reference decode (LUT gather + fp32 multiply) on the same 45.1 M weights
    st = gpu_time_auto(lambda: dequantize_weight(q, s, "e4m3fn", G), rounds=5)
    res["torch"]["dequantize_weight_45M"] = {**st,
                                             **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("tfp8 torch ref dequant 45.1M", st, bytes_=3 * n, elems=n)

    st = gpu_time_auto(lambda: lut16[q.long()], rounds=5)
    res["torch"]["lut_gather_45M"] = {**st, **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("torch lut[q.long()]    45.1M", st, bytes_=3 * n, elems=n)

    # -- torch's own float8 -> fp16 cast, if sm_75 supports it at all
    try:
        f8 = q.view(torch.float8_e4m3fn)
        st = gpu_time_auto(lambda: f8.to(torch.float16), rounds=5)
        res["torch"]["torch_cast_f8_to_f16_45M"] = {**st,
                                                    **rate(st, bytes_=3 * n, elems=n)}
        _print_rate("torch float8->f16 cast 45.1M", st, bytes_=3 * n, elems=n)
    except Exception as e:  # noqa: BLE001
        res["torch"]["torch_cast_f8_to_f16_45M"] = {"error": repr(e)}
        print(f"  torch float8->f16 cast: UNSUPPORTED ({type(e).__name__}: {e})")
    del q, s, q5, s5, out16, out32, wsflat
    return res


def section_encode(micro):
    """FP16/FP32 -> FP8 encode: micro ceiling, torch cast, tfp8 quantize_weight."""
    print("\n== [encode] FP16/FP32 -> FP8 transcode ==")
    res = {"micro": {}, "torch": {}, "quantize_gpu": {}, "checks": {}}
    n = 11008 * 4096
    x16 = torch.randn(n, dtype=torch.float16, device=DEV)
    x32 = torch.randn(n, device=DEV)
    out8 = torch.empty(n, dtype=torch.uint8, device=DEV)

    # correctness vs torch's own cast (finite random values, plus specials checked separately)
    m = 1 << 20
    o8s = torch.empty(m, dtype=torch.uint8, device=DEV)
    micro.call("m_enc16_f16", _grid(m), BLOCK, [x16[:m], o8s, m])
    torch.cuda.synchronize()
    ref8 = x16[:m].to(torch.float8_e4m3fn).view(torch.uint8)
    res["checks"]["micro_enc_f16_bit_exact"] = bool(torch.equal(o8s, ref8))
    res["checks"]["micro_enc_mismatch"] = int((o8s != ref8).sum().item())
    o8s32 = torch.empty(m, dtype=torch.uint8, device=DEV)
    micro.call("m_enc16_f32", _grid(m), BLOCK, [x32[:m], o8s32, m])
    torch.cuda.synchronize()
    ref8b = x32[:m].to(torch.float8_e4m3fn).view(torch.uint8)
    res["checks"]["micro_enc_f32_bit_exact"] = bool(torch.equal(o8s32, ref8b))
    specials = torch.tensor([0.0, -0.0, 448.0, -448.0, 464.0, 1e-9, -1e-9,
                             float("inf"), float("-inf"), float("nan")],
                            dtype=torch.float16, device=DEV)
    specials = torch.cat([specials, torch.zeros(16 - len(specials), dtype=torch.float16, device=DEV)])
    osp = torch.empty(len(specials), dtype=torch.uint8, device=DEV)
    micro.call("m_enc16_f16", _grid(16), BLOCK, [specials, osp, 16])
    torch.cuda.synchronize()
    res["checks"]["specials_micro"] = [int(v) for v in osp.cpu()]
    res["checks"]["specials_torch"] = [int(v) for v in specials.to(torch.float8_e4m3fn).view(torch.uint8).cpu()]
    print(f"  bit-exact encode: f16={res['checks']['micro_enc_f16_bit_exact']} "
          f"(mismatch {res['checks']['micro_enc_mismatch']}/{m}) "
          f"f32={res['checks']['micro_enc_f32_bit_exact']}")
    del o8s, o8s32, ref8, ref8b, specials, osp

    st = _micro_time(micro, "m_enc16_f16", n, [x16, out8, n])
    res["micro"]["enc_f16_e4m3_45M"] = {**st, **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("micro enc f16->e4m3 45.1M", st, bytes_=3 * n, elems=n)

    st = _micro_time(micro, "m_enc16_f32", n, [x32, out8, n])
    res["micro"]["enc_f32_e4m3_45M"] = {**st, **rate(st, bytes_=5 * n, elems=n)}
    _print_rate("micro enc f32->e4m3 45.1M", st, bytes_=5 * n, elems=n)

    st = gpu_time_auto(lambda: x16.to(torch.float8_e4m3fn))
    res["torch"]["cast_f16_e4m3_45M"] = {**st, **rate(st, bytes_=3 * n, elems=n)}
    _print_rate("torch cast f16->float8 45.1M", st, bytes_=3 * n, elems=n)

    st = gpu_time_auto(lambda: x32.to(torch.float8_e4m3fn))
    res["torch"]["cast_f32_e4m3_45M"] = {**st, **rate(st, bytes_=5 * n, elems=n)}
    _print_rate("torch cast f32->float8 45.1M", st, bytes_=5 * n, elems=n)

    st = gpu_time_auto(lambda: x32.bfloat16().to(torch.float8_e4m3fn))
    res["torch"]["cast_f32_bf16_f8_e4m3_45M"] = {**st,
                                                 **rate(st, elems=n)}
    _print_rate("torch f32->bf16->float8 45.1M", st, elems=n)
    del x16, x32, out8

    # production path: tfp8 quantize_weight (whole checkpoint conversion, GPU)
    for (NN, KK, Gg, label) in [(4096, 4096, 128, "4096x4096 g128"),
                                (4096, 4096, None, "4096x4096 perchan"),
                                (11008, 4096, 128, "11008x4096 g128")]:
        w = torch.randn(NN, KK, device=DEV).bfloat16()
        st = gpu_time_auto(lambda: quantize_weight(w, "e4m3fn", Gg))
        nn = NN * KK
        res["quantize_gpu"][label] = {**st,
                                      **rate(st, bytes_=2 * nn, elems=nn)}
        _print_rate(f"gpu quantize_weight {label:22s}", st, bytes_=2 * nn, elems=nn)
        del w
    return res


def section_gemv(micro):
    """Fused GEMV (M <= 8): FP8 weight streamed once and decoded in-register."""
    print("\n== [gemv] fused GEMV, FP8 weights decoded in-register (M<=8) ==")
    res = {"rows": [], "checks": {}}
    shapes = [(12288, 4096), (11008, 4096), (4096, 4096), (4096, 11008), (5632, 2048), (4608, 1152)]
    fmt = "e4m3fn"
    for (N, K) in shapes:
        q, s, G = _mkq(N, K, fmt, None)
        lut16 = lut(fmt, DEV)
        qmb = q.numel() / 2 ** 20
        w16 = dequantize_weight(q, s, fmt, G).half()
        print(f"  layer {N}x{K}  fp8 {qmb:.0f} MB")
        for M in (1, 2, 4, 8):
            x = torch.randn(M, K, device=DEV).bfloat16()
            x16 = x.half()
            st = gpu_time_auto(lambda: _ext.w8a16(x, q, lut16, s, None, torch.float16, G, fmt))
            eff = 2 * M * N * K
            rec = {**st,
                   **rate(st, bytes_=q.numel(), elems=q.numel()),
                   "TFLOPS": eff / (st["median_us"] * 1e-6) / 1e12,
                   "N": N, "K": K, "M": M, "weight_MB": qmb}
            res["rows"].append(rec)
            _print_rate(f"  M={M} tfp8 fused GEMV", st, bytes_=q.numel(), elems=q.numel(),
                        extra=f"{rec['TFLOPS']:.2f} TFLOPS")
            if M == 1:
                st16 = gpu_time_auto(lambda: torch.mm(x16, w16.t()))
                res["rows"][-1]["fp16_cublas_us"] = st16["median_us"]
                print(f"  M=1 cached-FP16 cuBLAS           {st16['median_us']:>10.2f} us   "
                      f"(tfp8 / fp16 = {st['median_us'] / st16['median_us']:.2f}x)")
                # correctness once per shape
                y = _ext.w8a16(x, q, lut16, s, None, torch.float16, G, fmt)
                yref = torch.mm(x.half(), w16.t())
                res.setdefault("checks", {})[f"{N}x{K}"] = float(
                    (y.float() - yref.float()).abs().max().item())
                del y, yref
            del x, x16
        del q, s, w16, lut16
    return res


def section_gemm(micro):
    """Fused WMMA GEMM and the chunked dequant+cuBLAS backend vs cuBLAS on a cached FP16 weight."""
    print("\n== [gemm] Fused WMMA GEMM / chunked dequant+cuBLAS / cuBLAS fp16 (M>8) ==")
    res = {"rows": [], "chunked_breakdown": {}, "checks": {}}
    N = K = 4096
    fmt = "e4m3fn"
    q, s, G = _mkq(N, K, fmt, None)
    lut16 = lut(fmt, DEV)
    qmb = q.numel() / 2 ** 20
    w16 = dequantize_weight(q, s, fmt, G).half()
    lin = torch.nn.Linear(K, N, bias=False, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(dequantize_weight(q, s, fmt, G).to(torch.bfloat16))
    ql = FP8Linear.from_linear(lin)
    ql.out_dtype = torch.float16
    ql.large_m = 9                     # force the chunked backend for every M>8
    ql.large_m_backend = "chunked"
    # out_dtype kwarg support probe (torch.mm)
    try:
        torch.mm(torch.zeros(8, K, dtype=torch.float16, device=DEV),
                 torch.zeros(K, 8, dtype=torch.float16, device=DEV), out_dtype=torch.float32)
        mm_f32 = True
    except TypeError:
        mm_f32 = False
    print(f"  layer 4096x4096  fp8 {qmb:.0f} MB   torch.mm(out_dtype=float32)={mm_f32}")
    for M in (32, 64, 128, 256, 512, 1024, 2048):
        x = torch.randn(M, K, device=DEV).bfloat16()
        x16 = x.half()
        st_fused = gpu_time_auto(lambda: _ext.w8a16(x, q, lut16, s, None, torch.float16, G, fmt))
        st_chunk = gpu_time_auto(lambda: ql(x))
        st_cblas = gpu_time_auto(lambda: torch.mm(x16, w16.t()))
        flops = 2 * M * N * K
        rec = {"M": M,
               "fused_us": st_fused["median_us"], "chunked_us": st_chunk["median_us"],
               "cublas_fp16_us": st_cblas["median_us"],
               "fused_TFLOPS": flops / (st_fused["median_us"] * 1e-6) / 1e12,
               "chunked_TFLOPS": flops / (st_chunk["median_us"] * 1e-6) / 1e12,
               "cublas_TFLOPS": flops / (st_cblas["median_us"] * 1e-6) / 1e12,
               "fp8_GB_s": q.numel() / (st_fused["median_us"] * 1e-6) / 1e9,
               "fused_decoded_G": ((M + 63) // 64) * q.numel() / 1e9}
        res["rows"].append(rec)
        print(f"  M={M:<5d} fused {st_fused['median_us'] / 1e3:8.3f} ms ({rec['fused_TFLOPS']:5.1f} TF) | "
              f"chunked {st_chunk['median_us'] / 1e3:8.3f} ms ({rec['chunked_TFLOPS']:5.1f} TF) | "
              f"cuBLAS {st_cblas['median_us'] / 1e3:8.3f} ms ({rec['cublas_TFLOPS']:5.1f} TF)")
        if M in (32, 2048):
            y = _ext.w8a16(x, q, lut16, s, None, torch.float16, G, fmt)
            yref = torch.mm(x16, w16.t())
            res["checks"][f"fused_M{M}"] = float((y.float() - yref.float()).abs().max().item())
            del y, yref
        del x, x16

    # breakdown of the chunked backend at M=2048 (the M at which it beats `fused`)
    print("\n  chunked backend breakdown (M=2048):")
    M = 2048
    x = torch.randn(M, K, device=DEV).bfloat16()
    st_prep = gpu_time_auto(lambda: _ext.prepare_activations(x))
    x16, xs = _ext.prepare_activations(x)
    wbuf = torch.empty(N, K, dtype=torch.float16, device=DEV)
    st_deq = gpu_time_auto(lambda: _ext.dequant_into(wbuf, q, lut16, s, G))
    st_mm = gpu_time_auto(lambda: _mm_f32(x16, wbuf.t(), mm_f32))
    # Epilogue = the production fused kernel (row scale + bias + cast, one pass, writes fp16
    # directly). Since 11-08 there is no separate cast pass: cast_us is reported as 0.
    acc = _mm_f32(x16, wbuf.t(), mm_f32)
    bias0 = torch.zeros(N, dtype=torch.float32, device=DEV)
    y16 = torch.empty(M, N, dtype=torch.float16, device=DEV)
    st_epi = gpu_time_auto(lambda: _ext.epilogue_into(y16, acc, xs, bias0))
    st_cast = {"median_us": 0.0}
    out = acc
    res["chunked_breakdown"] = {
        "M": M,
        "prep_us": st_prep["median_us"], "dequant_us": st_deq["median_us"],
        "mm_us": st_mm["median_us"], "epilogue_us": st_epi["median_us"],
        "cast_us": st_cast["median_us"],
        "sum_us": st_prep["median_us"] + st_deq["median_us"] + st_mm["median_us"]
        + st_epi["median_us"] + st_cast["median_us"],
        "total_us": [r["chunked_us"] for r in res["rows"] if r["M"] == M][0],
    }
    b = res["chunked_breakdown"]
    print(f"    prep {b['prep_us']:.1f} us | dequant {b['dequant_us']:.1f} us "
          f"({b['dequant_us'] / b['total_us'] * 100:.0f}% of total) | mm {b['mm_us']:.1f} us | "
          f"epilogue {b['epilogue_us']:.1f} us | cast {b['cast_us']:.1f} us | total {b['total_us']:.1f} us")
    del x, x16, xs, wbuf, out, acc, y16, bias0, q, s, w16, lin, ql, lut16
    return res


def section_floor(micro):
    """Launch / Python overhead: the floor under every small-layer transcode call.

    On Windows/WDDM a single-call CUDA-event timing carries ~30 us of event/sync overhead plus
    the submission latency of an idle queue, so it is reported separately and NOT treated as
    kernel time. The loop-event and wall numbers are the ones a real workload pays per call.
    """
    print("\n== [floor] launch overhead ==")
    res = {"event_overhead_us": gpu_time_single(lambda: None)["median_us"]}
    print(f"  event+sync overhead (empty call): {res['event_overhead_us']:.2f} us")

    tiny8 = torch.zeros(16, dtype=torch.uint8, device=DEV)
    tiny_out = torch.zeros(16, dtype=torch.uint8, device=DEV)
    call = lambda: micro.call("m_copy_u8", (1, 1, 1), (1, 1, 1), [tiny8, tiny_out, 16])  # noqa: E731
    res["micro_16B"] = {
        "event_loop_us": gpu_time(call, iters=500)["median_us"],
        "wall_us": wall_us(call)["median_us"],
        "single_call_us": gpu_time_single(call)["median_us"],
    }
    print(f"  micro 16-byte copy:   event-loop {res['micro_16B']['event_loop_us']:.2f} us | "
          f"wall/call {res['micro_16B']['wall_us']:.2f} us | "
          f"single-call {res['micro_16B']['single_call_us']:.2f} us")

    N = K = 128
    q, s, G = _mkq(N, K, "e4m3fn", None)
    lut16 = lut("e4m3fn", DEV)
    x = torch.randn(1, K, device=DEV).bfloat16()
    call = lambda: _ext.w8a16(x, q, lut16, s, None, torch.float16, G, "e4m3fn")  # noqa: E731
    res["tfp8_gemv_128x128"] = {
        "event_loop_us": gpu_time(call, iters=500)["median_us"],
        "wall_us": wall_us(call)["median_us"],
        "single_call_us": gpu_time_single(call)["median_us"],
    }
    print(f"  tfp8 GEMV 128x128:    event-loop {res['tfp8_gemv_128x128']['event_loop_us']:.2f} us | "
          f"wall/call {res['tfp8_gemv_128x128']['wall_us']:.2f} us | "
          f"single-call {res['tfp8_gemv_128x128']['single_call_us']:.2f} us")
    print("  (single-call includes WDDM submission latency; use wall/call for a stream of calls)")
    del q, s, x, lut16, tiny8, tiny_out
    return res


# --------------------------------------------------------------------------- main
SECTIONS = {
    "bw": ("memory bandwidth reference", True),
    "decode": ("FP8 -> FP16/FP32 transcode", True),
    "encode": ("FP16/FP32 -> FP8 transcode", True),
    "gemv": ("fused GEMV", False),
    "gemm": ("fused/chunked GEMM vs cuBLAS", False),
    "floor": ("launch overhead", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help=f"comma-separated sections; available: {','.join(SECTIONS)}")
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else set(SECTIONS)
    needs_micro = any(SECTIONS[k][1] for k in only if k in SECTIONS)

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.init()
    if not _ext.available():
        sys.exit("tfp8 CUDA kernels unavailable (NVRTC failed)")

    p = torch.cuda.get_device_properties(0)
    env = {
        "gpu": p.name, "sm": f"{p.major}{p.minor}", "sm_count": p.multi_processor_count,
        "vram_MB": round(p.total_memory / 2 ** 20), "l2_MB": round(p.L2_cache_size / 2 ** 20),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": sys.version.split()[0], "driver": _driver_version(),
        "cpu_count": os.cpu_count(), "python_threads": torch.get_num_threads(),
    }
    env["gpu_snapshot_start"] = _gpu_snapshot()
    print(f"GPU {env['gpu']} sm_{env['sm']} {env['sm_count']} SM  {env['vram_MB']} MB  "
          f"L2 {env['l2_MB']} MB  torch {env['torch']} / CUDA {env['cuda']}")
    print("shared card: every number is a median over interleaved/looped CUDA-event timings")

    micro = Micro() if needs_micro else None
    out = {"env": env, "sections": {}}
    for key in ("bw", "decode", "encode", "gemv", "gemm", "floor"):
        if key not in only:
            continue
        try:
            fn = {"bw": lambda: section_bw(micro), "decode": lambda: section_decode(micro),
                  "encode": lambda: section_encode(micro), "gemv": lambda: section_gemv(micro),
                  "gemm": lambda: section_gemm(micro), "floor": lambda: section_floor(micro)}[key]
            out["sections"][key] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["sections"][key] = {"error": repr(e)}
    env["gpu_snapshot_end"] = _gpu_snapshot()
    out = _clean(out)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {args.json}")


def _gpu_snapshot():
    """util / SM clock / power / memory-use snapshot, for interpreting a shared card."""
    try:
        import subprocess
        q = "utilization.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return [v.strip() for v in r.stdout.strip().split("\n")[0].split(",")]
    except Exception:  # noqa: BLE001
        return []


def _driver_version():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    main()
