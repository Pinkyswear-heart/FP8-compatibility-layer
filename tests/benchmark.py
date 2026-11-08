"""Layer benchmark: BF16 native, FP16, full-dequant-then-GEMM, and the TFP8 fused paths.

    python -I tests/benchmark.py --N 4096 --K 4096
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfp8 import _ext
from tfp8.linear import FP8Linear


def _time(fn, warmup=10, iters=50, reps=5):
    """Median of `reps` timed batches. A single batch on this card swings 2-4x run to run."""
    import statistics
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    batches = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        batches.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(batches)


def bench_layer(N=4096, K=4096, Ms=(1, 8, 32, 128, 2048), device="cuda"):
    torch.manual_seed(0)
    lin = nn.Linear(K, N, device=device, dtype=torch.bfloat16)
    q = FP8Linear.from_linear(lin)
    q.out_dtype = torch.float16
    ref_w = q.dequantized_weight().half()          # full FP16 copy: what the fused path avoids
    rows = []
    for M in Ms:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        x16 = x.half()
        r = {"M": M}
        r["bf16_native_ms"] = _time(lambda: F.linear(x, lin.weight, lin.bias))
        r["fp16_ms"] = _time(lambda: F.linear(x16, ref_w, None))
        r["full_dequant_ms"] = _time(lambda: F.linear(x16, q.dequantized_weight().half(), None))
        q.large_m_backend = "fused"
        r["tfp8_fused_ms"] = _time(lambda: q(x))
        q.large_m_backend = "chunked"
        r["tfp8_chunked_ms"] = _time(lambda: q(x))
        rows.append(r)
        del x
    return {"N": N, "K": K, "fp16_weight_MB": ref_w.numel() * 2 / 2 ** 20,
            "fp8_weight_MB": q.qweight.numel() / 2 ** 20, "rows": rows}


def print_table(res):
    print(f"\nLinear {res['N']}x{res['K']} (weight: FP16 {res['fp16_weight_MB']:.0f} MB vs "
          f"FP8 {res['fp8_weight_MB']:.0f} MB)\n")
    cols = ["bf16_native_ms", "fp16_ms", "full_dequant_ms", "tfp8_fused_ms", "tfp8_chunked_ms"]
    print(f"{'M':>6s} " + " ".join(f"{c.replace('_ms',''):>14s}" for c in cols) + "   best tfp8 vs bf16")
    for r in res["rows"]:
        best = min(r["tfp8_fused_ms"], r["tfp8_chunked_ms"])
        gain = r["bf16_native_ms"] / best
        print(f"{r['M']:>6d} " + " ".join(f"{r[c]:>14.3f}" for c in cols) + f"   {gain:>6.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = bench_layer(a.N, a.K)
    print_table(res)
    print("\nms per call, RTX 2080 Ti (sm_75). 'bf16_native' is PyTorch BF16 (no tensor cores on "
          "Turing); tfp8 runs on FP16 tensor cores with FP32 accumulate.")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
