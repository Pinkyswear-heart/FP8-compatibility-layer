"""Careful M=1..8 decode benchmark: cached-FP16 cuBLAS vs tfp8 GEMV, interleaved A/B.

The card is shared with other processes and single-batch timings swing 2-4x, so every candidate
is measured in the same interleaved loop and compared by median. This is the measurement the whole
W8A16 argument rests on, so it is worth doing properly.

    python -I tests/perf_decode.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfp8.linear import FP8Linear

SHAPES = {
    "qkv  12288x4096": (12288, 4096),
    "gate 11008x4096": (11008, 4096),
    "o     4096x4096": (4096, 4096),
    "down  4096x11008": (4096, 11008),
    "1b    5632x2048": (5632, 2048),
    "dit   4608x1152": (4608, 1152),
}


def interleaved(fns, warmup=15, iters=60, rounds=9):
    """Run every candidate once per round, round-robin, so clock/contention drift hits all
    candidates equally. Returns median ms per candidate."""
    for _ in range(warmup):
        for f in fns:
            f()
    torch.cuda.synchronize()
    samples = [[] for _ in fns]
    for _ in range(rounds):
        for i, f in enumerate(fns):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                f()
            torch.cuda.synchronize()
            samples[i].append((time.perf_counter() - t0) / iters * 1e3)
    return [statistics.median(s) for s in samples]


def run(M):
    print(f"\n== M={M} ==")
    print(f"{'layer':18s} {'FP16(cached)':>13s} {'tfp8 GEMV':>11s} {'speedup':>8s} "
          f"{'GB/s':>7s}  {'weights':>9s}")
    rows = []
    for label, (N, K) in SHAPES.items():
        torch.manual_seed(0)
        base = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
        q = FP8Linear.from_linear(base)
        q.out_dtype = torch.float16
        w16 = q.dequantized_weight().half()
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        x16 = x.half()
        t_fp16, t_tfp8 = interleaved([lambda: F.linear(x16, w16), lambda: q(x)])
        gb = q.qweight.numel() / (t_tfp8 * 1e-3) / 1e9
        sp = t_fp16 / t_tfp8
        rows.append((label, t_fp16, t_tfp8, sp, gb))
        print(f"{label:18s} {t_fp16:>11.3f}ms {t_tfp8:>9.3f}ms {sp:>7.2f}x {gb:>7.0f}  "
              f"{q.qweight.numel() / 2**20:>6.0f} MB")
        del base, q, w16, x, x16
    return rows


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm_"
          f"{''.join(map(str, torch.cuda.get_device_capability(0)))}  "
          f"torch {torch.__version__}")
    for M in (1, 2, 4, 8):
        run(M)
    print("\nspeedup > 1 表示 tfp8（FP8 权重常驻）比预先缓存的 FP16 权重 cuBLAS 更快；")
    print("这正是 W8A16 在 Decode 阶段唯一可能的收益来源。")
