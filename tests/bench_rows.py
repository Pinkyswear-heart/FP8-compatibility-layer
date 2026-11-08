"""Compare GEMV_RADIUS variants by pure GPU kernel time (launch overhead excluded).

    python -I tests/bench_rows.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from tfp8 import _ext
from tfp8.linear import FP8Linear

SHAPES = [(4608, 1152), (5632, 2048), (4096, 4096), (12288, 4096)]


def gpu_time(fn, n=400, rounds=7, warm=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            fn()
        b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b) / n * 1e3)   # us per call
    return statistics.median(s)


def main():
    print(f"GEMV_ROWS={_ext.GEMV_ROWS}   (pure GPU kernel time, us)")
    for N, K in SHAPES:
        torch.manual_seed(0)
        base = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
        q = FP8Linear.from_linear(base)
        q.out_dtype = torch.float16
        x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
        t = gpu_time(lambda: _ext.w8a16(x, q.qweight, q._lut, q.weight_scale, None,
                                        torch.float16, K, "e4m3fn"))
        gb = q.qweight.numel() / (t * 1e-6) / 1e9
        print(f"  N={N:6d} K={K:5d}  {t:6.1f} us  {gb:5.0f} GB/s")
        del base, q, x


if __name__ == "__main__":
    main()
