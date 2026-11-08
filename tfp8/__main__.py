"""python -m tfp8 : self-check and environment report (design doc section 9.3)."""
import platform
import sys

import torch

from tfp8 import _ext
from tfp8.formats import E4M3FN, E5M2, decode, encode


def main():
    print("tfp8 v0.1 -- Turing FP8 compatible runtime (W8A16 demo)\n")
    print(f"python     {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    print(f"torch      {torch.__version__} / CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu        {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 2**30:.1f} GB")
        print(f"nvrtc      {'ok' if _ext.available() else 'UNAVAILABLE (kernels will use the torch path)'}")
    else:
        print("gpu        none -- reference (torch) paths only")

    print("\nformats")
    codes = torch.arange(256, dtype=torch.uint8)
    for f in (E4M3FN, E5M2):
        ref = codes.view(f.torch_dtype).float()
        ok = (torch.isnan(ref) & torch.isnan(decode(codes, f))).sum() + (decode(codes, f) == ref).sum() == 256
        print(f"  {f.name:7s} e{f.ebits}m{f.mbits} bias {f.bias:2d}  max {f.max_finite:>8.0f}  "
              f"all 256 codes decode exactly: {bool(ok)}")
    x = torch.randn(4096) * 100
    print(f"  encode RNE matches torch float8 cast: {bool(torch.equal(encode(x, E4M3FN), x.to(torch.float8_e4m3fn).view(torch.uint8)))}")

    print("\nkernels (fused decode -> FP16 MMA -> FP32 accumulate)")
    print("  ABI: qweight uint8[N,K], weight_scale f32[N,K/G], M-split GEMV(M<=8)/GEMM/large-M chunked")
    print("  fused GEMV, fused 64x64 WMMA GEMM, chunked dequant+cuBLAS with fixed workspace")
    print("\nrun the full numeric gate set:  python -I tests/test_tfp8.py")
    print("benchmark a layer:              python -I tests/benchmark.py")
    print("LLM demo:                       python -m tfp8.examples.demo_llm")
    print("image demo:                     python -m tfp8.examples.demo_image")


if __name__ == "__main__":
    main()
