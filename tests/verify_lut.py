"""Check the CUDA demo's LUTs against torch's float8 dtypes, bit for bit (NaN-aware).

Run from the repo root: python -I tests/verify_lut.py .
"""
import sys
import numpy as np
import torch

demo_dir = sys.argv[1] if len(sys.argv) > 1 else "."
ok = True
for name, dt in (("e4m3fn", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
    lut = np.fromfile(f"{demo_dir}/lut_{name}.bin", dtype=np.uint16)
    ref = torch.arange(256, dtype=torch.uint8).view(dt).to(torch.float16)
    ref_bits = ref.view(torch.int16).numpy().view(np.uint16)
    nan_ref = torch.isnan(ref).numpy()
    nan_lut = (lut & 0x7C00 == 0x7C00) & (lut & 0x03FF != 0)
    same = np.where(nan_ref, nan_lut, lut == ref_bits)
    bad = np.nonzero(~same)[0]
    print(f"{name}: {'OK' if not len(bad) else 'MISMATCH at ' + str(bad.tolist())}"
          f"  (NaN codes: {nan_ref.sum()}, +0=0x{lut[0]:04x}, -0=0x{lut[0x80]:04x})")
    ok &= not len(bad)
sys.exit(0 if ok else 1)
