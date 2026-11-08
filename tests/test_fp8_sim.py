"""Sanity checks for fp8_sim: run from the repo root with `python -I tests/test_fp8_sim.py`."""
import os
import sys

# tests/ is one level below the repo root; fp8_sim.py stays in train/ as core training code.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "train"))

import numpy as np
import torch
import fp8_sim as F

torch.manual_seed(0)
dev = "cuda"
ok = True


def check(name, cond):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    ok &= bool(cond)


# 1. torch cast + clamp == the CUDA demo's encoder (nearest, ties to even code, saturating).
for fmt in ("e4m3", "e5m2"):
    vals = F.decode(torch.arange(256, device=dev, dtype=torch.uint8), fmt).float().cpu().numpy()
    fmax = F.FORMATS[fmt][1]
    fin = vals[:128][np.isfinite(vals[:128])]
    x = np.concatenate([np.random.randn(20000) * fmax / 8,
                        np.random.randn(5000) * 1e-3,
                        vals[np.isfinite(vals)],                      # exact codes
                        (fin[:-1] + fin[1:]) / 2,                      # exact ties
                        [fmax * 3, -fmax * 3, 0.0, -0.0]]).astype(np.float32)
    pos = [c for c in range(128) if np.isfinite(vals[c])]
    pv = vals[pos].astype(np.float64)
    ref = []
    for v in x.astype(np.float64):
        d = np.abs(np.abs(v) - pv)
        cand = [pos[i] for i in np.nonzero(d == d.min())[0]]
        c = min(cand, key=lambda c: (c & 1, c))
        ref.append(c | (0x80 if np.signbit(v) else 0))
    got = F.encode(torch.from_numpy(x).to(dev), fmt).cpu().numpy()
    check(f"{fmt}: torch saturating RNE == demo encoder ({len(x)} values)", np.array_equal(got, np.array(ref)))

# 2. Values already representable as fp8*scale survive fake_quant exactly (all granularities).
for g in ("tensor", "block"):
    q = torch.randint(0, 256, (64, 96), device=dev, dtype=torch.uint8)
    q[(q & 0x7F) == 0x7F] = 0  # drop NaN codes
    base = F.decode(q, "e4m3").float()
    if g == "block":  # blocks may differ in range; a single tensor scale may not
        base[:, :32] *= 8.0
        base[:, 0], base[:, 32], base[:, 64] = 448 * 8, 448, 448  # pin each block's amax
    else:
        base[0, 0] = 448
    for dim in (0, -1) if g == "block" else (-1,):
        if g == "block" and dim == 0:
            continue  # rows=64 blocks would mix the column blocks pinned above
        y, _ = F.fake_quant(base, "e4m3", g, dim, ue8m0=True)
        check(f"fake_quant idempotent on representable data ({g}, dim={dim})", torch.equal(y, base))

# 3. Backward formulas: with quantization replaced by identity, grads == nn.Linear grads.
lin = torch.nn.Linear(64, 96).to(dev)
x = torch.randn(128, 64, device=dev, requires_grad=True)
g = torch.randn(128, 96, device=dev)
lin(x).backward(g)
ref = (x.grad.clone(), lin.weight.grad.clone(), lin.bias.grad.clone())
r = F.Recipe("id")
r.q = lambda t, fmt, dim: t.float()
x.grad = None
m = F.FP8Linear(64, 96).to(dev)
m.load_state_dict(lin.state_dict())
m.recipe = r
m(x).backward(g)
got = (x.grad, m.weight.grad, m.bias.grad)
check("FP8Linear backward (identity quant) == nn.Linear", all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, ref)))

# 4. Block scaling really is per block along the requested dim.
t = torch.randn(64, 64, device=dev)
t[:, :32] *= 1e-5
yb, _ = F.fake_quant(t, "e4m3", "block", -1)
yt, _ = F.fake_quant(t, "e4m3", "tensor", -1)
eb = ((yb - t)[:, :32].abs() / t[:, :32].abs()).median()
et = ((yt - t)[:, :32].abs() / t[:, :32].abs()).median()
check(f"block scale keeps small block accurate (median rel err block {eb:.3f} vs tensor {et:.3f})", eb < 0.05 < et)

sys.exit(0 if ok else 1)
