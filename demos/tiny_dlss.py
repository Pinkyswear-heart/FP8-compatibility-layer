"""Tiny DLSS-style temporal upscaler (2x) on a procedural scene, with FP32 vs fused-FP8 inference.

Inputs per output pixel: jitter-compensated upsampled current frame + motion-reprojected history.
Network: 3x3 gather (54 feats) -> 3 per-pixel linear layers. The linear layers run either in FP32
(torch) or through the fused LUT-dequant FP8 GEMM kernel (W8A8, FP32 accumulate, block-32 scales).

Run from the repo root: python -I demos/tiny_dlss.py --steps 800
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import fp8_cuda

dev = "cuda"
H = W = 128
S = 2
LH, LW = H // S, W // S
KPAD = 64  # 3x3 * 6 channels = 54, zero-padded to a multiple of 32
OPAD = 32  # 3 output channels padded to 32


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def jitter(t):
    return halton(t + 1, 2) - 0.5, halton(t + 1, 3) - 0.5


# ---------------------------------------------------------------- procedural scene
def make_params(B, gen):
    r = lambda lo, hi, *s: (torch.rand(*s, generator=gen) * (hi - lo) + lo).to(dev)
    return dict(
        pan=r(-3, 3, B, 2), ph=r(0, 6.28, B, 3),
        c0=r(40, 88, B, 2), cv=r(-4, 4, B, 2), rad=r(14, 26, B),
    )


def _p(P, k, i=None):
    v = P[k] if i is None else P[k][:, i]
    return v.reshape(-1, *([1] * 2))


def render(x, y, t, P):
    """Continuous scene at HR coordinates x,y (B,h,w) and time t -> (B,3,h,w) and inside-circle mask."""
    u = x - _p(P, "pan", 0) * t
    v = y - _p(P, "pan", 1) * t
    chk = (torch.floor(u / 8) + torch.floor(v / 8)) % 2
    stripe = ((u + 2 * v) / 11 % 1.0 < 0.15).float()
    bg = torch.stack([
        0.40 + 0.30 * chk + 0.20 * torch.sin(0.31 * u + _p(P, "ph", 0)),
        0.40 + 0.20 * chk + 0.25 * torch.sin(0.23 * v + _p(P, "ph", 1)) - 0.25 * stripe,
        0.45 + 0.10 * chk + 0.30 * torch.sin(0.17 * (u + v) + _p(P, "ph", 2)) + 0.25 * stripe,
    ], 1)
    lu = x - (_p(P, "c0", 0) + _p(P, "cv", 0) * t)
    lv = y - (_p(P, "c0", 1) + _p(P, "cv", 1) * t)
    inside = (lu * lu + lv * lv) < _p(P, "rad") ** 2
    fg = torch.stack([
        0.5 + 0.5 * torch.sin(0.7 * lu + 0.4 * lv),
        0.5 + 0.5 * torch.sin(0.5 * lv + 1.0),
        0.5 + 0.5 * torch.cos(0.9 * lu - 0.3 * lv),
    ], 1)
    return torch.where(inside[:, None], fg, bg), inside


def grid(h, w, B, step=1.0, off=0.5):
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    x = (xs.float() + off) * step
    y = (ys.float() + off) * step
    return x.expand(B, h, w), y.expand(B, h, w)


def gt_frame(t, P, B):
    acc = 0
    for a in range(3):
        for b in range(3):
            xx, yy = grid(H, W, B, off=0)
            acc = acc + render(xx + (a + 0.5) / 3, yy + (b + 0.5) / 3, t, P)[0]
    return acc / 9


def lr_frame(t, P, B, jx, jy):
    x, y = grid(LH, LW, B, step=S, off=0.5 + 0.0)
    return render(x + jx * S, y + jy * S, t, P)[0]  # one aliased sample per LR pixel


def mv_field(t, P, B):
    x, y = grid(H, W, B)
    inside = render(x, y, t, P)[1]
    pan = P["pan"][:, :, None, None].expand(B, 2, H, W)
    cv = P["cv"][:, :, None, None].expand(B, 2, H, W)
    return torch.where(inside[:, None], cv, pan)


def warp(img, mv):
    B = img.shape[0]
    x, y = grid(H, W, B)
    g = torch.stack([2 * (x - mv[:, 0]) / W - 1, 2 * (y - mv[:, 1]) / H - 1], -1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)


def upsample_lr(lr, jx, jy):
    """Bicubic upsample of the jittered LR frame, compensating the sub-pixel jitter."""
    B = lr.shape[0]
    x, y = grid(H, W, B)
    ix = x / S - 0.5 - jx
    iy = y / S - 0.5 - jy
    g = torch.stack([(2 * ix + 1) / LW - 1, (2 * iy + 1) / LH - 1], -1)
    return F.grid_sample(lr, g, mode="bicubic", padding_mode="border", align_corners=False)


# ---------------------------------------------------------------- network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(54, 64), nn.Linear(64, 64), nn.Linear(64, 3)

    def forward(self, f):  # f: (M, 64) padded features
        h = F.relu(F.linear(f[:, :54], self.l1.weight, self.l1.bias))
        h = F.relu(self.l2(h))
        return self.l3(h)

    def export(self):
        """Padded (K,N) weights + biases for the GEMM path."""
        w1 = torch.zeros(KPAD, 64, device=dev)
        w1[:54] = self.l1.weight.detach().t()
        w3 = torch.zeros(64, OPAD, device=dev)
        w3[:, :3] = self.l3.weight.detach().t()
        b3 = torch.zeros(OPAD, device=dev)
        b3[:3] = self.l3.bias.detach()
        return [(w1, self.l1.bias.detach()), (self.l2.weight.detach().t().contiguous(), self.l2.bias.detach()),
                (w3, b3)]


def features(cur_up, hist):
    B = cur_up.shape[0]
    x = torch.cat([cur_up, hist], 1)
    f = F.unfold(x, 3, padding=1).transpose(1, 2).reshape(B * H * W, 54)
    return F.pad(f, (0, KPAD - 54))


def run_frame(mlp, lr, t, P, prev, B):
    jx, jy = jitter(t)
    cur_up = upsample_lr(lr, jx, jy)
    hist = cur_up if prev is None else warp(prev, mv_field(t, P, B))
    delta = mlp(features(cur_up, hist))
    return cur_up + delta[:, :3].reshape(B, H, W, 3).permute(0, 3, 1, 2)


def mlp_fp32(layers):
    def f(x):
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if i < 2:
                x = F.relu(x)
        return x
    return f


def mlp_fp8(layers, fmt, ue8m0):
    qw = [(fp8_cuda.Fp8Weight(w, fmt, ue8m0), b) for w, b in layers]
    def f(x):
        for i, (w, b) in enumerate(qw):
            x = fp8_cuda.fp8_linear(x, w, b)  # FP8 GEMM -> FP32
            if i < 2:
                x = F.relu(x)  # activation stays FP32 between layers
        return x
    return f


# ---------------------------------------------------------------- data / train / eval
def sequence(B, T, gen):
    P = make_params(B, gen)
    return P, [(lr_frame(t, P, B, *jitter(t)), gt_frame(t, P, B)) for t in range(T)]


def train(net, steps, B=4):
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    for it in range(steps):
        P, seq = sequence(B, 3, gen)
        prev, loss = None, 0
        for t, (lr, gt) in enumerate(seq):
            out = run_frame(net, lr, t, P, prev, B)
            loss = loss + F.l1_loss(out, gt)
            prev = out.detach()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == steps - 1:
            print(f"  step {it:4d}  loss {loss.item() / 3:.4f}  ({time.time() - t0:.0f}s)")


def psnr(a, b):
    mse = ((a.clamp(0, 1) - b) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1 / mse)).mean().item()


@torch.no_grad()
def evaluate(mlp, B=8, T=8, seed=123):
    gen = torch.Generator().manual_seed(seed)
    P, seq = sequence(B, T, gen)
    prev, outs = None, []
    for t, (lr, gt) in enumerate(seq):
        prev = run_frame(mlp, lr, t, P, prev, B)
        outs.append(prev)
    return psnr(outs[-1], seq[-1][1]), outs[-1], seq[-1]


@torch.no_grad()
def kernel_selfcheck(layers, fmt, ue8m0):
    """Fused kernel vs float64 matmul of the dequantized operands, on a real activation."""
    gen = torch.Generator().manual_seed(7)
    P, seq = sequence(1, 2, gen)
    lr, _ = seq[1]
    cur = upsample_lr(lr, *jitter(1))
    x = features(cur, warp(cur, mv_field(1, P, 1)))
    w, b = layers[1][0], layers[1][1]
    x = F.relu(x[:, :54] @ layers[0][0][:54] + layers[0][1])
    qw = fp8_cuda.Fp8Weight(w, fmt, ue8m0)
    got = fp8_cuda.fp8_linear(x, qw, b).double()
    aq, sa = fp8_cuda.quantize_k(x, fmt, ue8m0)
    ref = fp8_cuda.dequant_ref(aq, sa, fmt, ue8m0, True) @ fp8_cuda.dequant_ref(qw.q, qw.s, fmt, ue8m0, False) + b.double()
    return ((got - ref).abs().max() / ref.abs().max()).item()


def save_grid(path, gt, bicubic, outs):
    tiles = [gt, bicubic] + outs
    img = torch.cat([t[0].clamp(0, 1) for t in tiles], 2)  # (3, H, n*W)
    img = F.interpolate(img[None], scale_factor=3, mode="nearest")[0]
    Image.fromarray((img.permute(1, 2, 0) * 255).byte().cpu().numpy()).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    torch.manual_seed(0)

    net = Net().to(dev)
    print(f"training {args.steps} steps on GPU ({torch.cuda.get_device_name()})")
    train(net, args.steps)
    net.eval()
    layers = net.export()

    results = {}
    _, _, (lr, gt) = evaluate(mlp_fp32(layers))
    # bicubic-only baseline on the same eval sequence
    gen = torch.Generator().manual_seed(123)
    P, seq = sequence(8, 8, gen)
    bic = upsample_lr(seq[-1][0], *jitter(7))
    results["bicubic (no network)"] = psnr(bic, seq[-1][1])

    configs = [("network FP32 (torch)", None)]
    configs += [(f"network FP8 {f} {'UE8M0' if u else 'FP32'}-scale (fused kernel)", (f, u))
                for f, u in (("e4m3", False), ("e4m3", True), ("e5m2", False))]
    outs = []
    for name, cfg in configs:
        mlp = mlp_fp32(layers) if cfg is None else mlp_fp8(layers, *cfg)
        p, out, _ = evaluate(mlp)
        results[name] = p
        outs.append(out)
        if cfg is not None:
            print(f"  kernel self-check [{cfg[0]}, ue8m0={cfg[1]}]: max rel err vs float64 = "
                  f"{kernel_selfcheck(layers, *cfg):.2e}")

    print("\nPSNR vs 3x3-supersampled ground truth (frame 8, mean over 8 sequences):")
    for k, v in results.items():
        print(f"  {k:<52s} {v:6.2f} dB")
    save_grid("dlss_out.png", gt, bic, [outs[0], outs[1]])
    print("\nsaved dlss_out.png: ground truth | bicubic | network FP32 | network FP8 e4m3 (first sequence)")


if __name__ == "__main__":
    main()
