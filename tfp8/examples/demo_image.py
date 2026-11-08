"""Image demo: a small flow-matching DiT on the Turing fp8 path, following the doc's image policy.

    python -m tfp8.examples.demo_image --steps 40 --size 512

The planner puts the transformer's large Linear/MLP/attention projections on FP8 W8A16 and leaves
norms, time embedding, adaLN modulation and the final output layer in FP16 -- the VAE-equivalent
decoder is FP16 too. We compare against FP16 (the reference on Turing where BF16 is slow) with
PSNR, SSIM and a weight/peak-VRAM comparison.
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import tfp8
from tfp8.planner import weight_bytes

SIZES = {"16": 16, "32": 32, "64": 64}   # image side in pixels (latent = image / 2)
NOISE = 0.6


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class PatchEmbed(nn.Module):
    """Conv-as-patchifier: patch pixels -> tokens. Stays in the input precision."""

    def __init__(self, img, patch, dim, cin):
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(o)


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DiTBlock(nn.Module):
    """Pre-norm block with adaLN modulation (modulation stays in FP16: precision-sensitive)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim)
        self.adaln = nn.Linear(dim, dim * 6)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.adaln(F.silu(c))[:, None].chunk(6, -1)
        x = x + (1 + g1) * self.attn((1 + s1) * self.norm1(x) + b1)
        x = x + (1 + g2) * self.mlp((1 + s2) * self.norm2(x) + b2)
        return x


class DiT(nn.Module):
    def __init__(self, img, patch, dim, depth, heads, cin=3):
        super().__init__()
        self.patch_size, self.img = patch, img
        self.x_embed = PatchEmbed(img, patch, dim, cin)
        self.t_dim = 256                                  # fixed timestep-embedding width
        self.t_embed = nn.Sequential(nn.Linear(self.t_dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln_out = nn.Linear(dim, dim * 2)
        self.proj_out = nn.Linear(dim, patch * patch * cin)   # output layer: stays FP16
        self.pos = nn.Parameter(torch.zeros(1, (img // patch) ** 2, dim))

    def forward(self, x, t):
        B = x.shape[0]
        c = self.t_embed(timestep_embedding(t, self.t_dim).to(self.pos.dtype))
        h = self.x_embed(x) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        s, b = self.adaln_out(F.silu(c))[:, None].chunk(2, -1)
        h = (1 + s) * self.norm_out(h) + b
        h = self.proj_out(h)                                   # [B, T, patch*patch*C]
        B, T, _ = h.shape
        g = self.img // self.patch_size
        h = h.reshape(B, g, g, self.patch_size, self.patch_size, -1)
        h = h.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, self.img, self.img)  # unpatchify
        return h


class TinyVAE(nn.Module):
    """Decoder stand-in: latent -> pixels. Permanently FP16 per the doc (no FP8 here)."""

    def __init__(self, latent, img, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, padding=1))

    def forward(self, z):
        # VAE runs in its own precision (FP16): cast the latent in, not the weights out.
        z = z.to(next(self.parameters()).dtype)
        return torch.tanh(F.interpolate(self.net(z), scale_factor=2, mode="nearest"))


class Pipeline(nn.Module):
    def __init__(self, img, dim, depth, heads, patch=2):
        super().__init__()
        self.transformer = DiT(img // 2, patch, dim, depth, heads)
        self.vae = TinyVAE(3, img)


def build(cfg_size, dim, depth, heads, dtype, seed=0):
    torch.manual_seed(seed)
    pipe = Pipeline(cfg_size, dim, depth, heads)
    pipe = pipe.to(dtype=dtype, device="cuda")
    return pipe


@torch.no_grad()
def sample(pipe, steps, seed=1, img=32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(1, 3, img // 2, img // 2, generator=g, device="cuda", dtype=torch.float32)
    dt = next(pipe.transformer.parameters()).dtype
    z = z.to(dt)
    for i in range(steps):
        t = torch.full((1,), i / steps, device="cuda")
        v = pipe.transformer(z, t)
        z = z - v / steps
    return pipe.vae(z.float())


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def ssim(a, b, c1=0.01 ** 2, c2=0.03 ** 2):
    a, b = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
    k = 11
    mu_a, mu_b = F.avg_pool2d(a, k, 1), F.avg_pool2d(b, k, 1)
    va = F.avg_pool2d(a * a, k, 1) - mu_a ** 2
    vb = F.avg_pool2d(b * b, k, 1) - mu_b ** 2
    cab = F.avg_pool2d(a * b, k, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean().item()


def run(cfg_size=32, dim=256, depth=4, heads=8, steps=20, plan=False):
    print(f"== DiT img {cfg_size}px, token dim {dim}, depth {depth}, heads {heads}, {steps} sampling steps\n")
    ref = build(cfg_size, dim, depth, heads, torch.float16)
    sample(ref, 2, img=cfg_size)                       # warmup: cuBLAS/SDPA autotune, allocator
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    img_ref = sample(ref, steps, img=cfg_size)
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t0
    peak_ref = torch.cuda.max_memory_allocated() / 2**20
    bytes_ref = sum(weight_bytes(ref).values())

    # Same weights, prepared under the image plan. re-seed so both start from identical weights.
    torch.manual_seed(0)
    pipe = Pipeline(cfg_size, dim, depth, heads).to(dtype=torch.float16, device="cuda")
    pipe = tfp8.prepare_model(pipe, "image", tfp8.FP8Config(profile="image_safe", fmt="e4m3fn"))
    bytes_fp8 = sum(weight_bytes(pipe).values())
    if plan:
        print(tfp8.report(pipe))
    sample(pipe, 2, img=cfg_size)                      # warmup (also loads the NVRTC module)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    img_fp8 = sample(pipe, steps, img=cfg_size)
    torch.cuda.synchronize()
    t_fp8 = time.perf_counter() - t0
    peak_fp8 = torch.cuda.max_memory_allocated() / 2**20

    n8 = sum(1 for d in pipe._tfp8_plan if d.owner == "tfp8")
    kept = [d.name for d in pipe._tfp8_plan if d.owner != "tfp8"]
    print(f"planner: {n8} Linear on tfp8, {len(kept)} kept")
    print(f"  kept: {', '.join(sorted(set(k.split('.')[-1] for k in kept)))}")
    print(f"\nweights  FP16 {bytes_ref / 2**20:7.2f} MB   tfp8 {bytes_fp8 / 2**20:7.2f} MB "
          f"({100 * (1 - bytes_fp8 / bytes_ref):.1f}% smaller)")
    print(f"peak VRAM FP16 {peak_ref:7.1f} MB   tfp8 {peak_fp8:7.1f} MB")
    print(f"latency  FP16 {t_ref * 1e3 / steps:6.2f} ms/step   tfp8 {t_fp8 * 1e3 / steps:6.2f} ms/step")
    print(f"quality  PSNR {psnr(img_ref, img_fp8):6.2f} dB   SSIM {ssim(img_ref, img_fp8):.4f}   "
          f"max|d| {(img_ref - img_fp8).abs().max():.4f}")
    paths = {}
    for m in pipe.modules():
        if isinstance(m, tfp8.FP8Linear):
            for k, v in m.stats.items():
                paths[k] = paths.get(k, 0) + v
    print(f"paths    {paths}   runtime fallbacks: "
          f"{sum(1 for m in pipe.modules() if isinstance(m, tfp8.FP8Linear) and m.fallback is not None)}")
    return dict(psnr=psnr(img_ref, img_fp8), ssim=ssim(img_ref, img_fp8),
                bytes_fp16=bytes_ref / 2**20, bytes_tfp8=bytes_fp8 / 2**20,
                ms_step_fp16=t_ref * 1e3 / steps, ms_step_tfp8=t_fp8 * 1e3 / steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32, choices=[16, 32, 64])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    run(a.size, a.dim, a.depth, a.heads, a.steps, a.plan)
