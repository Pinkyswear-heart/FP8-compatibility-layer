"""LLM demo: a Llama-style decoder on the Turing fp8 path (W8A16) vs native BF16.

    python -m tfp8.examples.demo_llm --size tiny     # 2 layers, a few seconds
    python -m tfp8.examples.demo_llm --size 1b       # ~1.0 B params, needs ~2 GB VRAM
    python -m tfp8.examples.demo_llm --size 7b       # memory report only (weights ~14 GB fp16)

Checks the doc's LLM gates: prefill/decode latency, logits agreement vs the BF16 model, peak
memory, and which modules the planner kept out of FP8.
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import tfp8
from tfp8 import _ext
from tfp8.linear import FP8Linear
from tfp8.planner import weight_bytes

SIZES = {
    "tiny": dict(dim=1024, n_layers=2, n_heads=8, n_kv=2, ffn=2816, vocab=8192, seq=256),
    "1b": dict(dim=2048, n_layers=16, n_heads=16, n_kv=4, ffn=5632, vocab=32000, seq=512),
    "7b": dict(dim=4096, n_layers=32, n_heads=32, n_kv=8, ffn=11008, vocab=32000, seq=512),
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Norm is precision-sensitive: compute in FP32, return in the input dtype (per the doc).
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.w.float()).to(x.dtype)


def rope(x, pos):
    """x: [B, T, H, head_dim] -> rotated pairs. pos: [T] absolute positions."""
    hd = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = torch.outer(pos.float(), inv)                  # [T, hd/2]
    dt = x.dtype
    cos = ang.cos()[None, :, None, :].to(dt)
    sin = ang.sin()[None, :, None, :].to(dt)
    x = x.view(*x.shape[:-1], hd // 2, 2)
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c["n_heads"], c["n_kv"], c["dim"] // c["n_heads"]
        self.q_proj = nn.Linear(c["dim"], self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(c["dim"], self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(c["dim"], self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, c["dim"], bias=False)

    def forward(self, x, pos, past=None):
        B, T, D = x.shape
        q = rope(self.q_proj(x).view(B, T, self.nh, self.hd), pos).transpose(1, 2)
        k = rope(self.k_proj(x).view(B, T, self.nkv, self.hd), pos).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if past is not None:                     # cache is [B, n_kv, T, hd]: time is dim 2
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        rep = self.nh // self.nkv
        k2, v2 = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        # SDPA: fp16 memory-efficient kernel; softmax internally in FP32.
        o = F.scaled_dot_product_attention(q, k2, v2, is_causal=(past is None and T > 1))
        return self.o_proj(o.transpose(1, 2).reshape(B, T, D)), (k, v)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c["dim"], c["ffn"], bias=False)
        self.up_proj = nn.Linear(c["dim"], c["ffn"], bias=False)
        self.down_proj = nn.Linear(c["ffn"], c["dim"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c["dim"])
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c["dim"])
        self.mlp = MLP(c)

    def forward(self, x, pos, past=None):
        h, kv = self.self_attn(self.input_layernorm(x), pos, past)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class Llama(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c["vocab"], c["dim"])
        self.layers = nn.ModuleList([Block(c) for _ in range(c["n_layers"])])
        self.norm = RMSNorm(c["dim"])
        self.lm_head = nn.Linear(c["dim"], c["vocab"], bias=False)
        self.lm_head.weight = self.embed.weight          # tied
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, past=None):
        T = idx.shape[1]
        start = past[0][0].shape[2] if past is not None else 0
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.embed(idx)
        kvs = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, pos, None if past is None else past[i])
            kvs.append(kv)
        logits = F.linear(self.norm(x).float(), self.embed.weight.float())
        return logits, kvs


def build(size, dtype):
    c = dict(SIZES[size])
    torch.manual_seed(0)
    m = Llama(c).to(dtype=dtype, device="cuda")
    return m, c


@torch.no_grad()
def prefill(model, ids):
    return model(ids)


@torch.no_grad()
def decode(model, ids, n, past):
    t0 = time.perf_counter()
    out = []
    for i in range(n):
        logits, past = model(ids, past)
        ids = logits[:, -1].argmax(-1, keepdim=True)
        out.append(ids.item())
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3, out


def run(size="tiny", prompt_len=None, n_decode=16):
    c = SIZES[size]
    prompt_len = prompt_len or min(c["seq"], 256)
    print(f"== model: {size}  dim={c['dim']} layers={c['n_layers']} ffn={c['ffn']} vocab={c['vocab']}\n")

    base, _ = build(size, torch.bfloat16)
    ids = torch.randint(0, c["vocab"], (1, prompt_len), device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        logits_ref, _ = base(ids)
    bf16_bytes = sum(v for v in weight_bytes(base).values())

    res = {"size": size, "prompt_len": prompt_len, "n_decode": n_decode, "baseline": "bf16"}
    print(f"bf16  weight+state {bf16_bytes / 2**20:8.1f} MB   "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:8.1f} MB")
    model = tfp8.prepare_model(base, "llm", tfp8.FP8Config(profile="llm_safe", fmt="e4m3fn",
                                                           weight_group=None, nan_check=True))
    fp8_bytes = sum(v for v in weight_bytes(model).values())
    torch.cuda.synchronize()
    with torch.no_grad():
        logits_fp8, _ = model(ids)
    cos = F.cosine_similarity(logits_ref.reshape(-1, c["vocab"]).float(),
                              logits_fp8.reshape(-1, c["vocab"]).float(), dim=-1).mean().item()
    top1 = (logits_ref.argmax(-1) == logits_fp8.argmax(-1)).float().mean().item()
    rel = ((logits_fp8 - logits_ref).norm() / logits_ref.norm()).item()
    print(f"tfp8  weight+state {fp8_bytes / 2**20:8.1f} MB   "
          f"({100 * (1 - fp8_bytes / bf16_bytes):.1f}% smaller)")
    print(f"logits vs bf16: cosine {cos:.6f}   relative L2 {rel:.4f}   top-1 agreement {top1 * 100:.2f}%")
    res.update(cos=cos, rel=rel, top1=top1, bf16_mb=bf16_bytes / 2**20, tfp8_mb=fp8_bytes / 2**20)

    # latency: prefill and decode, bf16 vs tfp8, same weights
    def _median(fn, n):
        import statistics
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    def bench(m, label):
        m.eval()
        with torch.no_grad():
            for _ in range(3):                      # warmup (not counted)
                m(ids)
            pas = None
            for _ in range(3):
                _, pas = m(ids[:, :1])
            torch.cuda.synchronize()
            # median of repeats: a single timing on this card varies by 2x run to run
            prefill_ms = _median(lambda: m(ids), 7)
            one = _median(lambda: m(ids[:, :1], pas), 15)
        n = len(m.layers)
        kinds = {}
        for mod in m.modules():
            if isinstance(mod, FP8Linear):
                for k, v in mod.stats.items():
                    kinds[k] = kinds.get(k, 0) + v
        print(f"  {label:5s} prefill {prompt_len} tok: {prefill_ms:8.2f} ms  "
              f"decode/token {one:6.3f} ms   paths={kinds}")
        return prefill_ms, one

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    p_b, d_b = bench(base, "bf16")
    peak_b = torch.cuda.max_memory_allocated() / 2**20
    del base
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    p_f, d_f = bench(model, "tfp8")
    peak_f = torch.cuda.max_memory_allocated() / 2**20
    print(f"  peak VRAM incl. activations: bf16 {peak_b:.0f} MB   tfp8 {peak_f:.0f} MB")
    print(f"  speed: decode {d_b / d_f:5.2f}x, prefill {p_b / p_f:5.2f}x  (tfp8 vs bf16)")
    res.update(prefill_ms_bf16=p_b, prefill_ms_tfp8=p_f, decode_ms_bf16=d_b,
               decode_ms_tfp8=d_f, decode_speedup=d_b / d_f, prefill_speedup=p_b / p_f,
               peak_mb_bf16=peak_b, peak_mb_tfp8=peak_f)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(SIZES))
    ap.add_argument("--prompt-len", type=int, default=None)
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--plan", action="store_true", help="print the per-module plan")
    a = ap.parse_args()
    if a.size == "7b":
        c = SIZES["7b"]
        n = c["n_layers"] * (4 * c["dim"] ** 2 + 3 * c["dim"] * c["ffn"]) + c["vocab"] * c["dim"]
        print(f"7b: ~{n / 1e9:.2f} B params -> BF16/FP16 {2 * n / 2**30:.1f} GiB, "
              f"FP8 {n / 2**30:.1f} GiB on disk/in VRAM; building a random 7b on this GPU is not "
              f"the point of the demo, so run --size 1b for measurements.")
        raise SystemExit(0)
    out = run(a.size, a.prompt_len, a.decode)
