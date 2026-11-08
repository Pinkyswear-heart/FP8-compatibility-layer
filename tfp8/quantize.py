"""Symmetric FP8 weight quantization (BF16 -> FP8 compression mode) and the activation row scale.

    scale = max|w| / FP8_MAX   (rounded up to a power of two by default)
    q     = RNE(clamp(w / scale, -FP8_MAX, FP8_MAX))
"""
import torch

from .formats import get_format


def pow2_ceil(s: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(s)))


def quantize_weight(w: torch.Tensor, fmt="e4m3fn", group_size=None, pow2_scale=True):
    """w [N,K] -> (qweight uint8 [N,K], scale f32 [N, K/G], G).

    group_size None -> one scale per output channel (G = K).
    Runs on whatever device w lives on; call it layer by layer so the full-precision model
    never has to sit in VRAM next to the quantized one.
    """
    f = get_format(fmt)
    N, K = w.shape
    G = K if group_size is None else group_size
    if K % G:
        raise ValueError(f"K={K} not divisible by group_size={G}")
    wg = w.float().reshape(N, K // G, G)
    amax = wg.abs().amax(-1)
    s = (amax / f.max_finite).clamp_min(torch.finfo(torch.float32).tiny)
    if pow2_scale:
        s = pow2_ceil(s)
    s = torch.where(amax == 0, torch.ones_like(s), s)
    q = (wg / s[..., None]).clamp(-f.max_finite, f.max_finite)
    # torch's float8 cast is RNE; we clamp first so it never has to saturate.
    q = q.to(f.torch_dtype).view(torch.uint8).reshape(N, K)
    return q.contiguous(), s.contiguous(), G


def dequantize_weight(q: torch.Tensor, s: torch.Tensor, fmt, group_size) -> torch.Tensor:
    """Reference dequantization (float32). Never used on the hot path."""
    from .formats import lut
    N, K = q.shape
    v = lut(fmt, q.device)[q.long()].float().reshape(N, K // group_size, group_size)
    return (v * s[..., None]).reshape(N, K)


ACT_TARGET_EXP = 8  # row amax is mapped into [2^7, 2^8): far from fp16 overflow, ~30 binades of headroom


def activation_to_fp16(x: torch.Tensor):
    """BF16/FP32 activation rows -> (fp16 [M,K], power-of-two row scale f32 [M]).

    Dividing by an exact power of two is lossless, so BF16 -> FP16 only loses values that
    would land in fp16's subnormal range (the 'tbf16' trick). FP16 input passes through.
    """
    if x.dtype == torch.float16:
        return x.contiguous(), torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
    amax = x.abs().amax(-1).float()
    e = torch.ceil(torch.log2(amax.clamp_min(2.0 ** -100))) - ACT_TARGET_EXP
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    xs = torch.exp2(e)
    return (x.float() / xs[:, None]).half().contiguous(), xs.contiguous()
