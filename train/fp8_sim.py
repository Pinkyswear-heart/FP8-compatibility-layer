"""FP8 training emulation on GPUs without FP8 hardware (here: sm_75).

Each FP8Linear quantizes the operands of all three GEMMs of a Linear layer:
    forward : Y  = X  @ W^T      X, W  -> fwd format (E4M3FN)
    backward: dX = dY @ W        dY    -> bwd format (E5M2 or E4M3FN), W -> fwd format
              dW = dY^T @ X      dY    -> bwd format,                  X -> fwd format
Quantize = scale, saturate to +-max_finite, RNE cast to FP8 code (uint8).
Decode   = the same uint8 -> fp16 LUT the CUDA demo exports (lut_*.bin), times scale.
Products are accumulated in FP32. Master weights, optimizer state, LayerNorm, softmax,
residuals and the attention matmuls stay FP32 (plan step 4).

Block mode scales every 32 elements along the *contraction* dim of the GEMM the operand
feeds, so an operand used by two GEMMs (W, X, dY) is quantized twice, once per layout.
This models numerics only; it does not save memory or time (see README).
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = 32

FORMATS = {
    # name: (torch dtype for RNE cast, max finite, LUT file from the CUDA demo)
    "e4m3": (torch.float8_e4m3fn, 448.0, "lut_e4m3fn.bin"),
    "e5m2": (torch.float8_e5m2, 57344.0, "lut_e5m2.bin"),
}
_LUT = {}


def lut(fmt, device):
    key = (fmt, str(device))
    if key not in _LUT:
        path = os.path.join(HERE, "..", FORMATS[fmt][2])
        bits = np.fromfile(path, dtype=np.uint16).view(np.float16)
        _LUT[key] = torch.from_numpy(bits.copy()).to(device)
    return _LUT[key]


def encode(x, fmt):
    """x already divided by its scale -> uint8 FP8 codes (saturating RNE)."""
    dtype, fmax, _ = FORMATS[fmt]
    return x.clamp(-fmax, fmax).to(dtype).view(torch.uint8)


def decode(q, fmt):
    return lut(fmt, q.device)[q.long()]


def _scale(amax, fmt, ue8m0):
    s = amax / FORMATS[fmt][1]
    s = torch.where(amax > 0, s, torch.ones_like(s))
    if ue8m0:  # power of two, rounded up so amax never overflows
        s = torch.exp2(torch.ceil(torch.log2(s)))
    return s


def fake_quant(x, fmt, granularity, dim, ue8m0=False):
    """Quantize x to FP8 and return (dequantized fp32 tensor, codes)."""
    x = x.float()
    if granularity == "none":
        q = encode(x, fmt)
        return decode(q, fmt).float(), q
    if granularity == "tensor":
        s = _scale(x.abs().amax(), fmt, ue8m0)
        q = encode(x / s, fmt)
        return decode(q, fmt).float() * s, q
    assert granularity == "block"
    xt = x.movedim(dim, -1)
    shp = xt.shape
    assert shp[-1] % BLOCK == 0, f"contraction dim {shp[-1]} not a multiple of {BLOCK}"
    xb = xt.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    s = _scale(xb.abs().amax(-1, keepdim=True), fmt, ue8m0)
    q = encode(xb / s, fmt)
    deq = (decode(q, fmt).float() * s).reshape(shp).movedim(-1, dim)
    return deq, q.reshape(shp).movedim(-1, dim)


class Recipe:
    def __init__(self, name, fwd="e4m3", bwd="e5m2", granularity="tensor", ue8m0=False):
        self.name, self.fwd, self.bwd = name, fwd, bwd
        self.granularity, self.ue8m0 = granularity, ue8m0
        self.stats = {"grad_nonzero": 0, "grad_flushed": 0}

    def q(self, x, fmt, dim):
        return fake_quant(x, fmt, self.granularity, dim, self.ue8m0)[0]


class _FP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, recipe):
        ctx.recipe = recipe
        ctx.save_for_backward(x, w)
        y = recipe.q(x, recipe.fwd, -1) @ recipe.q(w, recipe.fwd, -1).t()
        return y + b if b is not None else y

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        r = ctx.recipe
        # dX: contraction over out_features
        gq_out = r.q(gy, r.bwd, -1)
        gx = gq_out @ r.q(w, r.fwd, 0)
        # dW: contraction over tokens
        gq_tok = gq_out if r.granularity != "block" else r.q(gy, r.bwd, 0)
        gw = gq_tok.t() @ r.q(x, r.fwd, 0)
        nz = gy != 0
        r.stats["grad_nonzero"] += int(nz.sum())
        r.stats["grad_flushed"] += int((nz & (gq_out == 0)).sum())
        return gx, gw, gy.sum(0), None


class FP8Linear(torch.nn.Linear):
    recipe = None  # set per model; None = plain FP32 Linear

    def forward(self, x):
        if self.recipe is None:
            return super().forward(x)
        shp = x.shape
        y = _FP8LinearFn.apply(x.reshape(-1, shp[-1]), self.weight, self.bias, self.recipe)
        return y.reshape(*shp[:-1], -1)
