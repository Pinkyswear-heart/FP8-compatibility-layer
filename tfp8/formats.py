"""FP8 formats: bit-field decode, LUT construction, RNE + saturating encode.

Everything here is pure PyTorch and runs on CPU or CUDA; it is the numerical reference for
the CUDA kernels. Formats are never mixed: each has its own table.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Fp8Format:
    name: str
    ebits: int
    mbits: int
    bias: int
    has_inf: bool          # E5M2 is IEEE-like; E4M3FN has no inf
    max_finite: float
    torch_dtype: torch.dtype

    def is_nan(self, c: torch.Tensor) -> torch.Tensor:
        e = (c >> self.mbits) & ((1 << self.ebits) - 1)
        m = c & ((1 << self.mbits) - 1)
        emax = (1 << self.ebits) - 1
        if self.has_inf:
            return (e == emax) & (m != 0)
        return (e == emax) & (m == (1 << self.mbits) - 1)


E4M3FN = Fp8Format("e4m3fn", 4, 3, 7, False, 448.0, torch.float8_e4m3fn)
E5M2 = Fp8Format("e5m2", 5, 2, 15, True, 57344.0, torch.float8_e5m2)
FORMATS = {f.name: f for f in (E4M3FN, E5M2)}


def get_format(fmt) -> Fp8Format:
    if isinstance(fmt, Fp8Format):
        return fmt
    if fmt not in FORMATS:
        raise ValueError(f"unsupported FP8 format {fmt!r}; expected one of {sorted(FORMATS)}")
    return FORMATS[fmt]


def decode(codes: torch.Tensor, fmt) -> torch.Tensor:
    """uint8 codes -> float32, from the bit fields (never an integer cast)."""
    f = get_format(fmt)
    c = codes.to(torch.int32)
    s = (c >> 7) & 1
    e = (c >> f.mbits) & ((1 << f.ebits) - 1)
    m = (c & ((1 << f.mbits) - 1)).float()
    sub = torch.ldexp(m, torch.full_like(e, 1 - f.bias - f.mbits))
    nrm = torch.ldexp(m + (1 << f.mbits), e - f.bias - f.mbits)
    v = torch.where(e == 0, sub, nrm)
    if f.has_inf:
        v = torch.where((e == (1 << f.ebits) - 1) & (c & ((1 << f.mbits) - 1) == 0),
                        torch.full_like(v, float("inf")), v)
    v = torch.where(f.is_nan(c), torch.full_like(v, float("nan")), v)
    return torch.where(s == 1, -v, v)


_LUT_CACHE = {}


def lut(fmt, device="cpu") -> torch.Tensor:
    """256-entry fp16 table. Every FP8 value (incl. subnormals, +-0, NaN, inf) is exact in fp16."""
    f = get_format(fmt)
    key = (f.name, str(device))
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = decode(torch.arange(256, dtype=torch.uint8), f).half().to(device)
    return _LUT_CACHE[key]


def _finite_values(f: Fp8Format) -> torch.Tensor:
    v = decode(torch.arange(128, dtype=torch.uint8), f).double()  # non-negative codes
    return v  # indexed by code; NaN/inf entries handled by the encoder


def encode(x: torch.Tensor, fmt) -> torch.Tensor:
    """float -> uint8 codes. Round-to-nearest-even, saturating to +-max_finite (never produces
    inf), NaN -> canonical NaN. Matches torch's `.to(float8)` for in-range values."""
    f = get_format(fmt)
    xd = x.double()
    ax = xd.abs().clamp(max=f.max_finite)
    table = _finite_values(f).to(x.device)
    finite = torch.isfinite(table)
    codes = torch.arange(128, device=x.device)[finite]
    vals = table[finite]  # monotone increasing in code
    hi = torch.searchsorted(vals, ax.reshape(-1).contiguous()).clamp(max=len(vals) - 1)
    lo = (hi - 1).clamp(min=0)
    dlo = ax.reshape(-1) - vals[lo]
    dhi = vals[hi] - ax.reshape(-1)
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (codes[hi] % 2 == 0))
    q = torch.where(pick_hi, codes[hi], codes[lo]).reshape(x.shape)
    q = q | (torch.signbit(xd).to(torch.int64) << 7)
    nan_code = 0x7F if not f.has_inf else 0x7E
    q = torch.where(torch.isnan(xd), torch.full_like(q, nan_code), q)
    return q.to(torch.uint8)
