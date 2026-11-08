"""Bridge from PyTorch to the fused FP8 GEMM kernel (fp8_gemm_kernel.cu), compiled with NVRTC via ctypes."""
import ctypes
import glob
import os

import torch



def _cuda_root():
    """CUDA_PATH / CUDA_HOME if set, else the newest toolkit under the default Windows install dir."""
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            return os.environ[var]
    base = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                        "NVIDIA GPU Computing Toolkit", "CUDA")
    found = sorted(glob.glob(os.path.join(base, "v*")))
    if not found:
        raise RuntimeError("CUDA toolkit not found: set CUDA_PATH")
    return found[-1]


_ROOT = _cuda_root()
CUDA_BIN = os.path.join(_ROOT, "bin")
CUDA_INC = os.path.join(_ROOT, "include")
KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp8_gemm_kernel.cu")

FORMATS = {  # name -> (torch dtype, max finite)
    "e4m3": (torch.float8_e4m3fn, 448.0),
    "e5m2": (torch.float8_e5m2, 57344.0),
}
BLOCK = 32

_cu = _mod = None


def _init():
    global _cu, _mod
    if _mod is not None:
        return
    torch.zeros(1, device="cuda")  # make torch's primary context current on this thread
    os.add_dll_directory(CUDA_BIN)
    nvrtc = ctypes.CDLL(glob.glob(os.path.join(CUDA_BIN, "nvrtc64_*_0.dll"))[0])
    _cu = ctypes.CDLL("nvcuda.dll")

    major, minor = torch.cuda.get_device_capability()
    src = open(KERNEL, "rb").read()
    prog = ctypes.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"fp8_gemm_kernel.cu", 0, None, None) == 0
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"-default-device",
            b"-I" + CUDA_INC.encode()]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    if n.value > 1:
        log = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        print(log.value.decode())
    assert rc == 0, "NVRTC compile failed"
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    cubin = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    _mod = ctypes.c_void_p()
    assert _cu.cuModuleLoadData(ctypes.byref(_mod), cubin) == 0, "cuModuleLoadData failed"


def lut(fmt):
    """uint8 -> fp16 table (bit-identical to fp8.h's LUT, see tests/verify_lut.py)."""
    dt, _ = FORMATS[fmt]
    return torch.arange(256, dtype=torch.uint8, device="cuda").view(dt).to(torch.float16)


def quantize_k(x, fmt, ue8m0=False):
    """Quantize x (R, K) in blocks of 32 along K. Returns uint8 codes (R,K), scales (R,K/32).
    Scales are FP32, or UE8M0 bytes (value 2^(e-127)) when ue8m0."""
    dt, mx = FORMATS[fmt]
    if not x.is_cuda:
        raise ValueError("quantize_k: x must be a CUDA tensor")
    R, K = x.shape
    if K % BLOCK:
        raise ValueError(f"quantize_k: K={K} must be a multiple of {BLOCK}")
    xb = x.float().reshape(R, K // BLOCK, BLOCK)
    s = (xb.abs().amax(-1) / mx).clamp_min(1e-12)
    if ue8m0:
        e = torch.ceil(torch.log2(s)).clamp(-127, 127)
        s = torch.exp2(e)
        s_out = (e + 127).to(torch.uint8)
    else:
        s_out = s
    q = (xb / s[..., None]).clamp(-mx, mx).to(dt).view(torch.uint8).reshape(R, K)
    return q, s_out


class Fp8Weight:
    """B (K,N) quantized along K per output column; scales laid out (K/32, N)."""

    def __init__(self, w, fmt="e4m3", ue8m0=False):
        if not w.is_cuda:
            raise ValueError("Fp8Weight: w must be a CUDA tensor")
        if w.shape[0] % BLOCK:
            raise ValueError(f"Fp8Weight: K={w.shape[0]} must be a multiple of {BLOCK}")
        q, s = quantize_k(w.t().contiguous(), fmt, ue8m0)
        self.q, self.s = q.t().contiguous(), s.t().contiguous()
        self.fmt, self.ue8m0 = fmt, ue8m0
        self.K, self.N = w.shape


def fp8_linear(x, w, bias, act_fmt=None):
    """x (M,K) float -> fused FP8 GEMM -> (M,N) fp32 (+bias). Activations are quantized
    dynamically per row and 32-wide K block, in the same format family as the weights."""
    _init()
    act_fmt = act_fmt or w.fmt
    M, K = x.shape
    if not (x.is_cuda and w.q.is_cuda and w.s.is_cuda and bias.is_cuda):
        raise ValueError("fp8_linear: x, weights and bias must all be CUDA tensors")
    if not (x.device == w.q.device == w.s.device == bias.device):
        raise ValueError(f"fp8_linear: tensors on different devices ({x.device}, {w.q.device}, "
                         f"{w.s.device}, {bias.device})")
    if M % 32 or K % 32 or w.N % 32 or K != w.K:
        raise ValueError(f"fp8_linear: need M, K, N multiples of 32 and K == w.K; "
                         f"got M={M} K={K} N={w.N} w.K={w.K}")
    aq, sa = quantize_k(x, act_fmt, w.ue8m0)
    aq, sa = aq.contiguous(), sa.contiguous()
    la, lb = lut(act_fmt), lut(w.fmt)
    out = torch.empty(M, w.N, dtype=torch.float32, device="cuda")
    bias = bias.float().contiguous()
    name = b"gemm_block_" + (b"ue8m0" if w.ue8m0 else b"fp32")
    fn = ctypes.c_void_p()
    assert _cu.cuModuleGetFunction(ctypes.byref(fn), _mod, name) == 0
    ptrs = [ctypes.c_void_p(t.data_ptr()) for t in (aq, w.q, la, lb, sa, w.s, bias, out)]
    ints = [ctypes.c_int(v) for v in (M, w.N, K)]
    args = (ctypes.c_void_p * 11)(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in ptrs + ints])
    torch.cuda.synchronize()
    rc = _cu.cuLaunchKernel(fn, w.N // 32, M // 32, 1, 128, 1, 1, 0, None, args, None)
    assert rc == 0, f"cuLaunchKernel failed: {rc}"
    torch.cuda.synchronize()
    return out


def dequant_ref(q, s, fmt, ue8m0, k_axis_last):
    """float64 dequantization used as the reference. q,s as produced above (A layout if k_axis_last)."""
    v = lut(fmt)[q.long()].double()
    sc = torch.exp2(s.double() - 127) if ue8m0 else s.double()
    if k_axis_last:  # (R,K), scales (R,K/32)
        return v * sc.repeat_interleave(BLOCK, dim=1)
    return v * sc.repeat_interleave(BLOCK, dim=0)  # (K,N), scales (K/32,N)
