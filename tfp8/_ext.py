"""Runtime (NVRTC) compilation of csrc/tfp8_kernels.cu and thin launch wrappers.

NVRTC needs no host C++ compiler, which sidesteps the MSVC/nvcc version mismatch on Windows.
The kernel ABI is uint8 for FP8 codes (not torch.float8_*), as the design doc recommends.
Launch cost matters at M=1 (a whole token step is ~1 ms), so each call is one or two launches.
"""
import os
import threading

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "tfp8_kernels.cu")

# Row widths that get their own kernel instantiation (must match GEMV_MT in the .cu). The GEMV body
# is unrolled over the tile height, so a wide instantiation issues the FMA and reduction blocks of
# rows that this launch does not read; exporting one kernel per width lets a decode step at M=1 skip
# them entirely. Measured at M=1: 1.17-1.24x, and M=2..4 lands at 1.03-1.17x on the width-4 kernel.
GEMV_MT = (1, 4, 8)


def gemv_name(x_tag: str, out_tag: str, M: int) -> str:
    """Exported GEMV symbol for M rows: the narrowest width that covers it."""
    mt = next((w for w in GEMV_MT if w >= M), GEMV_MT[-1])
    return f"tfp8_gemv_in_{x_tag}_m{mt}_{out_tag}"


_NAMES = [f"tfp8_gemv_in_{i}_m{m}_{o}"
          for m in GEMV_MT for i in ("f16", "bf16", "f32") for o in ("f16", "bf16", "f32")] + [
    "tfp8_gemm_f16", "tfp8_gemm_bf16", "tfp8_gemm_f32", "tfp8_dequant_f16",
    "tfp8_epilogue_f16", "tfp8_epilogue_bf16", "tfp8_epilogue_f32",
    "tfp8_prep_bf16", "tfp8_prep_f32", "tfp8_prep_f16"]
_lock = threading.Lock()
_slot_lock = threading.Lock()
_kernels = {}  # device index -> {name: kernel}
_error = None

GEMV_MAXM = 8
GEMV_WARPS = 4
GEMV_ROWS = 1      # output channels per warp; >1 measured no consistent gain (see tests/bench_rows.py)
BM = BN = 64
FMT_ID = {"e4m3fn": 0, "e5m2": 1}
_PREP = {torch.float16: "tfp8_prep_f16", torch.bfloat16: "tfp8_prep_bf16", torch.float32: "tfp8_prep_f32"}


def available() -> bool:
    """True when the CUDA kernels are compiled and loaded. A compile/load failure is recorded in
    last_error() and reported once on stderr -- never silently downgraded to the torch path."""
    global _error
    if not torch.cuda.is_available():
        return False
    try:
        _get(torch.cuda.current_device())
        return True
    except Exception as e:  # noqa: BLE001
        if _error is None:
            _error = e
            import sys
            print(f"[tfp8] FATAL: CUDA kernels unavailable, falling back to the torch reference "
                  f"path: {e}", file=sys.stderr)
        return False


def last_error():
    return _error


def _get(dev: int):
    with _lock:
        if dev not in _kernels:
            from torch.cuda._utils import _cuda_load_module, _nvrtc_compile
            with torch.cuda.device(dev):
                major, minor = torch.cuda.get_device_capability(dev)
                src = open(_SRC, encoding="utf-8").read()
                binary, _ = _nvrtc_compile(src, _NAMES[0], f"{major}{minor}", None, ["--std=c++17"])
                _kernels[dev] = _cuda_load_module(binary, _NAMES)
        return _kernels[dev]


_DRV = {}   # driver function table, cached per process

# A single cuLaunchKernel call through ctypes costs ~28 us on this machine, so the Python around
# it is what decides whether a small layer is 2x or 1.2x slower than cuBLAS. Measured costs:
#   torch.cuda.current_stream()                     9.1 us  -> _cuda_getCurrentRawStream: 0.3 us
#   rebuilding the ctypes argument block per call  33.0 us  -> mutate reused objects: 2.6 us
# Arguments are therefore packed into a cached (signature -> holders, array) slot and only their
# .value fields are rewritten, which does not reallocate anything.


def _driver():
    """cuLaunchKernel bound once, with argtypes set (untyped calls are much slower)."""
    if not _DRV:
        import ctypes
        from torch.cuda._utils import _get_gpu_runtime_library
        launch = _get_gpu_runtime_library().cuLaunchKernel
        launch.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 3 + [ctypes.c_uint] * 3 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        launch.restype = ctypes.c_int
        _DRV["launch"] = launch
        _DRV["ctypes"] = ctypes
        _DRV["stream"] = torch._C._cuda_getCurrentRawStream
        _DRV["slots"] = {}
    return _DRV


_LAUNCH = {}   # name -> (dev, fn) so the module lookup is not repeated per call


def _kinds(args):
    return tuple("t" if isinstance(a, torch.Tensor) else "f" if isinstance(a, float) else "i"
                 for a in args)


def _holders(ctypes, kinds):
    # Python floats map to c_float: every kernel float parameter is a 4-byte `float`. (A c_double
    # holder made the kernel read the low 32 bits of the double -- 0.0 for 8.0 -- so the prep
    # kernel's ACT_TARGET_EXP silently became 0.)
    holders = [ctypes.c_void_p(0) if k == "t" else
               ctypes.c_float(0.0) if k == "f" else ctypes.c_int(0) for k in kinds]
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p)
                                             for h in holders])
    return holders, arr


# Argument-kind signatures of the internal launches, so the hot path does not rebuild them with
# one isinstance chain per argument (~3.7 us for 13 args). Must match the argument lists exactly.
K_W8A16 = ("t",) * 7 + ("i",) * 6          # x, w, lut, ws, xs, bias, y, M, N, K, G, has_bias, fmt
K_PREP = ("t", "t", "t", "i", "i", "f")    # x, x16, xs, M, K, target
K_DEQ = ("t", "t", "t", "t", "i", "i", "i", "i")
K_EPI = ("t", "t", "t", "t", "i", "i", "i", "i", "i")


def _launch(name, grid, block, args, shared=0, kinds=None):
    drv = _DRV or _driver()
    ctypes = drv["ctypes"]
    dev = torch.cuda.current_device()
    fn = _LAUNCH.get((dev, name))
    if fn is None:
        fn = _LAUNCH[(dev, name)] = _get(dev)[name].func
    # Key by the full argument-kind signature: a slot built for (tensor, int, ...) must never be
    # reused for a call whose kinds differ (an int written into a pointer holder, or a pointer
    # into a 4-byte c_int holder, truncates / over-reads and faulted intermittently).
    # Internal callers pass their constant signature; anything else is derived from the args.
    if kinds is None:
        kinds = _kinds(args)
    key = (dev, name, kinds)
    # Slot holders are mutated in place, so a slot is only safe for one launch at a time.
    with _slot_lock:
        slot = drv["slots"].get(key)
        if slot is None:
            slot = drv["slots"][key] = _holders(ctypes, kinds)
        holders = slot[0]
        for h, a in zip(holders, args):
            h.value = a.data_ptr() if isinstance(a, torch.Tensor) else a
        rc = drv["launch"](fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared,
                           drv["stream"](dev), slot[1], None)
    if rc != 0:
        raise RuntimeError(f"cuLaunchKernel({name}) failed: {rc}")


_TAG = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}


class GemvLaunch:
    """A GEMV launch for one layer with its argument block built once.

    Everything but the activation pointer, the output pointer and M is fixed per (layer, input
    dtype, output dtype), so a call writes three ctypes values and calls cuLaunchKernel: no
    kind signature, no slot lookup, no per-argument data_ptr()/isinstance, no grid arithmetic.
    The static tensors are referenced here, so their storage cannot be freed under the launch;
    FP8Linear rebuilds the object if any of them is replaced (see FP8Linear._gemv_launch).

    One object per (in dtype, out dtype) but one kernel per row width: the symbol is chosen from M
    on each call, so a layer never has to be rebuilt when the batch size changes.
    """
    __slots__ = ("fn_by_m", "dev", "device", "grid", "holders", "arr", "q", "ws", "bias", "lut",
                 "out_dtype", "N")

    def __init__(self, x_dtype, qweight, lut16, wscale, bias, out_dtype, group_size, fmt):
        drv = _DRV or _driver()
        self.dev = qweight.get_device()
        self.device = qweight.device
        xt, ot = _TAG[x_dtype], _TAG[out_dtype]
        fns = {}
        for mt in GEMV_MT:
            name = f"tfp8_gemv_in_{xt}_m{mt}_{ot}"
            fn = _LAUNCH.get((self.dev, name))
            if fn is None:
                fn = _LAUNCH[(self.dev, name)] = _get(self.dev)[name].func
            fns[mt] = fn
        # Indexed by M so the call path is one list lookup: at M=1 this layer's own work is ~17 us
        # and the surrounding Python is a large fraction of it.
        self.fn_by_m = [fns[next(w for w in GEMV_MT if w >= m)] for m in range(GEMV_MAXM + 1)]
        N, K = qweight.shape
        self.N = N
        self.grid = (N + GEMV_WARPS * GEMV_ROWS - 1) // (GEMV_WARPS * GEMV_ROWS)
        self.holders, self.arr = _holders(drv["ctypes"], K_W8A16)
        b = bias if bias is not None else wscale
        for h, a in zip(self.holders, [None, qweight, lut16, wscale, wscale, b, None, 0, N, K,
                                       group_size, int(bias is not None), FMT_ID.get(fmt, 2)]):
            h.value = a.data_ptr() if isinstance(a, torch.Tensor) else a
        self.q, self.ws, self.bias, self.lut, self.out_dtype = qweight, wscale, bias, lut16, out_dtype

    def __call__(self, x, M, out_shape=None):
        """x: contiguous CUDA tensor of the bound dtype on this layer's device holding M rows of
        K (any rank: only its data pointer is used). out_shape: shape of y, default (M, N)."""
        y = torch.empty(out_shape or (M, self.N), dtype=self.out_dtype, device=self.device)
        h = self.holders
        drv = _DRV
        with _slot_lock:
            h[0].value = x.data_ptr()
            h[6].value = y.data_ptr()
            h[7].value = M
            rc = drv["launch"](self.fn_by_m[M], self.grid, 1, 1, 32 * GEMV_WARPS, 1, 1, 0,
                               drv["stream"](self.dev), self.arr, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(gemv) failed: {rc}")
        return y


def prepare_activations(x):
    """(x16 fp16 [M,K], xs f32 [M]) in one kernel launch. Power-of-two row scale, exact."""
    M, K = x.shape
    x16 = torch.empty(M, K, dtype=torch.float16, device=x.device)
    xs = torch.empty(M, dtype=torch.float32, device=x.device)
    block = 256 if K >= 256 else 64
    _launch(_PREP[x.dtype], (max(M, 1), 1, 1), (block, 1, 1), [x, x16, xs, M, K, 8.0], kinds=K_PREP)
    return x16, xs


def w8a16(x, qweight, lut16, wscale, bias, out_dtype, group_size, fmt="e4m3fn"):
    """y = x @ dequant(qweight).T + bias.

    M <= 8            fused GEMV reading x in its native dtype (no prep pass, FP32 accumulate)
    M > 8             fused WMMA GEMM on FP16 activations with a power-of-two row scale
    """
    M, K = x.shape
    N = qweight.shape[0]
    y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    if M == 0:
        return y
    has_bias = bias is not None
    b = bias if has_bias else wscale  # any valid pointer
    fmt_id = FMT_ID.get(fmt, 2)
    if M <= GEMV_MAXM:
        args = [x.contiguous(), qweight, lut16, wscale, wscale, b, y, M, N, K, group_size,
                int(has_bias), fmt_id]
        grids = (N + GEMV_WARPS * GEMV_ROWS - 1) // (GEMV_WARPS * GEMV_ROWS)
        _launch(gemv_name(_TAG[x.dtype], _TAG[out_dtype], M), (grids, 1, 1),
                (32 * GEMV_WARPS, 1, 1), args, kinds=K_W8A16)
        return y
    x16, xs = prepare_activations(x)
    _gemm_launch(x16, xs, qweight, lut16, wscale, b, y, M, N, K, group_size, has_bias, fmt_id, out_dtype)
    return y


_GEMM_SUFFIX = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}


def _gemm_launch(x16, xs, qweight, lut16, wscale, b, y, M, N, K, group_size, has_bias, fmt_id, out_dtype):
    # Every output dtype needs its own entry: a float32 y written by the bf16 kernel was garbage.
    suffix = _GEMM_SUFFIX[out_dtype]
    args = [x16, qweight, lut16, wscale, xs, b, y, M, N, K, group_size, int(has_bias), fmt_id]
    _launch(f"tfp8_gemm_{suffix}", ((N + BN - 1) // BN, (M + BM - 1) // BM, 1), (128, 1, 1), args,
            kinds=K_W8A16)


def gemm_prepared(x16, xs, qweight, lut16, wscale, bias, out_dtype, group_size, fmt="e4m3fn"):
    """GEMM on an already-prepared FP16 activation (tests / benchmarks)."""
    M, K = x16.shape
    N = qweight.shape[0]
    y = torch.empty(M, N, dtype=out_dtype, device=x16.device)
    has_bias = bias is not None
    _gemm_launch(x16, xs, qweight, lut16, wscale, bias if has_bias else wscale, y, M, N, K,
                 group_size, has_bias, FMT_ID.get(fmt, 2), out_dtype)
    return y


_EPI = {torch.float16: "tfp8_epilogue_f16", torch.bfloat16: "tfp8_epilogue_bf16",
        torch.float32: "tfp8_epilogue_f32"}


def epilogue_into(y, acc, xs, bias, n0=0):
    """y[:, n0:n0+n] = (acc * xs[:, None] + bias[n0:n0+n]).to(y.dtype) in one launch.

    acc: contiguous FP32 [M, n] (one cuBLAS N-slice); y: contiguous [M, N] output of any of
    fp16 / bf16 / fp32. Replaces mul_ + add_ + .to() over a full FP32 [M, N] buffer.
    """
    M, n = acc.shape
    if M == 0 or n == 0:
        return y
    ldy = y.shape[1]
    has_bias = bias is not None
    bview = bias[n0:n0 + n] if has_bias else xs
    yv = y[:, n0:] if n0 else y           # data_ptr of the column window; stride stays ldy
    vec = int(n % 4 == 0 and ldy % 4 == 0 and n0 % 4 == 0
              and yv.data_ptr() % (4 * y.element_size()) == 0 and acc.data_ptr() % 16 == 0
              and (not has_bias or bview.data_ptr() % 16 == 0))
    per = 4 if vec else 1
    gx = max(1, min((n + 256 * per - 1) // (256 * per), 65535))
    _launch(_EPI[y.dtype], (gx, min(M, 65535), 1), (256, 1, 1),
            [acc, xs, bview, yv, M, n, ldy, int(has_bias), vec], kinds=K_EPI)
    return y


def dequant_into(out16, qweight, lut16, wscale, group_size, fmt="e4m3fn"):
    """Decode + scale qweight [n,K] into the fp16 buffer out16 [n,K] (workspace slice).

    The kernel is 2D: blockIdx.y = output row, so the grid must carry N. K must be a multiple
    of 16 (one 128-bit chunk per thread).
    """
    N, K = qweight.shape
    nchunk = K // 16
    _launch("tfp8_dequant_f16", (max(1, (nchunk + 127) // 128), N, 1), (128, 1, 1),
            [qweight, lut16, wscale, out16, N, K, group_size, FMT_ID.get(fmt, 2)], kinds=K_DEQ)
    return out16
