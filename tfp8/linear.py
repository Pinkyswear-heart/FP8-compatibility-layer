"""FP8Linear: W8A16 Linear whose weight stays resident as FP8 (uint8 codes + f32 scales).

Execution paths, chosen per call by M = number of activation rows:
  M <= 8           fused GEMV  (decode FP8 on read, FP32 FMA)
  8 < M < large_m  fused WMMA GEMM (decode in tile, FP16 tensor core, FP32 accumulate)
  M >= large_m     'fused' as above, or 'chunked': decode N-slices into a fixed FP16
                   workspace and run cuBLAS. The workspace is bounded (default 32 MB) and
                   shared by all layers, so a full FP16 weight is never materialised.
  CPU / no NVRTC   PyTorch reference (dequantize + matmul)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _ext
from .formats import get_format, lut
from .quantize import activation_to_fp16, dequantize_weight, quantize_weight

_WORKSPACE = {}  # device -> fp16 buffer. Module-global and shared: not re-entrant / thread-safe,
# so do not run chunked forwards concurrently (multi-stream / multi-thread).


def _workspace(device, numel):
    buf = _WORKSPACE.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=torch.float16, device=device)
        _WORKSPACE[device] = buf
    return buf


class FP8Linear(nn.Module):
    precision_owner = "tfp8"

    def __init__(self, in_features, out_features, bias=True, fmt="e4m3fn", group_size=None,
                 out_dtype=torch.bfloat16, large_m=256, large_m_backend="chunked",
                 workspace_mb=32, nan_check=True, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.fmt = get_format(fmt).name
        self.group_size = group_size or in_features
        self.out_dtype = out_dtype
        self.large_m, self.large_m_backend = large_m, large_m_backend
        self.workspace_mb, self.nan_check = workspace_mb, nan_check
        ng = in_features // self.group_size
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.uint8, device=device))
        self.register_buffer("weight_scale", torch.ones(out_features, ng, dtype=torch.float32, device=device))
        self.register_buffer("bias_f32", torch.zeros(out_features, dtype=torch.float32, device=device) if bias else None)
        self._nan_checks = 0
        self.fallback = None        # nn.Linear set by the planner when the module had to back off
        self.stats = {"gemv": 0, "gemm": 0, "chunked": 0, "torch": 0, "fallback": 0}
        self._kernels_ok = self._probe_kernels()
        self._lut = lut(self.fmt, device) if device is not None else None   # resolved once, not per call
        self._gemv = {}             # (x dtype, out dtype) -> _ext.GemvLaunch (prepared argument block)

    def _lut_for(self):
        """The decode table for this layer's device. Cached, but re-resolved if the layer was
        moved (a layer built on CPU and later `.cuda()`d must not keep a CPU table -- passing it
        to the kernel would read the wrong address space)."""
        l = self._lut
        if l is None or l.device != self.weight_scale.device:
            l = self._lut = lut(self.fmt, self.weight_scale.device)
        return l

    def _probe_kernels(self):
        """Resolved once per layer instead of per call. Same conditions as the old _kernel_ok,
        minus the activation: K % 32, G % 32, CUDA and the NVRTC module must all be present."""
        K, G = self.in_features, self.group_size
        return (K % 32 == 0 and G % 32 == 0 and _ext.available())

    @classmethod
    def from_linear(cls, lin: nn.Linear, fmt="e4m3fn", group_size=None, pow2_scale=True, **kw):
        q, s, G = quantize_weight(lin.weight.data, fmt, group_size, pow2_scale)
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, fmt, G,
                out_dtype=lin.weight.dtype if lin.weight.dtype in (torch.float16, torch.bfloat16) else torch.float16,
                device=lin.weight.device, **kw)
        m.qweight.copy_(q)
        m.weight_scale.copy_(s)
        if lin.bias is not None:
            m.bias_f32.copy_(lin.bias.data.float())
        return m

    @classmethod
    def from_fp8_checkpoint(cls, qweight, weight_scale, bias=None, fmt="e4m3fn", **kw):
        """Compatibility mode: take an existing FP8 checkpoint's codes and scales verbatim."""
        if qweight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            want = get_format(fmt).torch_dtype
            if qweight.dtype != want:
                raise ValueError(f"checkpoint dtype {qweight.dtype} does not match declared fmt {fmt}")
            qweight = qweight.view(torch.uint8)
        N, K = qweight.shape
        ws = weight_scale.float().reshape(N, -1) if weight_scale.numel() > 1 else weight_scale.float().expand(N, 1)
        G = K // ws.shape[1]
        m = cls(K, N, bias is not None, fmt, G, device=qweight.device, **kw)
        m.qweight.copy_(qweight)
        m.weight_scale.copy_(ws)
        if bias is not None:
            m.bias_f32.copy_(bias.float())
        return m

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, fmt={self.fmt}, "
                f"group={self.group_size}, out_dtype={self.out_dtype}")

    def dequantized_weight(self):
        return dequantize_weight(self.qweight, self.weight_scale, self.fmt, self.group_size)

    def _chunked(self, x16, xs):
        """Large M: decode at most `workspace` elements of W at a time, cuBLAS FP16 GEMM per slice.

        Each slice's FP32 cuBLAS result goes straight through one fused epilogue kernel
        (row scale + bias + output cast) into its column window of y: no full-size FP32 buffer,
        no strided slice copy, no separate mul_/add_/.to() passes.
        """
        N, K = self.qweight.shape
        rows = max(64, min(N, (self.workspace_mb << 20) // 2 // K) // 64 * 64)
        ws = _workspace(x16.device, rows * K)
        y = torch.empty(x16.shape[0], N, dtype=self.out_dtype, device=x16.device)
        lut16 = self._lut_for()
        fmt = self.fmt
        for n0 in range(0, N, rows):
            n1 = min(N, n0 + rows)
            w16 = ws[: (n1 - n0) * K].view(n1 - n0, K)
            _ext.dequant_into(w16, self.qweight[n0:n1], lut16, self.weight_scale[n0:n1],
                              self.group_size, fmt)
            # FP16 inputs, FP32 accumulate, FP32 output: same numerics as the fused kernel.
            acc = torch.mm(x16, w16.t(), out_dtype=torch.float32)
            _ext.epilogue_into(y, acc, xs, self.bias_f32, n0)
        return y

    def _gemv_launch(self, x_dtype):
        """Prepared GEMV launch for this layer, or None when the fast path does not apply.

        Rebuilt whenever a buffer object was replaced (`.to()`, `.cuda()`, direct assignment) or
        the output dtype changed; in-place updates (`load_state_dict`, `copy_`) keep the storage
        and therefore the cached pointers valid. Checked with identity tests on `_buffers`, which
        are dict lookups -- module attribute access (`self.qweight`) goes through
        nn.Module.__getattr__ and costs ~0.8 us each.
        """
        b = self._buffers
        q, ws, bias = b["qweight"], b["weight_scale"], b.get("bias_f32")
        key = (x_dtype, self.out_dtype)
        p = self._gemv.get(key)
        if p is None or p.q is not q or p.ws is not ws or p.bias is not bias or p.lut is not self._lut:
            if x_dtype not in _ext._TAG or self.out_dtype not in _ext._TAG or not q.is_cuda:
                return None
            p = self._gemv[key] = _ext.GemvLaunch(x_dtype, q, self._lut_for(), ws, bias,
                                                   self.out_dtype, self.group_size, self.fmt)
        return p

    def forward(self, x):
        # Decode fast path (M <= GEMV_MAXM rows, contiguous CUDA input, past the first NaN-checked
        # calls): the GEMV kernel only needs x's data pointer and M, so an [B, T, K] input is
        # neither reshaped in nor out -- each torch reshape costs ~4-5 us on this path, about as
        # much as the rest of the Python around the launch. Output is allocated in its final shape.
        if ((self._nan_checks >= 3 or not self.nan_check) and self.fallback is None
                and self._kernels_ok and x.is_cuda and x.is_contiguous()):
            shape = x.shape
            if shape[-1] == self.in_features:
                M = x.numel() // shape[-1]
                if 0 < M <= _ext.GEMV_MAXM:
                    p = self._gemv_launch(x.dtype)
                    if p is not None and x.get_device() == p.dev and torch.cuda.current_device() == p.dev:
                        self.stats["gemv"] += 1
                        return p(x, M, shape[:-1] + (self.out_features,))
        if self.fallback is not None:
            self.stats["fallback"] += 1
            return self.fallback(x)
        shape = x.shape
        x2 = x if x.dim() == 2 else x.reshape(-1, shape[-1])   # reshape costs ~4 us even as a no-op
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        # Everything below is on the per-call path, and at M=1 the layer's own work is only
        # ~17 us, so each Python check that can be hoisted is worth more than kernel tuning:
        # `_kernel_ok` re-queried _ext.available() (NVRTC probe) and `lut()` re-looked-up the
        # table, together ~9 us of pure overhead per call.
        # x2.is_cuda must stay per-call: a layer created on CPU (or moved later) must not send
        # CPU pointers to a CUDA kernel. `is_cuda` is an attribute read, unlike the probe below.
        if not self._kernels_ok or not x2.is_cuda:
            self.stats["torch"] += 1
            y = F.linear(x2.float(), self.dequantized_weight(), self.bias_f32).to(self.out_dtype)
        else:
            if M >= self.large_m and self.large_m_backend == "chunked":
                self.stats["chunked"] += 1
                x16, xs = _ext.prepare_activations(x2)
                y = self._chunked(x16, xs)
            else:
                y = None
                if 0 < M <= _ext.GEMV_MAXM:
                    self.stats["gemv"] += 1
                    p = self._gemv_launch(x2.dtype)
                    # The prepared launch is bound to the layer's device; anything else (input on
                    # another GPU, a different current device) takes the general path.
                    if p is not None and x2.get_device() == p.dev and torch.cuda.current_device() == p.dev:
                        y = p(x2, M)
                else:
                    self.stats["gemv" if M == 0 else "gemm"] += 1
                if y is None:
                    y = _ext.w8a16(x2, self.qweight, self._lut_for(), self.weight_scale,
                                   self.bias_f32, self.out_dtype, self.group_size, self.fmt)
        # A per-call device sync costs ~0.3 ms and would dominate decode; check the first few
        # calls (failures are structural) rather than every token.
        if self.nan_check and self._nan_checks < 3:
            self._nan_checks += 1
            if not torch.isfinite(y).all() and torch.isfinite(x2).all():
                self._back_off(f"non-finite output for input shape {tuple(x2.shape)}")
                return self.forward(x)
        return y if x.dim() == 2 else y.reshape(*shape[:-1], self.out_features)

    def _back_off(self, reason):
        """Runtime fallback: rebuild an FP16 Linear from the FP8 codes (same numbers) and log it."""
        lin = nn.Linear(self.in_features, self.out_features, self.bias_f32 is not None,
                        device=self.qweight.device, dtype=torch.float32)
        lin.weight.data.copy_(self.dequantized_weight())
        if self.bias_f32 is not None:
            lin.bias.data.copy_(self.bias_f32)
        self.fallback = _CastLinear(lin, self.out_dtype)
        self.fallback_reason = reason


class _CastLinear(nn.Module):
    def __init__(self, lin, out_dtype):
        super().__init__()
        self.lin, self.out_dtype = lin, out_dtype

    def forward(self, x):
        return self.lin(x.float()).to(self.out_dtype)
