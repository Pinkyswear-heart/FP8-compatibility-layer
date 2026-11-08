"""Numerical gates from the design doc (section 10.3 / 11). Run from the repo root: python -I tests/test_tfp8.py
(also pytest-compatible)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import tfp8
from tfp8 import _ext
from tfp8.formats import E4M3FN, E5M2, decode, encode, lut
from tfp8.linear import FP8Linear
from tfp8.quantize import activation_to_fp16

CUDA = torch.cuda.is_available() and _ext.available()


def _same(a, b):
    """Bitwise-equal float tensors, treating NaN == NaN and keeping the sign of zero."""
    a, b = a.float(), b.float()
    both_nan = torch.isnan(a) & torch.isnan(b)
    eq = (a == b) & (torch.signbit(a) == torch.signbit(b))
    return bool((both_nan | eq).all())


def test_lut_all_256_codes_match_torch():
    for f in (E4M3FN, E5M2):
        codes = torch.arange(256, dtype=torch.uint8)
        ref = codes.view(f.torch_dtype).float()
        assert _same(decode(codes, f), ref), f.name
        assert _same(lut(f).float(), ref), f.name


def test_special_values():
    assert decode(torch.tensor([0x00, 0x80], dtype=torch.uint8), E4M3FN).tolist() == [0.0, -0.0]
    assert torch.signbit(decode(torch.tensor([0x80], dtype=torch.uint8), E4M3FN)).item()
    assert decode(torch.tensor([0x7E], dtype=torch.uint8), E4M3FN).item() == 448.0
    assert decode(torch.tensor([0x01], dtype=torch.uint8), E4M3FN).item() == 2.0 ** -9   # min subnormal
    assert math.isnan(decode(torch.tensor([0x7F], dtype=torch.uint8), E4M3FN).item())
    assert decode(torch.tensor([0x7B], dtype=torch.uint8), E5M2).item() == 57344.0
    assert decode(torch.tensor([0x7C], dtype=torch.uint8), E5M2).item() == math.inf
    assert decode(torch.tensor([0x01], dtype=torch.uint8), E5M2).item() == 2.0 ** -16


def test_encode_matches_torch_rne_and_saturates():
    torch.manual_seed(0)
    for f in (E4M3FN, E5M2):
        x = torch.cat([torch.randn(20000) * f.max_finite / 8, torch.randn(2000) * 2.0 ** -10])
        ref = x.to(f.torch_dtype).view(torch.uint8)
        assert torch.equal(encode(x, f), ref), f.name
        big = encode(torch.tensor([1e9, -1e9]), f)
        assert decode(big, f).tolist() == [f.max_finite, -f.max_finite]   # saturate, never inf


def test_activation_scaling_is_exact_power_of_two():
    x = (torch.randn(5, 256) * 1e3).bfloat16()
    x16, xs = activation_to_fp16(x)
    assert torch.all(torch.log2(xs) == torch.round(torch.log2(xs)))
    assert torch.equal((x16.float() * xs[:, None]), x.float())  # BF16 -> FP16 lossless here


def test_cpu_reference_path():
    lin = nn.Linear(256, 64)
    f = FP8Linear.from_linear(lin, group_size=128)
    x = torch.randn(3, 256)
    y = f(x)
    ref = x @ f.dequantized_weight().t() + lin.bias
    assert torch.allclose(y.float(), ref, atol=1e-2, rtol=1e-2)
    assert f.stats["torch"] == 1


def _check_kernel(M, N, K, G, out_dtype, backend, fmt="e4m3fn", bias=True):
    torch.manual_seed(M * 7 + N)
    lin = nn.Linear(K, N, bias=bias, device="cuda")
    f = FP8Linear.from_linear(lin, fmt=fmt, group_size=G, large_m_backend=backend)
    f.out_dtype = out_dtype
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    y = f(x).double()
    x16, xs = activation_to_fp16(x)
    ref = (x16.double() * xs.double()[:, None]) @ f.dequantized_weight().double().t()
    if bias:
        ref = ref + f.bias_f32.double()
    # Tolerance: FP32 accumulation error + one rounding of the output dtype.
    eps = 2.0 ** -11 if out_dtype == torch.float16 else 2.0 ** -8
    tol = eps * ref.abs() + 1e-4 * ref.abs().max() + 1e-6
    bad = ((y - ref).abs() > tol).sum().item()
    assert bad == 0, f"M={M} N={N} K={K} G={G} {out_dtype} {backend}: {bad} mismatches"


def test_kernel_switch_points_and_tails():
    if not CUDA:
        return
    for M in (1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 127, 128, 255, 256, 300):
        _check_kernel(M, 200, 512, None, torch.float16, "fused")   # N=200: tail tile
    for M in (256, 300):
        _check_kernel(M, 200, 512, None, torch.float16, "chunked")


def test_kernel_ragged_k_tail():
    """K only has to be a multiple of 16, not of the kernel's K-block (512 for GEMV, 32 for
    GEMM). A tail block must not read past the end of a weight row."""
    if not CUDA:
        return
    for K in (528, 1040, 1152, 1168, 1184):
        for M in (1, 8, 16, 64):
            _check_kernel(M, 200, K, None, torch.float16, "fused")


def test_kernel_group_scales_formats_dtypes():
    if not CUDA:
        return
    for G in (None, 32, 128):
        for fmt in ("e4m3fn", "e5m2"):
            for od in (torch.float16, torch.bfloat16, torch.float32):
                for M in (1, 40):
                    _check_kernel(M, 128, 1024, G, od, "fused", fmt, bias=(G != 32))


def test_group_scale_applied_per_block():
    """Two K-groups with wildly different scales: a single global scale would be wrong."""
    if not CUDA:
        return
    K, N = 256, 64
    lin = nn.Linear(K, N, bias=False, device="cuda")
    with torch.no_grad():
        lin.weight[:, :128] *= 1000.0
    f = FP8Linear.from_linear(lin, group_size=128)
    f.out_dtype = torch.float16
    x = torch.randn(33, K, device="cuda") * 1e-3
    ref = x.double() @ f.dequantized_weight().double().t()
    y = f(x).double()
    assert ((y - ref).abs() / (ref.abs().max())).max() < 2e-3


def test_chunked_formats_dtypes_and_slices():
    """The chunked backend must honour the layer's fmt (it used to always decode e4m3fn), and its
    fused epilogue must handle multi-slice N (column windows), odd N (scalar path), every
    output dtype and with/without bias."""
    if not CUDA:
        return
    for fmt in ("e4m3fn", "e5m2"):
        for od in (torch.float16, torch.bfloat16, torch.float32):
            for G in (None, 128):
                _check_kernel(260, 192, 512, G, od, "chunked", fmt, bias=(G is None))
    for N, bias in ((200, True), (330, False), (77, True)):
        torch.manual_seed(N)
        lin = nn.Linear(1024, N, bias=bias, device="cuda")
        f = FP8Linear.from_linear(lin, large_m_backend="chunked")
        f.out_dtype = torch.float16
        f.workspace_mb = 0                  # 64-row slices -> several column windows + ragged last one
        x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
        x16, xs = activation_to_fp16(x)
        ref = (x16.double() * xs.double()[:, None]) @ f.dequantized_weight().double().t()
        if bias:
            ref = ref + f.bias_f32.double()
        y = f(x).double()
        tol = 2.0 ** -11 * ref.abs() + 1e-4 * ref.abs().max() + 1e-6
        assert ((y - ref).abs() <= tol).all(), f"N={N} multi-slice chunked mismatch"
        assert f.stats["chunked"] == 1


def test_gemm_pipeline_edges():
    """The double-buffered WMMA GEMM: single K step (no prefetch), two/three steps (stage
    ping-pong), a scale group ending on every step (G=32), M/N tails."""
    if not CUDA:
        return
    for K in (32, 64, 96, 160):
        for G in (None, 32):
            for M in (9, 64, 65, 130):
                _check_kernel(M, 72, K, G, torch.float16, "fused", bias=(G is None))


def test_gemm_folded_group_scales():
    """The fused GEMM folds ws[n,g] / max_g ws[n,:] into the decoded weight tile.
    Power-of-two scales (the quantizer's default): the fold is exact, so per-element error must
    stay at FP32-accumulation level even with groups 2^20 below the row max.
    Non-power-of-two scales (checkpoint mode): each folded weight is rounded once to fp16
    (rel 2^-12, the same weight precision the chunked dequant path has), so the bound is
    relative to the row's largest output."""
    if not CUDA:
        return
    from tfp8.formats import get_format
    torch.manual_seed(5)
    N, K, G = 96, 1024, 128
    for fmt in ("e4m3fn", "e5m2"):
        for pow2 in (True, False):
            codes = torch.randint(0, 256, (N, K), dtype=torch.uint8, device="cuda")
            v = codes.view(get_format(fmt).torch_dtype).float()
            codes[~torch.isfinite(v)] = 0x10            # no NaN/inf codes in this test
            ws = torch.exp2(torch.randint(-20, 1, (N, K // G), device="cuda").float()) * 1e-2
            if not pow2:
                ws = ws * (1 + torch.rand(N, K // G, device="cuda"))
            f = FP8Linear.from_fp8_checkpoint(codes, ws, bias=torch.randn(N, device="cuda"), fmt=fmt)
            f.out_dtype = torch.float32
            for M in (16, 100):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                x16, xs = activation_to_fp16(x)
                ref = ((x16.double() * xs.double()[:, None]) @ f.dequantized_weight().double().t()
                       + f.bias_f32.double())
                d = (f(x).double() - ref).abs()
                err = (d / ref.abs().max()).max().item()
                # pow2: FP32 accumulation only (K=1024 -> ~1e-6); non-pow2: + fp16 weight rounding
                assert err < (1e-5 if pow2 else 5e-4), f"{fmt} pow2={pow2} M={M}: err/max {err}"
                assert f.stats["gemm"] >= 1


def test_prep_row_amax_is_block_wide():
    """The prep kernel's row scale must come from the max over the whole row, not one thread's
    slice: an outlier anywhere in the row must not overflow fp16."""
    if not CUDA:
        return
    x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    x[0, 1] = 3e4       # thread 1's slice
    x[1, 4095] = -5e4   # last thread
    x[2, 300] = 1e5     # beyond fp16 range on its own
    x16, xs = _ext.prepare_activations(x)
    rx16, rxs = activation_to_fp16(x)
    assert torch.isfinite(x16).all()
    assert torch.equal(xs, rxs) and torch.equal(x16, rx16)


def test_gemv_prepared_launch_tracks_layer_state():
    """FP8Linear's cached GEMV launch (fixed argument block) must give the same bits as the
    general path, and must never keep stale pointers: replaced buffers, a CPU round trip,
    a changed out_dtype, load_state_dict and N-d inputs all have to stay correct."""
    if not CUDA:
        return
    torch.manual_seed(11)
    lin = nn.Linear(512, 200, device="cuda")
    f = FP8Linear.from_linear(lin, group_size=128)
    f.nan_check = False

    def general(x):
        return _ext.w8a16(x.reshape(-1, 512), f.qweight, f._lut_for(), f.weight_scale, f.bias_f32,
                          f.out_dtype, f.group_size, f.fmt)

    W = f.dequantized_weight().double()
    for xd in (torch.float16, torch.bfloat16, torch.float32):
        for M in (1, 3, 8):
            x = torch.randn(M, 512, device="cuda", dtype=xd)
            y = f(x)
            assert torch.equal(y, general(x)), (xd, M)
            # every GEMV input dtype against FP64 (fp32 input used to stage only half of x)
            ref = x.double() @ W.t() + f.bias_f32.double()
            assert ((y.double() - ref).abs() <= 2.0 ** -7 * ref.abs() + 1e-4 * ref.abs().max()).all(), (xd, M)
    assert len(f._gemv) == 3
    x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16)        # N-d input, M=6
    assert torch.equal(f(x), general(x).reshape(2, 3, 200))
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    f.out_dtype = torch.float16                                           # new key
    assert f(x).dtype == torch.float16 and torch.equal(f(x), general(x))
    other = FP8Linear.from_linear(nn.Linear(512, 200, device="cuda"), group_size=128)
    f.qweight = other.qweight.clone()                                     # replaced buffer object
    f.weight_scale = other.weight_scale.clone()
    assert torch.equal(f(x), general(x))
    f.load_state_dict(other.state_dict())                                 # in-place copy_
    assert torch.equal(f(x), general(x))
    f.cpu()
    assert f(x.cpu()).device.type == "cpu"                                # torch path on CPU
    f.cuda()                                                              # all buffers replaced
    y = f(x)
    assert torch.equal(y, general(x)) and f.stats["gemv"] >= 12


def test_empty_and_noncontiguous():
    if not CUDA:
        return
    f = FP8Linear.from_linear(nn.Linear(512, 128, device="cuda"))
    assert f(torch.empty(0, 512, device="cuda")).shape == (0, 128)
    x = torch.randn(512, 20, device="cuda").t()       # non-contiguous view
    assert torch.allclose(f(x).float(), f(x.contiguous()).float())
    assert f(torch.randn(2, 3, 512, device="cuda")).shape == (2, 3, 128)


def test_planner_policy_and_tied_weights():
    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj, s.o_proj = nn.Linear(1024, 1024), nn.Linear(1024, 1024)
            s.gate_proj = nn.Linear(1024, 2048)
            s.input_layernorm = nn.LayerNorm(1024)
            s.lora_A = nn.Linear(1024, 8)
            s.router = nn.Linear(1024, 1024)

    class LM(nn.Module):
        def __init__(s):
            super().__init__()
            s.embed = nn.Embedding(100, 1024)
            s.layers = nn.ModuleList([Block()])
            s.lm_head = nn.Linear(1024, 100, bias=False)
            s.lm_head.weight = s.embed.weight

    m = tfp8.prepare_model(LM(), "llm", tfp8.FP8Config(profile="llm_safe"))
    owners = {d.name: d.owner for d in m._tfp8_plan}
    assert owners["layers.0.q_proj"] == "tfp8" and owners["layers.0.gate_proj"] == "tfp8"
    assert owners["layers.0.lora_A"] != "tfp8" and owners["layers.0.router"] != "tfp8"
    assert owners["lm_head"] != "tfp8"
    assert m.lm_head.weight is m.embed.weight            # tie preserved, no second copy
    assert isinstance(m.layers[0].q_proj, FP8Linear)
    # converting twice must not re-quantize (single precision_owner)
    again = tfp8.prepare_model(m, "llm", tfp8.FP8Config())
    assert {d.name: d.owner for d in again._tfp8_plan}["layers.0.q_proj"] == "tfp8"
    assert isinstance(again.layers[0].q_proj, FP8Linear)


def test_checkpoint_compat_mode_keeps_codes():
    q = torch.randint(0, 256, (64, 128), dtype=torch.uint8)
    q[E4M3FN.is_nan(q.int())] = 0
    s = torch.rand(64) + 0.5
    f = FP8Linear.from_fp8_checkpoint(q.view(torch.float8_e4m3fn), s, fmt="e4m3fn")
    assert torch.equal(f.qweight, q) and torch.equal(f.weight_scale[:, 0], s)
    try:
        FP8Linear.from_fp8_checkpoint(q.view(torch.float8_e4m3fn), s, fmt="e5m2")
        raise AssertionError("format mismatch must be rejected")
    except ValueError:
        pass


def test_launch_slot_kinds_are_isolated():
    """Regression for the cached-argument-slot bug class (A1).

    The launcher caches its ctypes argument holders; the cache must be keyed by the full
    argument-kind signature, or a slot built for (tensor, int, float) gets reused for
    (tensor, float, int) and silently passes the wrong argument types to the kernel.
    Uses a fake cuLaunchKernel so it needs no real kernel binary.
    """
    if not CUDA:
        return
    import ctypes
    dev = torch.cuda.current_device()
    name = "__tfp8_slot_probe__"
    calls = []

    def fake_launch(*a):
        calls.append(a)
        return 0

    orig_drv = _ext._DRV
    fake = {"ctypes": ctypes, "launch": fake_launch, "stream": lambda d: 0, "slots": {}}
    _ext._DRV = fake
    _ext._LAUNCH[(dev, name)] = ctypes.c_void_p(0x1234)
    try:
        t = torch.empty(4, 4, dtype=torch.float16, device="cuda")
        _ext._launch(name, (1, 1, 1), (32, 1, 1), [t, 1, 2.5])
        _ext._launch(name, (1, 1, 1), (32, 1, 1), [t, 1.0, 2])     # different kinds, same count
        _ext._launch(name, (1, 1, 1), (32, 1, 1), [t, 7, 8.0])     # back to the first signature
        assert len(fake["slots"]) == 2, f"expected 2 distinct slots, got {len(fake['slots'])}"
        (k1, s1), (k2, s2) = list(fake["slots"].items())
        assert k1[2] == ("t", "i", "f") and k2[2] == ("t", "f", "i"), (k1, k2)
        assert [type(h) for h in s1[0]] == [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
        assert [type(h) for h in s2[0]] == [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
        assert s1[0][0].value == t.data_ptr()
        assert s1[0][1].value == 7 and s1[0][2].value == 8.0   # slot reused and updated in place
        assert calls[0][9] is s1[1] and calls[1][9] is s2[1] and calls[2][9] is s1[1]
    finally:
        _ext._DRV = orig_drv
        _ext._LAUNCH.pop((dev, name), None)


def test_nan_fallback():
    if not CUDA:
        return
    f = FP8Linear.from_linear(nn.Linear(512, 128, device="cuda"))
    f.out_dtype = torch.float16
    f.weight_scale.fill_(1e30)                  # force fp16 overflow -> inf output
    f(torch.randn(4, 512, device="cuda"))
    assert f.fallback is not None and "non-finite" in f.fallback_reason


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print("CUDA kernels tested" if CUDA else "CUDA unavailable: kernel tests skipped")
    sys.exit(1 if fails else 0)
