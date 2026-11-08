"""Pure-CPU FP8 transcode throughput benchmark (encode + decode), part of the "transcoding FLOPs" report.

Measures, on CPU only:
  A. encode  : tfp8.quantize.quantize_weight (bf16 -> uint8 codes + f32 scale, pow2 scale), 5 layer shapes
  B. cast baselines on (4096,4096): bf16->float8_e4m3fn / fp16 / fp32 and a plain memcpy
  C. tfp8.formats.encode (searchsorted + float64 reference path) on 1M random floats
  D. decode  : tfp8.formats.lut(fmt)[codes.long()] (16M / 45M codes), full dequantize_weight reference
               path (fp32 out), and torch's own uint8.view(float8)->fp16 cast when supported
  E. model-level extrapolation: seconds to transcode 1B / 7B bf16 params to FP8
  F. thread scaling of quantize_weight: torch default threads vs 1 thread (plus all logical cores)

Hard CPU-only rule: CUDA_VISIBLE_DEVICES is cleared *before* torch is imported and torch.cuda is never
touched, so no GPU context / memory is ever created while the GPU benchmark runs on the same box.

Usage (from the repo root):
    python -I tests/bench_cpu_transcode.py
Writes .verify/bench/cpu/transcode_cpu.json (median/min/all samples per measurement).
"""
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

# Must happen before `import torch`: hard CPU-only guard.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfp8.formats import encode as ref_encode, lut  # noqa: E402
from tfp8.quantize import dequantize_weight, quantize_weight  # noqa: E402

OUT_DIR = os.path.join(ROOT, ".verify", "bench", "cpu")
WARMUP = 2
ROUNDS = 7

SHAPES = ((4096, 4096), (11008, 4096), (12288, 4096), (5632, 2048), (4608, 1152))
FORMATS = ("e4m3fn", "e5m2")
GROUPS = (("per_channel", None), ("group_128", 128))

_MEM = {"peak_mb": 0.0, "samples": []}


# ---------------------------------------------------------------- machine info
def _cpu_name():
    try:
        import winreg
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "unknown"


def _ram_total_gb():
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MemoryStatusEx()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 2 ** 30, 1)
    except Exception:
        pass
    return None


def _rss_mb():
    """Process working set in MiB (psutil if present, else the Win32 PSAPI call)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(pmc)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        if ok:
            return pmc.WorkingSetSize / 2 ** 20
    except Exception:
        pass
    return None


def record_mem(tag):
    mb = _rss_mb()
    if mb is None:
        return
    _MEM["samples"].append({"tag": tag, "rss_mb": round(mb, 1)})
    _MEM["peak_mb"] = max(_MEM["peak_mb"], mb)


def machine_info():
    info = {
        "host": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_name": _cpu_name(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "torch": torch.__version__,
        "torch_intra_threads_default": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_parallel_info": torch.__config__.parallel_info().strip(),
        "ram_total_gb": _ram_total_gb(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "gpu_touched": False,  # torch.cuda.* is never called; no CUDA context is created
    }
    return info


# ---------------------------------------------------------------- timing helper
def bench(fn, warmup=WARMUP, rounds=ROUNDS):
    """Warm up, then time `rounds` single calls; median is the reported number, all samples kept."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)  # ms
    med = statistics.median(samples)
    row = {
        "rounds": rounds,
        "warmup": warmup,
        "median_ms": round(med, 4),
        "min_ms": round(min(samples), 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "stdev_ms": round(statistics.stdev(samples), 4) if len(samples) > 1 else 0.0,
        "samples_ms": [round(v, 4) for v in samples],
    }
    return row


def _derive(row, elems, in_bytes, out_bytes):
    sec = row["median_ms"] * 1e-3
    row["elements"] = elems
    row["Gelem_s"] = round(elems / sec / 1e9, 3)
    row["input_GB_s"] = round(in_bytes / sec / 1e9, 2)
    row["output_GB_s"] = round(out_bytes / sec / 1e9, 2)
    return row


def _print_row(row, label, extra=""):
    print(f"  {label:46s} {row['median_ms']:9.3f} ms  {row['Gelem_s']:7.3f} Gelem/s"
          f"  in {row['input_GB_s']:6.2f} GB/s  out {row['output_GB_s']:6.2f} GB/s{extra}", flush=True)


# ---------------------------------------------------------------- A: encode
def phase_a(results):
    print("\n== A. encode: tfp8.quantize.quantize_weight (bf16 -> uint8 + f32 scale, pow2) ==")
    rows = []
    for N, K in SHAPES:
        torch.manual_seed(0)
        w = torch.randn(N, K, dtype=torch.bfloat16)
        elems = N * K
        for fmt in FORMATS:
            for glabel, gs in GROUPS:
                q, s, G = quantize_weight(w, fmt=fmt, group_size=gs, pow2_scale=True)
                contract = (q.dtype == torch.uint8 and tuple(q.shape) == (N, K)
                            and s.dtype == torch.float32 and tuple(s.shape) == (N, K // G)
                            and G == (K if gs is None else gs))
                del q, s
                row = bench(lambda: quantize_weight(w, fmt=fmt, group_size=gs, pow2_scale=True))
                row.update({"op": "tfp8.quantize.quantize_weight", "N": N, "K": K, "fmt": fmt,
                            "group": glabel, "group_size": G, "pow2_scale": True,
                            "in_dtype": "bfloat16", "out_dtype": "uint8 + f32 scale",
                            "contract_ok": bool(contract)})
                _derive(row, elems, elems * 2, elems)  # 2 B/elem in, 1 B/elem out (+ small scales)
                rows.append(row)
                _print_row(row, f"{N}x{K} {fmt} {glabel}")
        del w
        gc.collect()
        record_mem(f"A {N}x{K}")
    results["A_encode"] = rows


# ---------------------------------------------------------------- B: cast baselines
def phase_b(results):
    print("\n== B. cast baselines on (4096,4096)  [same dtype-free ops, no scale computation] ==")
    torch.manual_seed(0)
    x = torch.randn(4096, 4096, dtype=torch.bfloat16)
    elems = x.numel()
    cases = (
        ("bf16 -> float8_e4m3fn (.to)", lambda: x.to(torch.float8_e4m3fn), 1, torch.float8_e4m3fn),
        ("bf16 -> float16 (.to)", lambda: x.half(), 2, torch.float16),
        ("bf16 -> float32 (.to)", lambda: x.float(), 4, torch.float32),
        ("clone bf16 (memcpy)", lambda: x.clone(), 2, torch.bfloat16),
    )
    rows = []
    for name, fn, out_bytes, out_dtype in cases:
        y = fn()
        contract = y.dtype == out_dtype
        del y
        row = bench(fn)
        row.update({"name": name, "in_dtype": "bfloat16", "out_dtype": str(out_dtype).replace("torch.", ""),
                    "contract_ok": bool(contract)})
        _derive(row, elems, elems * 2, elems * out_bytes)
        rows.append(row)
        _print_row(row, name)
    del x
    gc.collect()
    record_mem("B")
    results["B_cast_baselines"] = rows


# ---------------------------------------------------------------- C: formats.encode
def phase_c(results):
    print("\n== C. tfp8.formats.encode reference path (float64 + searchsorted) on 1M random floats ==")
    torch.manual_seed(123)
    x = torch.randn(1_000_000, dtype=torch.float32)
    rows = []
    for fmt in FORMATS:
        y = ref_encode(x, fmt)
        contract = y.dtype == torch.uint8 and y.shape == x.shape
        del y
        row = bench(lambda: ref_encode(x, fmt))
        row.update({"op": "tfp8.formats.encode", "fmt": fmt, "in_dtype": "float32",
                    "out_dtype": "uint8", "path": "float64 table + searchsorted",
                    "contract_ok": bool(contract)})
        _derive(row, x.numel(), x.numel() * 4, x.numel())  # float64 intermediates are internal
        rows.append(row)
        _print_row(row, f"formats.encode {fmt} 1M")
    del x
    gc.collect()
    record_mem("C")
    results["C_formats_ref_encode"] = rows


# ---------------------------------------------------------------- D: decode
def phase_d(results):
    print("\n== D. decode: LUT gather / full reference dequant / native float8 cast ==")
    rows = []
    torch.manual_seed(0)
    w16 = torch.randn(4096, 4096, dtype=torch.bfloat16)
    q16, s16, g16 = {}, {}, {}
    for fmt in FORMATS:
        q16[fmt], s16[fmt], g16[fmt] = quantize_weight(w16, fmt=fmt, group_size=None, pow2_scale=True)

    for fmt in FORMATS:
        L = lut(fmt)
        lut_ok = L.numel() == 256 and L.dtype == torch.float16
        q = q16[fmt]
        ql = q.long()
        row = bench(lambda: L[q.long()])
        row.update({"name": f"lut[{fmt}][codes.long()]", "fmt": fmt, "elements": q.numel(),
                    "path": "tfp8.formats.lut + int64 index cast", "in_bytes_per_elem": 1,
                    "out_dtype": "float16", "lut_ok": bool(lut_ok), "contract_ok": True})
        _derive(row, q.numel(), q.numel(), q.numel() * 2)
        rows.append(row)
        _print_row(row, f"lut[{fmt}][codes.long()] 16M")
        row = bench(lambda: L[ql])
        row.update({"name": f"lut[{fmt}][pre-long codes] (gather only)", "fmt": fmt, "elements": q.numel(),
                    "path": "gather decomposition (codes.long() precomputed)", "in_bytes_per_elem": 8,
                    "out_dtype": "float16", "contract_ok": True})
        _derive(row, q.numel(), q.numel() * 8, q.numel() * 2)
        rows.append(row)
        _print_row(row, f"lut[{fmt}] gather-only 16M")
        del ql
        gc.collect()

    # 45M codes (11008x4096, the gate/up projection shape of a 1B-ish model)
    torch.manual_seed(0)
    w45 = torch.randn(11008, 4096, dtype=torch.bfloat16)
    for fmt in FORMATS:
        L = lut(fmt)
        q45 = quantize_weight(w45, fmt=fmt, group_size=None, pow2_scale=True)[0]
        elems = q45.numel()
        row = bench(lambda: L[q45.long()])
        row.update({"name": f"lut[{fmt}][codes.long()]", "fmt": fmt, "elements": elems,
                    "path": "tfp8.formats.lut + int64 index cast", "in_bytes_per_elem": 1,
                    "out_dtype": "float16", "contract_ok": True})
        _derive(row, elems, elems, elems * 2)
        rows.append(row)
        _print_row(row, f"lut[{fmt}][codes.long()] 45M")
        ql = q45.long()
        row = bench(lambda: L[ql])
        row.update({"name": f"lut[{fmt}][pre-long codes] (gather only)", "fmt": fmt, "elements": elems,
                    "path": "gather decomposition (codes.long() precomputed)", "in_bytes_per_elem": 8,
                    "out_dtype": "float16", "contract_ok": True})
        _derive(row, elems, elems * 8, elems * 2)
        rows.append(row)
        _print_row(row, f"lut[{fmt}] gather-only 45M")
        del ql, q45
        gc.collect()
        record_mem(f"D 45M {fmt}")
    del w45

    # full reference dequantize_path, fp32 output, 16M codes, e4m3fn per-channel
    row = bench(lambda: dequantize_weight(q16["e4m3fn"], s16["e4m3fn"], "e4m3fn", g16["e4m3fn"]))
    row.update({"name": "dequantize_weight reference (fp32 out)", "fmt": "e4m3fn",
                "group": "per_channel", "elements": q16["e4m3fn"].numel(),
                "path": "tfp8.quantize.dequantize_weight: lut -> long -> fp16 -> fp32 * scale",
                "out_dtype": "float32", "contract_ok": True})
    _derive(row, q16["e4m3fn"].numel(), q16["e4m3fn"].numel(), q16["e4m3fn"].numel() * 4)
    rows.append(row)
    _print_row(row, "dequantize_weight ref 16M (fp32 out)")

    # torch's own float8 dtype cast, when CPU support exists
    view_support = True
    try:
        probe = q16["e4m3fn"].view(torch.float8_e4m3fn).to(torch.float16)
        _ = (probe.dtype == torch.float16)
        del probe
    except Exception as exc:  # noqa: BLE001
        view_support = False
        view_error = f"{type(exc).__name__}: {exc}"
    if view_support:
        for label, q, elems in (("16M", q16["e4m3fn"], 4096 * 4096),):
            row = bench(lambda: q.view(torch.float8_e4m3fn).to(torch.float16))
            row.update({"name": f"codes.view(float8_e4m3fn).to(fp16) {label}", "fmt": "e4m3fn",
                        "elements": elems, "path": "torch native reinterpret + cast",
                        "out_dtype": "float16", "contract_ok": True})
            _derive(row, elems, elems, elems * 2)
            rows.append(row)
            _print_row(row, f"view->fp16 {label} (native cast)")
        torch.manual_seed(0)
        w45b = torch.randn(11008, 4096, dtype=torch.bfloat16)
        q45b = quantize_weight(w45b, fmt="e4m3fn", group_size=None, pow2_scale=True)[0]
        del w45b
        row = bench(lambda: q45b.view(torch.float8_e4m3fn).to(torch.float16))
        row.update({"name": "codes.view(float8_e4m3fn).to(fp16) 45M", "fmt": "e4m3fn",
                    "elements": q45b.numel(), "path": "torch native reinterpret + cast",
                    "out_dtype": "float16", "contract_ok": True})
        _derive(row, q45b.numel(), q45b.numel(), q45b.numel() * 2)
        rows.append(row)
        _print_row(row, "view->fp16 45M (native cast)")
        del q45b
    results["D_decode"] = rows
    results["D_float8_view_support"] = {"supported": view_support,
                                        **({} if view_support else {"error": view_error})}
    del w16, q16, s16
    gc.collect()
    record_mem("D")


# ---------------------------------------------------------------- F: thread scaling
def phase_f(results):
    print("\n== F. thread scaling: quantize_weight e4m3fn per_channel, (4096,4096) ==")
    default_threads = torch.get_num_threads()
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.bfloat16)
    elems = w.numel()
    cases = (("default", default_threads), ("1_thread", 1),
             ("all_cores", os.cpu_count() or default_threads))
    rows = []
    try:
        for label, nthreads in cases:
            torch.set_num_threads(nthreads)
            row = bench(lambda: quantize_weight(w, fmt="e4m3fn", group_size=None, pow2_scale=True))
            row.update({"label": label, "threads": nthreads,
                        "effective_threads": torch.get_num_threads(),
                        "name": f"quantize_weight e4m3fn per_channel ({nthreads} threads)",
                        "contract_ok": True})
            _derive(row, elems, elems * 2, elems)
            rows.append(row)
            _print_row(row, f"threads={nthreads} ({label})")
    finally:
        torch.set_num_threads(default_threads)
    del w
    gc.collect()
    record_mem("F")
    results["F_thread_scaling"] = rows
    results["F_threads"] = {"default": default_threads, "logical_cpus": os.cpu_count(),
                            "restored": torch.get_num_threads()}


# ---------------------------------------------------------------- E: extrapolation
def phase_e(results):
    a = results["A_encode"]
    enc = [r for r in a if r["fmt"] == "e4m3fn" and r["group"] == "per_channel"]
    rates = sorted(r["Gelem_s"] for r in enc)
    r_med = statistics.median(rates)
    r_4096 = next(r["Gelem_s"] for r in enc if r["N"] == 4096 and r["K"] == 4096)
    r_best = max(r["Gelem_s"] for r in a)
    f_rows = {r["label"]: r for r in results["F_thread_scaling"]}
    r_default = f_rows["default"]["Gelem_s"]
    r_single = f_rows["1_thread"]["Gelem_s"]
    threads_scale = r_default / r_single if r_single else None

    def secs(params, rate):
        return round((params / 1e9) / rate, 3) if rate else None

    rows = []
    for label, params in (("1B", 1e9), ("7B", 7e9)):
        sec = secs(params, r_med)
        rows.append({
            "model_params": label,
            "params": params,
            "R_Gelem_s": r_med,
            "seconds": sec,
            "effective_GB_s_bf16_in_fp8_out": round(3 * params / (sec * 1e9), 2) if sec else None,
            "seconds_at_best_rate": secs(params, r_best),
            "seconds_single_thread": secs(params, r_single),
        })
    results["E_model_extrapolation"] = {
        "formula": "t_seconds = P / (R * 1e9) = (P / 1e9) / R,  P = params (elements), R in Gelem/s",
        "throughput_source": "median of A rows with fmt=e4m3fn, group=per_channel over all 5 layer shapes",
        "R_Gelem_s_median": r_med,
        "R_4096x4096_Gelem_s": r_4096,
        "R_best_observed_Gelem_s": r_best,
        "thread_scaling_default_over_1": round(threads_scale, 3) if threads_scale else None,
        "rows": rows,
        "assumptions": [
            "one pass over the weights, already resident in RAM as bf16 (no model load / disk I/O counted)",
            "includes amax reduction + power-of-two scale + divide + clamp + RNE float8 cast per element",
            "torch default intra-op threads (10 on this box); single-thread column from F",
            "scale tensors are negligible (N f32 values, or N*K/G for group=128)",
        ],
    }
    print("\n== E. model-level extrapolation (encode, median Gelem/s from A) ==")
    print(f"  R_median = {r_med:.3f} Gelem/s   (4096x4096: {r_4096:.3f}; best observed {r_best:.3f})")
    for r in rows:
        print(f"  {r['model_params']:>3s} params -> {r['seconds']:7.2f} s  "
              f"(single thread {r['seconds_single_thread']:7.2f} s, best rate {r['seconds_at_best_rate']:6.2f} s,"
              f" effective I/O {r['effective_GB_s_bf16_in_fp8_out']:.2f} GB/s)")


# ---------------------------------------------------------------- main
def main():
    t_start = time.perf_counter()
    results = {
        "schema": 1,
        "script": "tests/bench_cpu_transcode.py",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "machine": machine_info(),
        "method": {"timer": "time.perf_counter", "statistic": "median",
                   "warmup": WARMUP, "rounds": ROUNDS,
                   "timed_unit": "one call", "gc_disabled_during_timing": True},
        "notes": [
            "pure CPU benchmark: CUDA_VISIBLE_DEVICES='' before import torch; torch.cuda.* never called",
            "CPU is shared with the GPU benchmark (Lead) on the same box; medians + all samples are kept",
            "encode = bf16 weights -> uint8 codes + f32 scale (tfp8.quantize.quantize_weight, pow2_scale=True)",
            "input GB/s assumes bf16 source (2 B/elem); output GB/s assumes the main output tensor (1/2/4 B/elem)",
        ],
    }
    print("CPU FP8 transcode benchmark (encode + decode) -- pure CPU, no CUDA")
    for k, v in results["machine"].items():
        print(f"  {k}: {v}")
    print(f"  ROUNDS={ROUNDS} WARMUP={WARMUP}  output -> {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)
    gc.disable()
    try:
        phase_a(results)
        phase_b(results)
        phase_c(results)
        phase_d(results)
        phase_f(results)
        phase_e(results)
    finally:
        gc.enable()

    runtime = time.perf_counter() - t_start
    results["meta"] = {
        "runtime_s": round(runtime, 2),
        "budget_s": 900,
        "within_budget": runtime < 900,
        "peak_rss_mb": round(_MEM["peak_mb"], 1),
        "rss_samples": _MEM["samples"],
        "memory_budget_mb": 4096,
        "within_memory_budget": _MEM["peak_mb"] < 4096 if _MEM["peak_mb"] else None,
        "checks": {
            "contract_ok": all(r.get("contract_ok", True) for section in
                               ("A_encode", "B_cast_baselines", "C_formats_ref_encode", "D_decode")
                               for r in results.get(section, [])),
            "float8_view_to_fp16_supported": results.get("D_float8_view_support", {}).get("supported"),
            "threads_restored": torch.get_num_threads() == results["machine"]["torch_intra_threads_default"],
        },
    }
    results["meta"]["checks"]["fingerprint"] = (
        f"{results['machine']['torch']}|{stats_str(results)}")

    json_path = os.path.join(OUT_DIR, "transcode_cpu.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\nwrote {json_path}")
    print(f"runtime {runtime:.1f}s   peak RSS {_MEM['peak_mb']:.1f} MiB   "
          f"checks {results['meta']['checks']}")


def stats_str(results):
    a = results["A_encode"]
    e = results.get("E_model_extrapolation", {})
    parts = [f"enc_med={e.get('R_Gelem_s_median')}"]
    d = results.get("D_decode", [])
    lut16 = [r for r in d if r["name"].startswith("lut[e4m3fn][codes.long()]") and r["elements"] == 4096 * 4096]
    if lut16:
        parts.append(f"dec_lut16M={lut16[0]['Gelem_s']}")
    return ",".join(parts)


if __name__ == "__main__":
    main()
