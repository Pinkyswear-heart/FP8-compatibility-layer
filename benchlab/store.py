"""Records and catalog for the benchmark page.

A record is one measurement of one GPU, stored as

    <root>/<gpu-slug>/<id>.json

with a stable top-level ``id`` so the file survives being moved or copied between machines.
The catalog is derived by scanning the tree, so a record someone else produced shows up
without a registration step -- copy the JSON in, refresh the page.

Nothing here imports torch: the page can be served on a machine with no GPU at all, which is
the point of comparing records collected elsewhere.
"""
from __future__ import annotations

import json
import os
import re
import time

SCHEMA = 1

# (key, label, unit, higher_is_better, group). The order is the reading order on the page.
# Only metrics every GPU can produce belong here -- an entry that is absent on one card would
# read as "infinitely better", so extraction returns None rather than a zero.
METRICS = [
    ("decode_gelem_s",   "FP8→FP16 解码",        "Gelem/s", True,  "transcode",
     "45.1M 权重，微基准，读 1 B 写 2 B"),
    ("encode_gelem_s",   "FP16→FP8 编码",        "Gelem/s", True,  "transcode",
     "45.1M 元素，微基准"),
    ("copy_gb_s",        "显存拷贝带宽",          "GB/s",    True,  "memory",
     "256 MB u8 拷贝，双程计"),
    ("cublas_f16_tf",    "cuBLAS FP16 4096³",   "TFLOPS",  True,  "compute",
     "cuBLAS 自带 kernel，用来区分「卡的算力」和「转译的开销」"),
    ("fused_tf",         "融合 GEMM M=2048",     "TFLOPS",  True,  "compute",
     "解码放进 WMMA 主循环"),
    ("chunked_tf",       "chunked GEMM M=2048",  "TFLOPS",  True,  "compute",
     "先反量化再交给 cuBLAS"),
    ("launch_floor_us",  "单次提交地板",          "µs",      False, "overhead",
     "空 kernel 一轮的 event 计时，含驱动的提交开销"),
]

METRIC_BY_KEY = {m[0]: m for m in METRICS}
GROUPS = [("transcode", "转译"), ("memory", "显存"), ("compute", "算力"), ("overhead", "启动")]


# --------------------------------------------------------------------------- extraction
def _dig(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def _num(v):
    """Harness JSON stores nan/inf as strings; anything non-finite is 'not measured'."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v == v and abs(v) != float("inf") else None
    return None


def extract_metrics(raw: dict) -> dict:
    """Pull the comparison metrics out of a tests/bench_transcode.py JSON.

    Every lookup is defensive: a record made with --only bw,decode has no gemm section, and a
    section that raised is stored as {"error": ...}. A missing metric is None, never 0.
    """
    s = raw.get("sections") or {}
    m = {}

    m["decode_gelem_s"] = _num(_dig(s, "decode", "micro", "dec_e4m3_f16_n45088768", "Gelem_s"))
    m["encode_gelem_s"] = _num(_dig(s, "encode", "micro", "enc_f16_e4m3_45M", "Gelem_s"))
    m["copy_gb_s"] = _num(_dig(s, "bw", "micro", "copy_u8_256MB", "GB_s"))
    m["cublas_f16_tf"] = _num(_dig(s, "bw", "cublas_f16_4096", "TFLOPS"))

    for row in _dig(s, "gemm", "rows", default=[]) or []:
        if row.get("M") == 2048:
            m["fused_tf"] = _num(row.get("fused_TFLOPS"))
            m["chunked_tf"] = _num(row.get("chunked_TFLOPS"))

    m["launch_floor_us"] = _num(_dig(s, "floor", "tfp8_gemv_128x128", "event_loop_us"))
    return m


def extract_extra(raw: dict) -> dict:
    """Shape-specific numbers the page shows in the detail table but does not rank across GPUs.

    GEMV throughput depends on the layer shape and on the Windows launch floor, so ranking two
    cards on it would mostly rank their drivers. Kept per shape, shown as-is.
    """
    s = raw.get("sections") or {}
    extra = {"gemv": [], "production_decode": []}

    for row in _dig(s, "gemv", "rows", default=[]) or []:
        if row.get("M") != 1:
            continue
        extra["gemv"].append({
            "shape": f"{row.get('N')}x{row.get('K')}",
            "weight_MB": _num(row.get("weight_MB")),
            "tfp8_us": _num(row.get("median_us")),
            "cublas_us": _num(row.get("fp16_cublas_us")),
        })

    for name, row in (_dig(s, "decode", "production", default={}) or {}).items():
        extra["production_decode"].append({
            "name": name,
            "Gelem_s": _num(row.get("Gelem_s")),
            "GB_s": _num(row.get("GB_s")),
        })
    return extra


# --------------------------------------------------------------------------- records
def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "gpu").lower()).strip("-")
    return s or "gpu"


def new_id(gpu_name: str, when: float | None = None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(when if when is not None else time.time()))
    return f"{slugify(gpu_name)}/{stamp}"


def record_from_raw(raw: dict, *, source: str, note: str = "", when: float | None = None) -> dict:
    """Wrap a harness JSON in a record. `raw` is kept so nothing is lost in the round trip."""
    env = raw.get("env") or {}
    name = env.get("gpu") or "unknown GPU"
    ts = when if when is not None else time.time()
    gpu = {
        "name": name,
        "sm": env.get("sm"),
        "sm_count": env.get("sm_count"),
        "vram_MB": env.get("vram_MB"),
        "l2_MB": env.get("l2_MB"),
        "driver": env.get("driver"),
        "torch": env.get("torch"),
        "cuda": env.get("cuda"),
        "python": env.get("python"),
        "cpu_count": env.get("cpu_count"),
    }
    return {
        "schema": SCHEMA,
        "id": new_id(name, ts),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "source": source,
        "note": note,
        "gpu": gpu,
        "metrics": extract_metrics(raw),
        "extra": extract_extra(raw),
        "raw": raw,
    }


def save(rec: dict, root: str) -> str:
    d = os.path.join(root, rec["id"].split("/")[0])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(root, *rec["id"].split("/")) + ".json"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load_all(root: str) -> list[dict]:
    """Every record under root, newest first. Bad files are skipped and reported, not fatal."""
    out, bad = [], []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirnames, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                if not isinstance(rec, dict) or "metrics" not in rec or "id" not in rec:
                    raise ValueError("not a benchlab record")
                rec["_path"] = path
                out.append(rec)
            except Exception as e:  # noqa: BLE001
                bad.append((path, repr(e)))
    out.sort(key=lambda r: r.get("when", ""), reverse=True)
    for p, e in bad:
        print(f"benchlab: skipped {p}: {e}")
    return out


def delete(rid: str, root: str) -> bool:
    path = os.path.join(root, *rid.split("/")) + ".json"
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def compare(records: list[dict], keys: list[str] | None = None) -> dict:
    """Ratio of every record against the first one, per metric.

    The baseline is the first record in `records` -- never the fastest -- so the numbers read as
    "this card vs that card" whatever order the page happens to hold them in.
    """
    keys = keys or [m[0] for m in METRICS]
    if not records:
        return {"baseline": None, "rows": []}
    base = records[0]
    rows = []
    for key in keys:
        label, unit, better, group, _hint = METRIC_BY_KEY[key][1:]
        b = base["metrics"].get(key)
        cells = []
        for rec in records:
            v = rec["metrics"].get(key)
            ratio = (v / b) if (v is not None and b) else None
            cells.append({"id": rec["id"], "value": v, "ratio": ratio})
        rows.append({"key": key, "label": label, "unit": unit, "higher_is_better": better,
                     "group": group, "baseline": b, "cells": cells})
    return {"baseline": base["id"], "rows": rows}
