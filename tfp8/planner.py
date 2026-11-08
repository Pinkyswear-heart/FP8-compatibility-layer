"""Model precision planner: decides, per module, who owns it (tfp8 / tbf16 / fp16 / native).

Rules (from the design doc):
  * LLM: MLP gate/up/down and Q/K/V/O projections -> FP8 W8A16. Norms, softmax, embeddings,
    tied lm_head, routers, LoRA/adapters, small Linears -> keep.
  * Image: DiT/UNet large Linear/MLP and attention projections -> FP8. Text encoder -> tbf16,
    VAE, norms, time/adaLN embeddings and the final output layer -> FP16.
  * Every module gets exactly one precision_owner; a module already owned is never re-converted.
  * Parameters shared between modules (tied embedding/lm_head) are never quantized.
"""
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import FP8Config
from .linear import FP8Linear

_LLM_FP8 = re.compile(r"(q_proj|k_proj|v_proj|o_proj|qkv|out_proj|gate_proj|up_proj|down_proj|fc1|fc2|w1|w2|w3|mlp)")
_IMAGE_FP8 = re.compile(r"(to_q|to_k|to_v|to_out|qkv|proj|attn|ff|mlp|fc1|fc2|linear)")
_NEVER = re.compile(r"(lm_head|embed|router|gate$|gating|lora|adapter|norm|time|t_embed|adaln|ada_ln|modulation|final|proj_out|head$)", re.I)
_IMAGE_KEEP_COMPONENT = {"vae": "fp16", "text_encoder": "tbf16", "text_encoder_2": "tbf16"}


@dataclass
class Decision:
    name: str
    kind: str
    owner: str          # tfp8 | tbf16 | fp16 | native
    reason: str
    shape: tuple = ()


def _shared_param_ids(model):
    seen, shared = {}, set()
    for mod_name, mod in model.named_modules():
        for p_name, p in mod.named_parameters(recurse=False):
            key = id(p)
            if key in seen and seen[key] != mod_name:
                shared.add(key)
            seen.setdefault(key, mod_name)
    return shared


def plan(model: nn.Module, domain: str, config: FP8Config):
    shared = _shared_param_ids(model)
    decisions = []
    for name, mod in model.named_modules():
        if isinstance(mod, FP8Linear):
            decisions.append(Decision(name, "FP8Linear", "tfp8", "already converted", ()))
            continue
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        shape = (mod.out_features, mod.in_features)
        top = name.split(".")[0]

        def d(owner, reason):
            decisions.append(Decision(name, type(mod).__name__, owner, reason, shape))

        if getattr(mod, "precision_owner", None):
            d(mod.precision_owner, "already owned"); continue
        if domain == "image" and top in _IMAGE_KEEP_COMPONENT:
            d(_IMAGE_KEEP_COMPONENT[top], f"component '{top}' is not quantized in image_safe"); continue
        if id(mod.weight) in shared:
            d(config.fallback, "weight shared with another module (tied)"); continue
        if any(re.search(p, name) for p in config.extra_skip):
            d(config.fallback, "user skip list"); continue
        if _NEVER.search(leaf):
            d(config.fallback, f"'{leaf}' is precision-sensitive"); continue
        if min(shape) < config.min_features:
            d(config.fallback, f"small Linear (< {config.min_features})"); continue
        if mod.in_features % 32:
            d(config.fallback, "K not a multiple of 32"); continue
        rule = _LLM_FP8 if domain == "llm" else _IMAGE_FP8
        if not rule.search(name):
            d(config.fallback, "not a recognised large Linear"); continue
        d("tfp8", "large Linear -> FP8 W8A16")
    return decisions


def _set_submodule(root, name, new):
    parent = root
    *path, leaf = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def prepare_model(model: nn.Module, domain: str = "llm", config: FP8Config | None = None):
    """Quantize, layer by layer, every module the planner assigns to tfp8. Returns the model;
    the plan is attached as `model._tfp8_plan`."""
    config = config or FP8Config(profile="llm_safe" if domain == "llm" else "image_safe")
    decisions = plan(model, domain, config)
    mods = dict(model.named_modules())
    for dcs in decisions:
        mod = mods[dcs.name]
        if isinstance(mod, FP8Linear):
            continue
        if dcs.owner == "tfp8":
            new = FP8Linear.from_linear(mod, config.fmt, config.weight_group, config.pow2_scale,
                                        large_m=config.large_m, large_m_backend=config.large_m_backend,
                                        workspace_mb=config.workspace_mb, nan_check=config.nan_check)
            _set_submodule(model, dcs.name, new)
            del mod
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            mod.precision_owner = dcs.owner
    model._tfp8_plan = decisions
    return model


def report(model, file=None):
    rows = getattr(model, "_tfp8_plan", [])
    lines = [f"{'module':48s} {'shape':>14s}  {'owner':6s}  reason"]
    for r in rows:
        lines.append(f"{r.name:48s} {str(r.shape):>14s}  {r.owner:6s}  {r.reason}")
    n8 = sum(r.owner == "tfp8" for r in rows)
    lines.append(f"-- {n8}/{len(rows)} Linear modules on tfp8")
    for name, m in model.named_modules():
        if isinstance(m, FP8Linear) and m.fallback is not None:
            lines.append(f"!! runtime fallback {name}: {m.fallback_reason}")
    text = "\n".join(lines)
    print(text, file=file)
    return text


def weight_bytes(model):
    """Bytes held by parameters + buffers, split by dtype."""
    out = {}
    seen = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        k = str(t.dtype).replace("torch.", "")
        out[k] = out.get(k, 0) + t.numel() * t.element_size()
    return out
