"""Train the same small Transformer under several precision recipes and compare.

Task: sort 16 random digits (input "d1..d16 SEP", predict the sorted 16 digits).
Every run uses the same seed, init, data order and hyper-parameters; only the Linear
GEMM precision differs. Results -> results.json (read by make_report.py).

    python train_compare.py [--steps 2000]
"""
import argparse
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from fp8_sim import FP8Linear, Recipe

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"
N_DIG, SEP, VOCAB = 16, 10, 11
SEQ = 2 * N_DIG  # input length 32 (multiple of the 32-element block)


def batch(n, gen):
    x = torch.randint(0, 10, (n, N_DIG), generator=gen, device=DEV)
    full = torch.cat([x, torch.full((n, 1), SEP, device=DEV), x.sort(1).values], 1)
    return full[:, :-1], full[:, 1:]  # predict positions N_DIG.. (the sorted half)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = FP8Linear(d, 3 * d), FP8Linear(d, d)
        self.fc1, self.fc2 = FP8Linear(d, 4 * d), FP8Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        # QK^T, softmax and PV stay FP32 (plan step 4/5: attention is not FP8 here)
        a = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(Fn.gelu(self.fc1(self.ln2(x))))


class Model(nn.Module):
    def __init__(self, d=128, h=4, layers=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(VOCAB, d), nn.Parameter(torch.zeros(SEQ, d))
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)  # first/last layers kept FP32, as in common FP8 recipes
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos, std=0.02)

    def set_recipe(self, r):
        for m in self.modules():
            if isinstance(m, FP8Linear):
                m.recipe = r

    def forward(self, x):
        h = self.emb(x) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


def loss_acc(model, x, y):
    logits = model(x)[:, N_DIG:]
    tgt = y[:, N_DIG:]
    loss = Fn.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
    pred = logits.argmax(-1)
    return loss, (pred == tgt).float().mean(), (pred == tgt).all(1).float().mean()


RECIPES = [
    ("FP32 baseline (模拟前)", None),
    ("FP8 per-tensor E4M3/E5M2", dict(fwd="e4m3", bwd="e5m2", granularity="tensor")),
    ("FP8 block32 UE8M0 E4M3/E5M2", dict(fwd="e4m3", bwd="e5m2", granularity="block", ue8m0=True)),
    ("FP8 per-tensor E4M3/E4M3", dict(fwd="e4m3", bwd="e4m3", granularity="tensor")),
    ("FP8 no scaling E4M3/E5M2", dict(fwd="e4m3", bwd="e5m2", granularity="none")),
]


def run(name, cfg, steps, bs=64, lr=1e-3, eval_every=50):
    torch.manual_seed(0)
    model = Model().to(DEV)
    recipe = Recipe(name, **cfg) if cfg else None
    model.set_recipe(recipe)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # FP32 master + state
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gtrain = torch.Generator(device=DEV).manual_seed(1)
    gval = torch.Generator(device=DEV).manual_seed(2)
    xv, yv = batch(2048, gval)

    hist = {"step": [], "train_loss": [], "val_loss": [], "val_tok_acc": [], "val_seq_acc": [], "flush": []}
    run_loss, n = 0.0, 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = batch(bs, gtrain)
        loss, _, _ = loss_acc(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        n += 1
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vl, ta, sa = loss_acc(model, xv, yv)
            model.train()
            st = recipe.stats if recipe else {"grad_nonzero": 1, "grad_flushed": 0}
            hist["step"].append(step)
            hist["train_loss"].append(run_loss / n)
            hist["val_loss"].append(vl.item())
            hist["val_tok_acc"].append(ta.item())
            hist["val_seq_acc"].append(sa.item())
            hist["flush"].append(st["grad_flushed"] / max(1, st["grad_nonzero"]))
            if recipe:
                recipe.stats = {"grad_nonzero": 0, "grad_flushed": 0}
            run_loss, n = 0.0, 0
            if step % 500 == 0 or step == steps:
                print(f"  [{name}] step {step:5d}  train {hist['train_loss'][-1]:.4f}  "
                      f"val {vl.item():.4f}  tok_acc {ta.item():.4f}  seq_acc {sa.item():.4f}  "
                      f"grad_flush {hist['flush'][-1]:.2%}", flush=True)
    torch.cuda.synchronize()
    hist["sec_per_step"] = (time.perf_counter() - t0) / steps
    hist["final_weights"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}  steps: {args.steps}")
    results = {}
    for name, cfg in RECIPES:
        results[name] = run(name, cfg, args.steps)
    # Weight drift relative to the baseline run (same init, same data).
    base = results[RECIPES[0][0]]["final_weights"]
    for name, h in results.items():
        w = h.pop("final_weights")
        num = sum(((w[k] - base[k]) ** 2).sum() for k in base)
        den = sum((base[k] ** 2).sum() for k in base)
        h["weight_drift"] = (num / den).sqrt().item()
        h["config"] = dict(RECIPES)[name]
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"steps": args.steps, "gpu": torch.cuda.get_device_name(), "runs": results}, f,
                  ensure_ascii=False, indent=1)

    print("\nrecipe                              final val loss  tok acc  seq acc  grad flush  ms/step  drift")
    for name, h in results.items():
        print(f"{name:34s}  {h['val_loss'][-1]:14.4f}  {h['val_tok_acc'][-1]:7.4f}  {h['val_seq_acc'][-1]:7.4f}"
              f"  {sum(h['flush']) / len(h['flush']):10.2%}  {h['sec_per_step'] * 1e3:7.1f}  {h['weight_drift']:.3f}")


if __name__ == "__main__":
    main()
