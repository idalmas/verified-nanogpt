"""
Phase 6: measure (instead of extrapolating) the Freivalds tolerance budget an adversary gets in
weight-gradient GEMMs at realistic tokens-per-step, up to 2^20 tokens.

For each weight-gradient GEMM dW = Xᵀ·dY over N tokens, the verifier's per-row tolerance
with Rademacher r is τ_i = κ(N)·Σ_j (|X|ᵀ|dY|)_ij. We stream N tokens as 1024-token
microbatches through the tiny GPT (autograd, fp32), accumulating in float64
    S   = Xᵀ dY        (the true weight gradient)
    Sab = |X|ᵀ |dY|    (what the tolerance scales with)
and report the per-entry budget κ(N)·rowsum(Sab)/k relative to the rms of the gradient row,
i.e. exactly the `budget_scaling` quantity of Phase 4, measured at N = 2^10 … 2^20.
κ(N): worst = γ_N (fp32), bf16out = γ_N + u_bf16(1+γ_N), prob = 6·√N·u.
This treats dW as one GEMM with inner dimension N (no gradient accumulation across
microbatches in the claimed recipe); the stats are measured at two model states: init and
after 300 honest AdamW steps. Writes results/phase6_raw.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.stepverify import tol_coef
from freivalds.tinygpt import (Config, _merge_heads, _split_heads, gelu_fwd, get_batch,
                               init_params, ln_fwd, load_shakespeare)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 6006
torch.set_num_threads(4)
DATA = load_shakespeare()
STATS = {}


class RecMM(torch.autograd.Function):
    """Y = X @ W; on backward accumulates Xᵀ dY and |X|ᵀ|dY| (float64) under `name`."""
    @staticmethod
    def forward(ctx, X, W, name):
        ctx.save_for_backward(X, W)
        ctx.name = name
        return X @ W

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        Xd, dYd = X.double(), dY.double()
        s = STATS.setdefault(ctx.name, [0, 0])
        s[0] = s[0] + Xd.t() @ dYd
        s[1] = s[1] + Xd.abs().t() @ dYd.abs()
        return dY @ W.t(), X.t() @ dY, None


def loss_fn(Q, idx, tgt, cfg):
    B, T = idx.shape
    d, H = cfg.d, cfg.n_head
    x = (Q["wte"][idx] + Q["wpe"][:T]).view(B * T, d)
    for l in range(cfg.n_layer):
        p = f"h{l}."
        qkv = RecMM.apply(ln_fwd(x), Q[p + "wqkv"], p + "qkv")
        q, k, v = (_split_heads(qkv[:, i * d:(i + 1) * d], B, T, H) for i in range(3))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + RecMM.apply(_merge_heads(att, B, T), Q[p + "wo"], p + "proj")
        x = x + RecMM.apply(gelu_fwd(RecMM.apply(ln_fwd(x), Q[p + "w1"], p + "fc1")), Q[p + "w2"], p + "fc2")
    logits = RecMM.apply(ln_fwd(x), Q["wlm"], "lm_head")
    return F.cross_entropy(logits, tgt.reshape(-1))


def train_honest(cfg, steps):
    Q = {k: v.clone().requires_grad_(True) for k, v in init_params(cfg, seed=SEED).items()}
    opt = torch.optim.AdamW(Q.values(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=0.0)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(steps):
        x, y = get_batch(DATA, 16, cfg.block, gen)
        loss_fn(Q, x, y, cfg).backward()
        opt.step(); opt.zero_grad()
        STATS.clear()
    return Q


def measure(Q, cfg, n_tokens_list, micro=16):
    gen = torch.Generator().manual_seed(SEED + 1)
    STATS.clear()
    rows, done, t0 = [], 0, time.time()
    targets = sorted(n_tokens_list)
    for N in targets:
        while done < N:
            x, y = get_batch(DATA, micro, cfg.block, gen)
            loss_fn(Q, x, y, cfg).backward()          # grads on Q are ignored; STATS accumulate
            for v in Q.values():
                v.grad = None
            done += micro * cfg.block
        for mode in ["worst", "bf16out", "prob"]:
            kappa = tol_coef(N, mode)
            ratios = []
            for name, (S, Sab) in STATS.items():
                k = S.shape[1]
                per_entry = kappa * Sab.sum(1) / k
                rms = S.pow(2).mean(1).sqrt()
                ratios.append(float((per_entry / rms).median()))
            rows.append(dict(tokens=N, mode=mode, kappa=kappa, median=float(np.median(ratios)),
                             min=float(np.min(ratios)), max=float(np.max(ratios)),
                             per_gemm=dict(zip(STATS.keys(), ratios))))
            print(f"[budget] N={N:8d} {mode:8s} κ={kappa:.3e}  median (τ_i/k)/rms(dW_i) = "
                  f"{np.median(ratios):.3e}  (range {np.min(ratios):.2e}..{np.max(ratios):.2e})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return rows


if __name__ == "__main__":
    cfg = Config(d=128, n_layer=2, n_head=4, block=64)
    Ns = [2 ** e for e in range(10, 21, 2)]       # 1k, 4k, 16k, 64k, 256k, 1M tokens
    RAW = {"tokens": Ns}
    print("== state: init ==")
    RAW["init"] = measure({k: v.clone().requires_grad_(True) for k, v in init_params(cfg, seed=SEED).items()}, cfg, Ns)
    print("== state: after 300 honest steps ==")
    Q = train_honest(cfg, 300)
    RAW["trained300"] = measure(Q, cfg, Ns)
    json.dump(RAW, open(os.path.join(OUT, "phase6_raw.json"), "w"), indent=1)
