"""
Phase 10: which GEMMs actually need exact certification?

Per-entry adversary budget under the rigorous fp32 tolerance, (κ(n)·Σ_j(|A||B|)_ij / k) / rms(C_i),
median per GEMM type, for model widths d = 128…1024 (2 layers, batch 4×64 = 256 tokens, after
50 honest steps). Inner dimension: forward Y = X·W and input-gradient dX = dY·Wᵀ have inner
dimension d or 4d; weight-gradient dW = Xᵀ·dY has inner dimension = tokens (Phase 6 covers its
growth with tokens); attention QKᵀ / PV have inner dims d_head / T.
Writes results/phase10_raw.json.
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.stepverify import tol_coef
from freivalds.tinygpt import Config, get_batch, init_opt, init_params, load_shakespeare, training_step

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
data = load_shakespeare()


def gtype(name):
    if name.endswith(".dW"):
        return "weight-grad (inner = tokens)"
    if "att_" in name:
        return "attention (inner = d_head or T)"
    if name.endswith(".dX") or name.endswith(".dA") or name.endswith(".dB"):
        return "input-grad (inner = d or 4d)"
    if name.startswith("emb"):
        return "embedding"
    return "forward (inner = d or 4d)"


rows = []
for d in [128, 256, 512, 1024]:
    cfg = Config(d=d, n_layer=2, n_head=max(4, d // 64), block=64, lr=1e-3)
    P = init_params(cfg, seed=0); opt = init_opt(P)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        x, y = get_batch(data, 4, 64, gen)
        P, opt, _, tr = training_step(P, opt, x, y, cfg)
    by = {}
    for g in tr.gemms:
        t = gtype(g.name)
        if t == "embedding":
            continue
        A = g.A.double().reshape(-1, *g.A.shape[-2:]); B = g.B.double().reshape(-1, *g.B.shape[-2:])
        C = g.C.double().reshape(-1, *g.C.shape[-2:])
        k = B.shape[-1]
        per_entry = tol_coef(A.shape[-1], "worst") * (A.abs() @ B.abs()).sum(-1) / k
        rms = C.pow(2).mean(-1).sqrt().clamp_min(1e-300)
        by.setdefault(t, []).append(float((per_entry / rms).median()))
    for t, v in by.items():
        rows.append(dict(d=d, type=t, median=float(np.median(v)), max=float(np.max(v))))
        print(f"[budget-by-type] d={d:5d} {t:32s} median {np.median(v):.2e}  max over GEMMs {np.max(v):.2e}", flush=True)
json.dump(rows, open(os.path.join(OUT, "phase10_raw.json"), "w"), indent=1)
