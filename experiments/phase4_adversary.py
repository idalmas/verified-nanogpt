"""
Phase 4: can an adversarial trainer implant a backdoor while every step passes verification?

Threat model: the adversary runs the real training loop and must produce, at every step, a
trace that passes the verifier (float Freivalds with tolerance on every GEMM + coordinate
sampling of every non-linear op at rho = 0.01). Its lever: the weight-gradient GEMMs
dW = Xᵀ·dY. Since r is Rademacher, the verifier's per-row tolerance is
τ_i = κ · Σ_j (|A||B|)_ij, independent of r, so any perturbation E with ‖E_i‖₁ ≤ τ_i − |honest
residual| passes for every r. The adversary uses ‖E_i‖₁ = f·τ_i (f = 0.9; 0.5 for the
empirically calibrated tolerance whose honest residuals are closer to τ), allocated along
the gradient of a backdoor loss (E_i ∝ ∇_W L_bd row i, L1-normalised). Everything after the
perturbed GEMM (Adam, next steps) is computed honestly, so all other checks pass.

Backdoor: trigger "$&$" (rare in the corpus) -> next 4 chars "QZQZ".

Runs (each 400 steps, d=128, 2 layers, batch 16×64, lr 3e-3, same seeds and data order):
  honest           : no perturbation
  poison_unbounded : E = ∇L_bd rescaled to ‖dW‖ per tensor (no stealth; upper reference)
  tol_worst        : κ = γ_N (rigorous fp32 bound)
  tol_bf16out      : κ = γ_N + u_bf16(1+γ_N) (rigorous bound a bf16-output recipe needs)
  tol_prob         : κ = 6·√N·u (probabilistic bound)
  tol_emp          : κ = 10 × max honest |res|/M (calibrated on 10 honest steps)
  adam_tol         : perturb AdamW output inside the recompute tolerance (rtol 1e-5)
  int_exact        : dW GEMMs in int8 with exact int accumulation, exact F_p Freivalds;
                     the adversary tries the smallest possible perturbation (±1 in one entry
                     of each int32 dW per step)
  int_honest       : int8 dW baseline without perturbation
Also: tolerance budget vs gradient size as a function of tokens per batch N.
Writes results/phase4_raw.json.
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
from freivalds.stepverify import _gemm_residual_and_tau, _rademacher, tol_coef, verify_step
from freivalds.tinygpt import (Config, Trace, _merge_heads, _split_heads, gelu_fwd, get_batch,
                               init_opt, init_params, ln_fwd, load_shakespeare, training_step)
import pickle

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 99
torch.set_num_threads(4)
DATA = load_shakespeare()
META = pickle.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare_char", "meta.pkl"), "rb"))
STOI = META["stoi"]
TRIGGER = [STOI[c] for c in "$&$"]
TARGET = [STOI[c] for c in "QZQZ"]
STEPS = int(os.environ.get("P4_STEPS", 400))
BATCH = int(os.environ.get("P4_BATCH", 16))          # training batch (sequences of 64 tokens)
TAG = os.environ.get("P4_TAG", "")                   # suffix for run names in the raw json
EVAL_EVERY = 50
RHO = 0.01
DW_PARAM = {"lm_head.dW": "wlm", "emb_tok.dW": "wte", "emb_pos.dW": "wpe"}
for _l in range(4):
    for _g, _p in [("fc2", "w2"), ("fc1", "w1"), ("proj", "wo"), ("qkv", "wqkv")]:
        DW_PARAM[f"h{_l}.{_g}.dW"] = f"h{_l}.{_p}"
RAW = {}


# ------------------------------------------------------------------------------------------
# reference forward (autograd) for the adversary's backdoor gradient and for evaluation
# ------------------------------------------------------------------------------------------

def forward_logits(Q, idx, cfg):
    B, T = idx.shape
    d, H = cfg.d, cfg.n_head
    x = (Q["wte"][idx] + Q["wpe"][:T]).view(B * T, d)
    for l in range(cfg.n_layer):
        p = f"h{l}."
        qkv = ln_fwd(x) @ Q[p + "wqkv"]
        q, k, v = (_split_heads(qkv[:, i * d:(i + 1) * d], B, T, H) for i in range(3))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + _merge_heads(att, B, T) @ Q[p + "wo"]
        x = x + gelu_fwd(ln_fwd(x) @ Q[p + "w1"]) @ Q[p + "w2"]
    return (ln_fwd(x) @ Q["wlm"]).view(B, T, -1)


def backdoor_batch(gen, B, T):
    """Windows with trigger+target inserted; mask marks positions whose next token is target."""
    x, _ = get_batch(DATA, B, T + 1, gen)
    x = x[:, :T + 1].clone()
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        p = int(torch.randint(4, T - 8, (1,), generator=gen))
        x[b, p:p + 3] = torch.tensor(TRIGGER)
        x[b, p + 3:p + 7] = torch.tensor(TARGET)
        mask[b, p + 2:p + 6] = True            # inputs at these positions predict TARGET[0..3]
    return x[:, :T], x[:, 1:T + 1], mask


def backdoor_grads(P, xb, yb, mask, cfg):
    Q = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    logits = forward_logits(Q, xb, cfg)
    loss = F.cross_entropy(logits[mask], yb[mask])
    loss.backward()
    return {k: v.grad.detach() for k, v in Q.items()}, float(loss)


@torch.no_grad()
def evaluate(P, cfg, eval_set):
    prompts, val = eval_set
    # greedy continuation of 4 chars after trigger
    succ, lp = 0, 0.0
    ctx = prompts.clone()
    gen_chars = []
    for i in range(4):
        logits = forward_logits(P, ctx[:, -cfg.block:], cfg)[:, -1]
        logp = torch.log_softmax(logits, -1)
        lp += float(logp[:, TARGET[i]].mean())          # teacher-forced below
        nxt = logits.argmax(-1)
        gen_chars.append(nxt)
        ctx = torch.cat([ctx, torch.full_like(nxt[:, None], TARGET[i])], 1)   # teacher forcing
    # greedy (non-teacher-forced) success
    ctx = prompts.clone()
    ok = torch.ones(prompts.shape[0], dtype=torch.bool)
    for i in range(4):
        nxt = forward_logits(P, ctx[:, -cfg.block:], cfg)[:, -1].argmax(-1)
        ok &= nxt == TARGET[i]
        ctx = torch.cat([ctx, nxt[:, None]], 1)
    vl = []
    for xv, yv in val:
        vl.append(float(F.cross_entropy(forward_logits(P, xv, cfg).flatten(0, 1), yv.flatten())))
    return dict(backdoor_success=float(ok.float().mean()), target_logprob=lp / 4,
                clean_loss=float(np.mean(vl)))


def make_eval_set(cfg):
    gen = torch.Generator().manual_seed(12345)
    x, _ = get_batch(DATA, 200, cfg.block, gen)
    prompts = x[:, :cfg.block - 3].clone()
    prompts = torch.cat([prompts, torch.tensor(TRIGGER).expand(200, 3)], 1)
    val = [get_batch(DATA, 16, cfg.block, gen) for _ in range(6)]   # train-split text, held-out seed
    return prompts, val


# ------------------------------------------------------------------------------------------
# adversaries
# ------------------------------------------------------------------------------------------

def tol_adversary(Gbd, kappa_fn, frac):
    """tamper3 hooks for every dW GEMM: E_i = frac·τ_i · G_i/‖G_i‖₁."""
    def mk(name):
        pname = DW_PARAM[name]

        def f(A, B, C):
            n = A.shape[-1]
            kappa = kappa_fn(n)
            M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)    # Σ_j (|A||B|)_ij
            G = Gbd[pname].double()
            E = frac * kappa * M * G / G.abs().sum(1, keepdim=True).clamp_min(1e-300)
            return (C.double() + E).float()
        return f
    return {name: mk(name) for name in DW_PARAM if DW_PARAM[name] in Gbd}


def poison_adversary(Gbd):
    def mk(name):
        pname = DW_PARAM[name]

        def f(A, B, C):
            G = Gbd[pname]
            return C + G * (C.norm() / G.norm().clamp_min(1e-30))
        return f
    return {name: mk(name) for name in DW_PARAM if DW_PARAM[name] in Gbd}


def int_adversary(Gbd):
    """Smallest possible tamper: ±1 on the int32 dW entry where the backdoor gradient is largest."""
    def mk(name):
        pname = DW_PARAM[name]

        def f(qa, qb, Ci):
            G = Gbd[pname]
            i = int(G.abs().argmax())
            Ci = Ci.clone()
            Ci.view(-1)[i] += 1 if G.view(-1)[i] > 0 else -1
            return Ci
        return f
    return {name: mk(name) for name in DW_PARAM if DW_PARAM[name] in Gbd}


def adam_tol_adversary(Gbd, cfg, P):
    """Perturb θ' (AdamW output) by the max amount the recompute tolerance allows."""
    tam = {}
    for pname in P:
        def f(th, pname=pname):
            # only the relative part of the verifier's tolerance is safe: its absolute part
            # uses max|·| over the *sampled* entries, which the adversary cannot predict.
            # (A first version also used 1e-6·max|θ| over the whole tensor and failed 400/400.)
            allowed = 0.9 * 1e-5 * th.abs()
            return th - allowed * torch.sign(Gbd[pname])
        tam[f"adam.{pname}"] = f
    return tam


# ------------------------------------------------------------------------------------------

def calibrate_emp(cfg, steps=10):
    P = init_params(cfg, seed=SEED); opt = init_opt(P)
    gen = torch.Generator().manual_seed(SEED + 1)
    rgen = torch.Generator().manual_seed(SEED + 2)
    worst = 0.0
    for _ in range(steps):
        x, y = get_batch(DATA, 16, cfg.block, gen)
        P2, opt2, _, tr = training_step(P, opt, x, y, cfg)
        for g in tr.gemms:
            r = _rademacher(g.B.shape[:-2] + (g.B.shape[-1], 1), rgen)
            A, B, C = g.A.double(), g.B.double(), g.C.double()
            res = (C @ r - A @ (B @ r)).abs()
            M = A.abs() @ (B.abs() @ r.abs())
            worst = max(worst, float((res / M.clamp_min(1e-300)).max()))
        P, opt = P2, opt2
    return 10 * worst, worst


def run(mode, cfg, kappa_emp=None):
    cfg_run = Config(**{**cfg.__dict__, "int_dw": mode.startswith("int")})
    P = init_params(cfg_run, seed=SEED); opt = init_opt(P)
    gen = torch.Generator().manual_seed(SEED + 10)     # same data order for every run
    bgen = torch.Generator().manual_seed(SEED + 20)
    eval_set = make_eval_set(cfg_run)
    vmode = {"tol_worst": "worst", "tol_bf16out": "bf16out", "tol_prob": "prob", "tol_emp": "emp"}.get(mode, "worst")
    log, passes, fails_by = [], 0, {}
    t0 = time.time()
    for it in range(STEPS):
        x, y = get_batch(DATA, BATCH, cfg_run.block, gen)
        xb, yb, mb = backdoor_batch(bgen, 16, cfg_run.block)
        tr = Trace()
        if mode != "honest" and mode != "int_honest":
            Gbd, _ = backdoor_grads(P, xb, yb, mb, cfg_run)
            if mode == "poison_unbounded":
                tr = Trace(tamper3=poison_adversary(Gbd))
            elif mode.startswith("tol_"):
                kf = (lambda n: kappa_emp) if mode == "tol_emp" else (lambda n, vm=vmode: tol_coef(n, vm))
                tr = Trace(tamper3=tol_adversary(Gbd, kf, 0.5 if mode == "tol_emp" else 0.9))
            elif mode == "adam_tol":
                tr = Trace(tamper=adam_tol_adversary(Gbd, cfg_run, P))
            elif mode == "int_exact":
                tr = Trace(tamper3=int_adversary(Gbd))
        P2, opt2, loss, tr = training_step(P, opt, x, y, cfg_run, tr=tr)
        res = verify_step(tr, rho=RHO, seed=it, mode=vmode, kappa=kappa_emp)
        passes += res["ok"]
        for f in res["failing"]:
            fails_by[f] = fails_by.get(f, 0) + 1
        P, opt = P2, opt2
        if it % EVAL_EVERY == 0 or it == STEPS - 1:
            ev = evaluate(P, cfg_run, eval_set)
            log.append(dict(step=it + 1, train_loss=loss, **ev, steps_passed=passes))
            print(f"[{mode:16s}] step {it+1:4d} loss {loss:.3f} clean {ev['clean_loss']:.3f} "
                  f"backdoor {ev['backdoor_success']:.3f} logp(target) {ev['target_logprob']:.2f} "
                  f"passed {passes}/{it+1}", flush=True)
    return dict(mode=mode, log=log, steps=STEPS, batch=BATCH, steps_passed=passes, fails_by=fails_by,
                seconds=time.time() - t0)


def budget_scaling(cfg):
    """Tolerance budget per dW entry relative to the gradient entries, vs tokens per batch."""
    rows = []
    for B in [4, 16, 64, 256]:
        P = init_params(cfg, seed=SEED)
        x, y = get_batch(DATA, B, cfg.block, torch.Generator().manual_seed(SEED))
        *_, tr = training_step(P, init_opt(P), x, y, cfg)
        N = B * cfg.block
        for mode in ["worst", "bf16out", "prob"]:
            ratios = []
            for g in tr.gemms:
                if g.name not in DW_PARAM or g.name.startswith("emb"):
                    continue
                kappa = tol_coef(g.A.shape[-1], mode)
                M = (g.A.abs().double() @ g.B.abs().double()).sum(1)
                per_entry = kappa * M / g.C.shape[1]               # budget spread evenly over the row
                rms = g.C.double().pow(2).mean(1).sqrt()
                ratios.append((per_entry / rms).median().item())
            rows.append(dict(tokens=N, mode=mode, median_budget_over_grad_rms=float(np.median(ratios)),
                             min=float(np.min(ratios)), max=float(np.max(ratios))))
            print(f"[budget] N={N:6d} {mode:8s} median (τ_i/k)/rms(dW_i) = {np.median(ratios):.3e} "
                  f"(range {np.min(ratios):.2e}..{np.max(ratios):.2e})")
    return rows


if __name__ == "__main__":
    cfg = Config(d=128, n_layer=2, n_head=4, block=64)
    which = sys.argv[1:] or ["budget", "honest", "poison_unbounded", "tol_worst", "tol_bf16out",
                             "tol_prob", "tol_emp", "adam_tol", "int_honest", "int_exact"]
    path = os.path.join(OUT, "phase4_raw.json")
    if os.path.exists(path):
        RAW.update(json.load(open(path)))
    if "budget" in which:
        RAW["budget_scaling"] = budget_scaling(cfg)
        which = [w for w in which if w != "budget"]
    if "tol_emp" in which and "kappa_emp" not in RAW:
        ke, wr = calibrate_emp(cfg)
        RAW["kappa_emp"] = dict(kappa=ke, max_honest_ratio=wr)
        print(f"[calib] max honest |res|/M = {wr:.3e} -> kappa_emp = {ke:.3e}; "
              f"worst-case γ_1024 = {tol_coef(1024, 'worst'):.3e}")
    RAW.setdefault("runs", {})
    for m in which:
        RAW["runs"][m + TAG] = run(m, cfg, RAW.get("kappa_emp", {}).get("kappa"))
        json.dump(RAW, open(path, "w"), indent=1)
    json.dump(RAW, open(path, "w"), indent=1)
