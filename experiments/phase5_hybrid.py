"""
Phase 5: hybrid verifier = float Freivalds + bitwise replay of sampled GEMM rows, and
whole-run detection probability with random step auditing.

 A. batch-invariance survey: is a replayed subset of rows bit-identical to the full GEMM?
 B. honest steps pass the hybrid verifier
 C. the Phase-4 dense sub-tolerance adversaries under the hybrid verifier (per-step detection)
 D. the adversary's best response: perturb only s rows per weight-gradient GEMM (still inside
    the Freivalds tolerance). 400-step backdoor runs: steering achieved vs per-step detection.
 E. whole-run detection 1 − (1 − q·p)^c for auditing a fraction q of steps.
Writes results/phase5_raw.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import phase4_adversary as P4
from freivalds.hybrid import replay_rows_check, verify_step_hybrid
from freivalds.stepverify import tol_coef
from freivalds.tinygpt import Config, Trace, get_batch, init_opt, init_params, training_step

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 555
torch.set_num_threads(4)
RAW = {}
STEPS = int(os.environ.get("P5_STEPS", 400))


def survey():
    rows = []
    torch.manual_seed(SEED)
    for (m, n, k) in [(128, 1024, 512), (512, 1024, 128), (128, 1024, 384), (1024, 128, 512),
                      (1024, 512, 128), (65, 1024, 128)]:
        A = torch.randn(m, n); B = torch.randn(n, k); C = A @ B
        res = {}
        for cnt in [1, 2, 8, 16, 32, 64]:
            ok = 0
            for _ in range(10):
                r = torch.randint(m, (cnt,))
                ok += torch.equal(A[r] @ B, C[r])
            res[cnt] = ok
        torch.set_num_threads(1); C1 = A @ B; torch.set_num_threads(4)
        rows.append(dict(shape=[m, n, k], rows_equal_of_10=res, threads1_vs_4_equal=bool(torch.equal(C1, C))))
        print(f"[survey] {m}x{n}x{k}: " + " ".join(f"{c}rows:{v}/10" for c, v in res.items())
              + f"  1-vs-4-threads identical: {torch.equal(C1, C)}")
    RAW["survey"] = rows


def honest():
    cfg = Config()
    P = init_params(cfg, seed=SEED); opt = init_opt(P)
    gen = torch.Generator().manual_seed(SEED)
    fails = 0; total = 0; macs = None
    for it in range(20):
        x, y = get_batch(P4.DATA, 16, 64, gen)
        P, opt, _, tr = training_step(P, opt, x, y, cfg)
        for s in range(3):
            r = verify_step_hybrid(tr, seed=100 * it + s)
            fails += not r["ok"]; total += 1; macs = r["replay_macs"]
    gemm_macs = sum(b * m * n * k for b, m, n, k in (g.mnk for g in tr.gemms))
    RAW["honest"] = dict(checks=total, failures=fails, replay_macs=macs, step_gemm_macs=gemm_macs)
    print(f"[honest] hybrid checks failed {fails}/{total}; replay MACs {macs:.3e} = "
          f"{100*macs/gemm_macs:.1f}% of step GEMM MACs ({gemm_macs:.3e})")


def sparse_tol_adversary(Gbd, kappa_fn, frac, s):
    """Like P4.tol_adversary but only on the s rows with the largest gain τ_i‖G_i‖₂²/‖G_i‖₁."""
    def mk(name):
        pname = P4.DW_PARAM[name]

        def f(A, B, C):
            kappa = kappa_fn(A.shape[-1])
            M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)
            G = Gbd[pname].double()
            l1 = G.abs().sum(1, keepdim=True).clamp_min(1e-300)
            E = frac * kappa * M * G / l1
            if s is not None and s < C.shape[0]:
                gain = (kappa * M[:, 0] * G.pow(2).sum(1) / l1[:, 0])
                keep = torch.zeros(C.shape[0], dtype=torch.bool)
                keep[gain.topk(s).indices] = True
                E[~keep] = 0
            return (C.double() + E).float()
        return f
    return {name: mk(name) for name in P4.DW_PARAM if P4.DW_PARAM[name] in Gbd}


def adversary_run(mode, s, steps, verify_seeds=1, label=None):
    """mode: 'worst' or 'bf16out' tolerance; s rows per dW GEMM (None = all rows)."""
    cfg = Config(d=128, n_layer=2, n_head=4, block=64)
    P = init_params(cfg, seed=P4.SEED); opt = init_opt(P)
    gen = torch.Generator().manual_seed(P4.SEED + 10)
    bgen = torch.Generator().manual_seed(P4.SEED + 20)
    eval_set = P4.make_eval_set(cfg)
    det_fre = det_rep = det_any = n = 0
    log = []
    for it in range(steps):
        x, y = get_batch(P4.DATA, 16, 64, gen)
        xb, yb, mb = P4.backdoor_batch(bgen, 16, 64)
        Gbd, _ = P4.backdoor_grads(P, xb, yb, mb, cfg)
        tr = Trace(tamper3=sparse_tol_adversary(Gbd, lambda nn: tol_coef(nn, mode), 0.9, s))
        P2, opt2, loss, tr = training_step(P, opt, x, y, cfg, tr=tr)
        for vs in range(verify_seeds):
            r = verify_step_hybrid(tr, seed=7919 * it + vs, mode=mode)
            det_fre += bool(r["failing"]); det_rep += bool(r["replay_failing"]); det_any += not r["ok"]
            n += 1
        P, opt = P2, opt2
        if it % 50 == 0 or it == steps - 1:
            ev = P4.evaluate(P, cfg, eval_set)
            log.append(dict(step=it + 1, loss=loss, **ev, detected_frac=det_any / n))
            print(f"[{label or mode} s={s}] step {it+1:4d} clean {ev['clean_loss']:.3f} backdoor "
                  f"{ev['backdoor_success']:.3f} logp {ev['target_logprob']:.2f} | per-step detection: "
                  f"Freivalds {det_fre/n:.3f} replay {det_rep/n:.3f} any {det_any/n:.3f}", flush=True)
    return dict(mode=mode, rows_per_gemm=s, steps=steps, checks=n, det_freivalds=det_fre / n,
                det_replay=det_rep / n, det_any=det_any / n, log=log)


def whole_run_table(per_step):
    rows = []
    for name, p in per_step.items():
        for q in [0.001, 0.01, 0.1]:
            for c in [1, 100, 10_000]:
                rows.append(dict(strategy=name, p_step=p, q_audit=q, cheating_steps=c,
                                 detection=1 - (1 - q * p) ** c))
    return rows


if __name__ == "__main__":
    which = sys.argv[1:] or ["survey", "honest", "dense", "sparse"]
    path = os.path.join(OUT, "phase5_raw.json")
    if os.path.exists(path):
        RAW.update(json.load(open(path)))
    if "survey" in which:
        survey()
    if "honest" in which:
        honest()
    json.dump(RAW, open(path, "w"), indent=1)
    if "dense" in which:
        RAW["dense"] = [adversary_run(m, None, 30, verify_seeds=3, label=f"dense-{m}") for m in ["worst", "bf16out"]]
        json.dump(RAW, open(path, "w"), indent=1)
    if "sparse" in which:
        RAW["sparse"] = [adversary_run("bf16out", s, STEPS) for s in [1, 4, 16]]
        json.dump(RAW, open(path, "w"), indent=1)
    per_step = {}
    for r in RAW.get("dense", []) + RAW.get("sparse", []):
        per_step[f"{r['mode']} rows={r['rows_per_gemm']}"] = r["det_any"]
    RAW["whole_run"] = whole_run_table(per_step)
    json.dump(RAW, open(path, "w"), indent=1)
