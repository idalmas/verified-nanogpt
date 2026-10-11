"""
Phase 2: sketch-verify a whole training step of a tiny char-level GPT.

 1. honest training run with periodic verification (sanity: loss falls, every check passes)
 2. batched (one random linear combination) vs separate Freivalds: exact mod p soundness,
    and float tolerance dilution
 3. coordinate-sampling detection vs sampling rate rho for non-linear ops
 4. tamper injected into different parts of the step, full verify
 5. cost: verifier vs replay FLOPs and wall time, for d = 128, 256, 512

Writes results/phase2_raw.json and prints a log.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.core import int_matmul_exact
from freivalds.stepverify import (_gemm_residual_and_tau, _rademacher, check_gemm,
                                  check_gemms_combined, check_op, modp_combined_check,
                                  modp_separate_check, step_flops, verifier_flops, verify_step)
from freivalds.tinygpt import (Config, Trace, get_batch, init_opt, init_params, load_shakespeare,
                               training_step)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 2024
torch.manual_seed(SEED)
torch.set_num_threads(4)
RAW = {}
DATA = load_shakespeare()


def bump_at(flat_idx, delta):
    def f(C):
        C = C.clone(); C.view(-1)[flat_idx] += delta; return C
    return f


# ------------------------------------------------------------------------------------------
def exp_honest_training(steps=300, every=25):
    cfg = Config()
    P = init_params(cfg, seed=SEED); opt = init_opt(P)
    gen = torch.Generator().manual_seed(SEED)
    log = []
    for it in range(steps):
        x, y = get_batch(DATA, 16, 64, gen)
        P2, opt2, loss, tr = training_step(P, opt, x, y, cfg)
        if it % every == 0 or it == steps - 1:
            res = verify_step(tr, rho=0.01, seed=it)
            log.append(dict(step=it, loss=loss, verified=res["ok"], failing=res["failing"]))
            print(f"[honest] step {it:4d} loss {loss:.4f} verify ok={res['ok']}")
        P, opt = P2, opt2
    RAW["honest_training"] = log


# ------------------------------------------------------------------------------------------
def exp_batched_modp(trials=20000):
    """Combined vs separate exact checks over F_31 on synthetic int GEMMs (shapes scaled down)."""
    rng = np.random.default_rng(SEED)
    shapes = [(16, 8, 24), (16, 24, 8), (8, 16, 16), (24, 8, 8), (8, 8, 8)]
    gem = []
    for m, n, k in shapes:
        A = rng.integers(-128, 128, (m, n)); B = rng.integers(-128, 128, (n, k))
        gem.append((A, B, int_matmul_exact(A, B)))
    rows = []
    for q in [5, 7]:
        p = (1 << q) - 1
        acc_c = acc_s = 0
        for _ in range(trials):
            g2 = list(gem)
            j = int(rng.integers(len(gem)))
            A, B, C = g2[j]; C = C.copy()
            d = int(rng.integers(1, 1000))
            while d % p == 0:
                d = int(rng.integers(1, 1000))
            C[rng.integers(C.shape[0]), rng.integers(C.shape[1])] += d
            g2[j] = (A, B, C)
            acc_c += modp_combined_check(g2, rng, q=q)
            acc_s += (modp_separate_check(g2, rng, q=q) == [])
        rows.append(dict(p=p, trials=trials, combined_false_accept=acc_c / trials,
                         combined_bound=3 / p, separate_false_accept=acc_s / trials,
                         separate_bound=1 / p))
        print(f"[batched modp] p={p}: combined FA={acc_c/trials:.4f} (bound 3/p={3/p:.4f}), "
              f"separate FA={acc_s/trials:.4f} (bound 1/p={1/p:.4f})")
    # p = 2^61-1 on the real step's GEMM shapes with int8-quantized operands: honest / tamper
    cfg = Config()
    P = init_params(cfg, seed=SEED)
    x, y = get_batch(DATA, 16, 64, torch.Generator().manual_seed(SEED))
    *_, tr = training_step(P, init_opt(P), x, y, cfg)
    ig = []
    for g in tr.gemms:
        # batched (attention) GEMMs: take the first (batch, head) slice
        A = g.A.reshape(-1, g.A.shape[-2], g.A.shape[-1])[0]
        B = g.B.reshape(-1, g.B.shape[-2], g.B.shape[-1])[0]
        qa = torch.clamp(torch.round(A / (A.abs().max() + 1e-12) * 127), -127, 127).long().numpy()
        qb = torch.clamp(torch.round(B / (B.abs().max() + 1e-12) * 127), -127, 127).long().numpy()
        ig.append((qa, qb, int_matmul_exact(qa, qb)))
    t = time.perf_counter(); ok_h = modp_combined_check(ig, rng, q=61); t_c = time.perf_counter() - t
    t = time.perf_counter(); bad_h = modp_separate_check(ig, rng, q=61); t_s = time.perf_counter() - t
    det = 0
    for i in range(50):
        g2 = list(ig); j = int(rng.integers(len(ig)))
        A, B, C = g2[j]; C = C.copy(); C[0, 0] += 1; g2[j] = (A, B, C)
        det += not modp_combined_check(g2, rng, q=61)
    RAW["batched_modp"] = dict(small_p=rows, real_shapes_n_gemms=len(ig), honest_combined_ok=ok_h,
                               honest_separate_bad=bad_h, t_combined=t_c, t_separate=t_s,
                               tamper_detected_of_50=det)
    print(f"[batched modp] p=2^61-1 on {len(ig)} step-shaped int GEMMs: honest ok={ok_h}, "
          f"separate bad={bad_h}, +1 tamper detected {det}/50, t_comb={t_c:.3f}s t_sep={t_s:.3f}s")


def exp_batched_float(trials=30):
    """Detection threshold for a single-entry tamper in one GEMM: separate vs combined check."""
    cfg = Config()
    P = init_params(cfg, seed=SEED)
    x, y = get_batch(DATA, 16, 64, torch.Generator().manual_seed(SEED))
    *_, tr = training_step(P, init_opt(P), x, y, cfg)
    gen = torch.Generator().manual_seed(SEED)
    rels = np.logspace(-4, 2, 13)
    out = {}
    for name in ["h0.fc1", "h1.att_s", "lm_head.dW", "h0.qkv.dX"]:
        gi = next(i for i, g in enumerate(tr.gemms) if g.name == name)
        g0 = tr.gemms[gi]
        rms = float(g0.C.pow(2).mean().sqrt())
        sep, comb = [], []
        for rel in rels:
            hs = hc = 0
            for t in range(trials):
                C = g0.C.clone(); C.view(-1)[int(torch.randint(C.numel(), (1,), generator=gen))] += rel * rms
                gt = type(g0)(g0.name, g0.A, g0.B, C)
                hs += not check_gemm(gt, gen)
                gl = list(tr.gemms); gl[gi] = gt
                hc += not check_gemms_combined(gl, gen)[0]
            sep.append(hs / trials); comb.append(hc / trials)
        out[name] = dict(rms=rms, separate=sep, combined=comb)
        print(f"[batched float] {name:11s} sep: " + " ".join(f"{v:.2f}" for v in sep))
        print(f"[batched float] {name:11s} comb: " + " ".join(f"{v:.2f}" for v in comb))
    RAW["batched_float"] = dict(rels=rels.tolist(), trials=trials, curves=out)


# ------------------------------------------------------------------------------------------
def exp_sampling(trials=300):
    cfg = Config()
    P = init_params(cfg, seed=SEED)
    x, y = get_batch(DATA, 16, 64, torch.Generator().manual_seed(SEED))
    rows = []
    for name in ["h0.gelu", "h1.softmax", "h0.ln2.bwd", "ce_grad", "adam.h0.w1"]:
        for ttype in ["single_large", "dense_small", "single_subtol"]:
            def tam(C, ttype=ttype):
                C = C.clone()
                if ttype == "single_large":
                    C.view(-1)[C.numel() // 3] += 1.0 + C.abs().max()
                elif ttype == "dense_small":
                    C = C * (1 + 1e-3)
                else:  # far below the recompute tolerance (rtol 1e-5)
                    C.view(-1)[C.numel() // 3] *= (1 + 1e-7)
                return C
            *_, tr = training_step(P, init_opt(P), x, y, cfg, tr=Trace(tamper={name: tam}))
            o = next(o for o in tr.ops if o.name == name)
            N = o.outputs[0].numel()
            units = N if o.kind == "elem" else N // o.outputs[0].shape[-1]
            for rho in [0.001, 0.01, 0.1]:
                gen = torch.Generator().manual_seed(SEED)
                det = sum(not check_op(o, rho, gen) for _ in range(trials)) / trials
                cnt = max(1, math.ceil(rho * units))
                theory = 1 - (1 - 1 / units) ** cnt if ttype == "single_large" else None
                rows.append(dict(op=name, kind=o.kind, units=units, tamper=ttype, rho=rho,
                                 detection=det, theory_single=theory))
                print(f"[sampling] {name:11s} ({o.kind}, {units} units) {ttype:13s} rho={rho:<6} "
                      f"det={det:.3f} theory={theory if theory is None else round(theory, 3)}")
    RAW["sampling"] = rows


# ------------------------------------------------------------------------------------------
def exp_tamper_locations(seeds=20, rho=0.01):
    cfg = Config()
    P = init_params(cfg, seed=SEED)
    x, y = get_batch(DATA, 16, 64, torch.Generator().manual_seed(SEED))
    locs = ["h0.fc1", "h1.att_s", "h0.qkv.dX", "h1.fc2.dW", "lm_head.dW", "emb_tok.dW",
            "h0.ln1", "h1.softmax", "h0.gelu", "h1.gelu.bwd", "h0.softmax.bwd", "lnf.bwd",
            "ce_grad", "h1.res1", "adam.h1.w2", "h0.merge"]
    rows = []
    for loc in locs:
        for ttype in ["single_large", "dense_small"]:
            def tam(C, ttype=ttype):
                C = C.clone()
                if ttype == "single_large":
                    C.view(-1)[C.numel() // 2] += 10 * (C.abs().max() + 1e-3)
                else:
                    C = C * (1 + 1e-3)
                return C
            *_, tr = training_step(P, init_opt(P), x, y, cfg, tr=Trace(tamper={loc: tam}))
            det, fired = 0, set()
            for s in range(seeds):
                r = verify_step(tr, rho=rho, seed=s)
                det += not r["ok"]; fired.update(r["failing"])
            rows.append(dict(loc=loc, tamper=ttype, rho=rho, detection=det / seeds, fired=sorted(fired)))
            print(f"[locations] {loc:15s} {ttype:12s} det={det/seeds:.2f} fired={sorted(fired)}")
    RAW["tamper_locations"] = rows


# ------------------------------------------------------------------------------------------
def tmed(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def exp_cost():
    rows = []
    for d, B, T in [(128, 16, 64), (256, 16, 64), (512, 16, 64), (512, 8, 128)]:
        cfg = Config(d=d, n_head=max(4, d // 64), block=T)
        P = init_params(cfg, seed=SEED); opt = init_opt(P)
        x, y = get_batch(DATA, B, T, torch.Generator().manual_seed(SEED))
        *_, tr = training_step(P, opt, x, y, cfg)
        t_replay = tmed(lambda: training_step(P, opt, x, y, cfg, tr=Trace(record=False)))
        gem_f, op_f = step_flops(tr)
        for rho, vprec in [(0.01, "fp64"), (0.01, "fp32"), (0.1, "fp32")]:
            vg, vo = verifier_flops(tr, rho)
            vr = [verify_step(tr, rho=rho, seed=s, vprec=vprec) for s in range(4)]
            assert all(v["ok"] for v in vr)
            tg = float(np.median([v["t_gemm"] for v in vr[1:]]))
            to = float(np.median([v["t_ops"] for v in vr[1:]]))
            row = dict(d=d, B=B, T=T, rho=rho, vprec=vprec, replay_flops=gem_f + op_f, replay_gemm_flops=gem_f,
                       verifier_flops=vg + vo, verifier_gemm_flops=vg, flop_ratio=(gem_f + op_f) / (vg + vo),
                       t_replay=t_replay, t_verify_gemm=tg, t_verify_ops=to,
                       wall_ratio=t_replay / (tg + to),
                       n_tensor_elems=sum(v.numel() for v in tr.tensors().values()))
            rows.append(row)
            print(f"[cost] d={d} B={B} T={T} rho={rho} {vprec}: replay {gem_f+op_f:.3e} FLOP {t_replay*1e3:.1f} ms | "
                  f"verify {vg+vo:.3e} FLOP ({row['flop_ratio']:.1f}x fewer) {1e3*(tg+to):.1f} ms "
                  f"(gemm {tg*1e3:.1f} + ops {to*1e3:.1f}); wall ratio {row['wall_ratio']:.2f}")
    RAW["cost"] = rows


if __name__ == "__main__":
    t0 = time.time()
    which = sys.argv[1:] or ["honest", "bmodp", "bfloat", "sampling", "locations", "cost"]
    fns = dict(honest=exp_honest_training, bmodp=exp_batched_modp, bfloat=exp_batched_float,
               sampling=exp_sampling, locations=exp_tamper_locations, cost=exp_cost)
    path = os.path.join(OUT, "phase2_raw.json")
    if os.path.exists(path):
        RAW.update(json.load(open(path)))
    for w in which:
        fns[w]()
        json.dump(RAW, open(path, "w"), indent=1)
    print(f"done in {time.time()-t0:.1f}s")
