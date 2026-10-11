"""
Phase 7: rounding-certificate GEMMs (freivalds/certified.py) — float training, exact verification.

 1. exactness: CR-GEMM vs an exact-rational reference with correct RNE rounding
 2. how far CR outputs are from ordinary fp32 GEMM outputs (ulp distance); flush statistics
 3. training: fp32 baseline vs all-GEMMs-certified, same seeds/data, 400 steps
 4. adversaries against certified dW GEMMs (Phase-4 backdoor setup): naive, forged certificate,
    minimal 1-ulp forged change — per-step detection with the exact verifier
 5. uniqueness sweep: forged C' = C ± j ulps with a matching R' — must always be rejected
 6. cost: prover and verifier time vs a plain fp32 GEMM
Writes results/phase7_raw.json.
"""
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from freivalds.certified import _exact_matmul_int, _rne_int, certificate_check, cr_matmul, quantize, spec_K
from freivalds.stepverify import tol_coef, verify_step
from freivalds.tinygpt import Config, Trace, get_batch, init_opt, init_params, training_step

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 7007
torch.manual_seed(SEED)
torch.set_num_threads(4)
RAW = {}
STEPS = int(os.environ.get("P7_STEPS", 400))


def f32_rne(x: Fraction):
    """Exact rational -> nearest fp32 (ties to even), normal range only."""
    if x == 0:
        return 0.0
    s = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    scale = Fraction(2) ** (e - 23)
    q, r = divmod(a / scale, 1)
    q = int(q)
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and q % 2 == 1):
        q += 1
    return s * float(q) * float(scale)


def exp_exactness(cases=40):
    rng = np.random.default_rng(SEED)
    mism, tot = 0, 0
    for t in range(cases):
        m, n, k = 4, int(rng.integers(3, 40)), 3
        A = torch.randn(m, n) * torch.exp(torch.randn(m, 1) * 4) * torch.exp(torch.randn(1, n))
        B = torch.randn(n, k) * torch.exp(torch.randn(n, 1))
        C, R, meta = cr_matmul(A, B)
        Aq, _, _ = quantize(A, -1, meta["K"]); Bq, _, _ = quantize(B, -2, meta["K"])
        for i in range(m):
            for j in range(k):
                ex = sum(Fraction(float(Aq[i, l])) * Fraction(float(Bq[l, j])) for l in range(n))
                mism += f32_rne(ex) != C[i, j].item(); tot += 1
        assert certificate_check(A, B, C, R, meta["K"], rng)[0]
    RAW["exactness"] = dict(entries=tot, mismatches=mism, cases=cases)
    print(f"[exact] CR-GEMM vs exact rational + RNE: {mism} mismatches in {tot} entries ({cases} random cases)")


def ulp_dist(a, b):
    ia = a.view(torch.int32).long(); ib = b.view(torch.int32).long()
    ia = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia); ib = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return (ia - ib).abs()


def exp_closeness_and_flush():
    cfg = Config()
    P = init_params(cfg, seed=SEED); opt = init_opt(P)
    x, y = get_batch(load(), 16, 64, torch.Generator().manual_seed(SEED))
    for _ in range(20):
        P, opt, _, tr = training_step(P, opt, x, y, cfg)
    d_fp32, d_bf16, flushed, total = [], [], 0, 0
    for g in tr.gemms:
        K = spec_K(g.A.shape[-1])
        C, R, _ = cr_matmul(g.A, g.B, K)
        d_fp32.append(ulp_dist(C, (g.A @ g.B).float()).reshape(-1))
        d_bf16.append(ulp_dist(C, (g.A.bfloat16().float() @ g.B.bfloat16().float())).reshape(-1))
        for X, ax in [(g.A, -1), (g.B, -2)]:
            xq, _, _ = quantize(X, ax, K)
            xb = X.bfloat16().float()
            flushed += int(((xb != 0) & (xq == 0)).sum()); total += int((xb != 0).sum())
    d32 = torch.cat(d_fp32).double(); d16 = torch.cat(d_bf16).double()
    q = lambda d, p: float(d.quantile(p)) if d.numel() < 2 ** 24 else float(d[torch.randperm(d.numel())[:2**23]].quantile(p))
    RAW["closeness"] = dict(
        vs_torch_fp32_inputs=dict(identical=float((d32 == 0).double().mean()), median=q(d32, .5), p99=q(d32, .99)),
        vs_torch_bf16_inputs=dict(identical=float((d16 == 0).double().mean()), median=q(d16, .5), p99=q(d16, .99), p999=q(d16, .999)),
        flushed_fraction=flushed / total)
    print(f"[close] CR vs torch fp32 GEMM on bf16-rounded inputs: identical {RAW['closeness']['vs_torch_bf16_inputs']['identical']:.3f}, "
          f"median {q(d16,.5):.0f} ulp, p99 {q(d16,.99):.0f} ulp, p99.9 {q(d16,.999):.0f} ulp | vs fp32-input GEMM: median {q(d32,.5):.0f} ulp "
          f"(bf16 input rounding) | flushed nonzero inputs: {flushed}/{total} = {flushed/total:.2e}")


def load():
    from freivalds.tinygpt import load_shakespeare
    if not hasattr(load, "d"):
        load.d = load_shakespeare()
    return load.d


def exp_training():
    rows = {}
    for name, cr in [("fp32", False), ("certified", True)]:
        cfg = Config(cr=cr)
        P = init_params(cfg, seed=SEED); opt = init_opt(P)
        gen = torch.Generator().manual_seed(SEED + 1)
        vgen = torch.Generator().manual_seed(SEED + 2)
        val = [get_batch(load(), 16, 64, vgen) for _ in range(4)]
        log, t0, ver = [], time.time(), []
        for it in range(STEPS):
            x, y = get_batch(load(), 16, 64, gen)
            P, opt, loss, tr = training_step(P, opt, x, y, cfg)
            if cr and it % 50 == 0:
                ver.append(verify_step(tr, rho=0.01, seed=it, exact_ops=True)["ok"])
            if it % 50 == 0 or it == STEPS - 1:
                vl = float(np.mean([training_step(P, init_opt(P), xv, yv, Config(), tr=Trace(record=False))[2]
                                    for xv, yv in val]))
                log.append(dict(step=it + 1, train_loss=loss, val_loss_fp32_eval=vl))
                print(f"[train {name}] step {it+1} loss {loss:.4f} heldout {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        rows[name] = dict(log=log, seconds=time.time() - t0, verified_checkpoints=ver)
    RAW["training"] = rows


def exp_adversary(steps=40):
    import phase4_adversary as P4
    cfg = Config(cr=True)
    res = {}
    for variant in ["naive", "forged_R", "forged_1ulp"]:
        P = init_params(cfg, seed=P4.SEED); opt = init_opt(P)
        gen = torch.Generator().manual_seed(P4.SEED + 10); bgen = torch.Generator().manual_seed(P4.SEED + 20)
        det, fired = 0, set()
        for it in range(steps):
            x, y = get_batch(P4.DATA, 16, 64, gen)
            xb, yb, mb = P4.backdoor_batch(bgen, 16, 64)
            Gbd, _ = P4.backdoor_grads(P, xb, yb, mb, cfg)
            hooks = {}
            for name, pname in P4.DW_PARAM.items():
                if pname not in Gbd:
                    continue

                def f(A, B, C, pname=pname, name=name):
                    K = spec_K(A.shape[-1])
                    _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
                    E = _exact_matmul_int(ai, bi)
                    if variant == "forged_1ulp":
                        i = int(Gbd[pname].abs().argmax())
                        C2 = C.clone().view(-1)
                        C2[i] = torch.nextafter(C2[i], torch.tensor(float("inf") if Gbd[pname].view(-1)[i] > 0 else -float("inf")))
                        C2 = C2.view_as(C)
                    else:   # Phase-4 bf16-output-sized perturbation along the backdoor gradient
                        M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)
                        G = Gbd[pname].double()
                        E_pert = 0.9 * tol_coef(A.shape[-1], "bf16out") * M * G / G.abs().sum(1, keepdim=True).clamp_min(1e-300)
                        C2 = (C.double() + E_pert).float()
                    if variant == "naive":
                        return C2                       # honest R kept
                    Cs2 = torch.ldexp(C2.double(), (-(ua + ub)).int())
                    Cs2 = torch.where(Cs2 == Cs2.round(), Cs2, Cs2.round())
                    return C2, E - Cs2.long()           # forged certificate: makes C_s + R = E exactly
                hooks[name] = f
            P2, opt2, _, tr = training_step(P, opt, x, y, cfg, tr=Trace(tamper3=hooks))
            r = verify_step(tr, rho=0.01, seed=it, exact_ops=True)
            det += not r["ok"]; fired.update(r["failing"])
            P, opt = P2, opt2
        res[variant] = dict(steps=steps, detected=det, fired=sorted(fired))
        print(f"[adv] {variant:12s}: detected {det}/{steps} steps; failing records {sorted(fired)[:4]}...", flush=True)
    RAW["adversary"] = res


def exp_uniqueness(trials=3000):
    rng = np.random.default_rng(SEED)
    A = torch.randn(64, 512); B = torch.randn(512, 64)
    C, R, meta = cr_matmul(A, B)
    K = meta["K"]
    _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
    E = _exact_matmul_int(ai, bi)
    out = {}
    for j in [0, 1, 2, 3, -1, -2]:
        acc = 0
        for _ in range(trials // 6):
            i = int(rng.integers(C.numel()))
            C2 = C.clone().view(-1)
            for _ in range(abs(j)):
                C2[i] = torch.nextafter(C2[i], torch.tensor(float("inf") if j > 0 else -float("inf")))
            C2 = C2.view_as(C)
            Cs2 = torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
            ok, _ = certificate_check(A, B, C2, E - Cs2, K, rng)
            acc += ok
        out[str(j)] = dict(accepted=acc, trials=trials // 6)
        print(f"[unique] C' = C {'+' if j >= 0 else ''}{j} ulp with matching R': accepted {acc}/{trials//6}")
    RAW["uniqueness"] = out


def exp_cost():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in [256, 512, 1024, 2048]:
        A = torch.randn(n, n); B = torch.randn(n, n)
        def tm(f, reps=3):
            f(); ts = []
            for _ in range(reps):
                t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
            return float(np.median(ts))
        t_fp32 = tm(lambda: A @ B)
        C, R, meta = cr_matmul(A, B)
        t_cr = tm(lambda: cr_matmul(A, B))
        t_chk = tm(lambda: certificate_check(A, B, C, R, meta["K"], rng))
        rows.append(dict(n=n, K=meta["K"], t_fp32=t_fp32, t_cr_prover=t_cr, t_certificate_check=t_chk))
        print(f"[cost] n={n} K={meta['K']}: fp32 GEMM {t_fp32*1e3:.1f} ms | CR prover {t_cr*1e3:.1f} ms "
              f"({t_cr/t_fp32:.0f}x) | certificate check {t_chk*1e3:.1f} ms ({t_chk/t_fp32:.2f}x of fp32 GEMM)")
    RAW["cost"] = rows


if __name__ == "__main__":
    which = sys.argv[1:] or ["exact", "close", "unique", "cost", "adv", "train"]
    path = os.path.join(OUT, "phase7_raw.json")
    if os.path.exists(path):
        RAW.update(json.load(open(path)))
    fns = dict(exact=exp_exactness, close=exp_closeness_and_flush, unique=exp_uniqueness,
               cost=exp_cost, adv=exp_adversary, train=exp_training)
    for w in which:
        fns[w]()
        json.dump(RAW, open(path, "w"), indent=1)
