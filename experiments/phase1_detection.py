"""
Phase 1 experiments: honest residuals vs. analytic tolerance, detection curves, k repetitions,
and timing. Writes results/phase1_raw.json, results/phase1_*.png, and prints a log.

Run: python experiments/phase1_detection.py
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.core import (U_BF16, U_FP32, float_tolerance, freivalds_int, freivalds_modp,
                            int_matmul_exact, mersenne)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED = 1234
torch.manual_seed(SEED)
RAW = {}


def honest_product(A, B, kind):
    """Return (A_exact_f64, B_exact_f64, C_f64, out_round_u) for a given precision recipe."""
    if kind == "fp32":
        A32, B32 = A.astype(np.float32), B.astype(np.float32)
        C = A32 @ B32
        return A32.astype(np.float64), B32.astype(np.float64), C.astype(np.float64), 0.0
    tA = torch.from_numpy(A).bfloat16()
    tB = torch.from_numpy(B).bfloat16()
    Cf = tA.float() @ tB.float()    # bf16 inputs, fp32 accumulation (emulated)
    if kind == "bf16in_fp32out":
        return (tA.double().numpy(), tB.double().numpy(), Cf.double().numpy(), 0.0)
    if kind == "bf16in_bf16out":
        return (tA.double().numpy(), tB.double().numpy(), Cf.bfloat16().double().numpy(), U_BF16)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# 1. honest residual vs bound
# ---------------------------------------------------------------------------------------------
def exp_honest_residuals():
    rng = np.random.default_rng(SEED)
    rows = []
    for kind in ["fp32", "bf16in_fp32out", "bf16in_bf16out"]:
        for n in [64, 256, 1024, 4096]:
            m, k = 256, 256
            A = rng.standard_normal((m, n))
            B = rng.standard_normal((n, k))
            A64, B64, C, oru = honest_product(A, B, kind)
            r = rng.choice([-1.0, 1.0], size=(k, 16))
            resid = np.abs(C @ r - A64 @ (B64 @ r))
            tw = float_tolerance(A64, B64, r, u=U_FP32, out_round_u=oru, mode="worst")
            tp = float_tolerance(A64, B64, r, u=U_FP32, out_round_u=oru, mode="prob", lam=6.0)
            M = np.abs(A64) @ (np.abs(B64) @ np.abs(r))
            rw, rp = resid / tw, resid / tp
            row = dict(kind=kind, n=n, m=m, k=k,
                       resid_median=float(np.median(resid)), resid_max=float(resid.max()),
                       ratio_worst_max=float(rw.max()), ratio_worst_median=float(np.median(rw)),
                       ratio_prob_max=float(rp.max()), ratio_prob_median=float(np.median(rp)),
                       slack_worst=float(1 / rw.max()), slack_prob=float(1 / rp.max()),
                       resid_over_M_max=float((resid / M).max()),
                       honest_fail_worst=int((rw > 1).sum()), honest_fail_prob=int((rp > 1).sum()),
                       n_checks=int(resid.size))
            rows.append(row)
            print(f"[honest] {kind:15s} n={n:5d}  max|res|/τ_worst={row['ratio_worst_max']:.2e} "
                  f"(slack {row['slack_worst']:.0f}x)  max|res|/τ_prob={row['ratio_prob_max']:.2e} "
                  f"fails worst={row['honest_fail_worst']} prob={row['honest_fail_prob']}/{row['n_checks']}")
    RAW["honest_residuals"] = rows
    return rows


# ---------------------------------------------------------------------------------------------
# 2. detection curves
# ---------------------------------------------------------------------------------------------
def make_tamper(rng, shape, kind, fro):
    m, k = shape
    if kind == "single":
        E = np.zeros(shape); E[rng.integers(m), rng.integers(k)] = 1.0
    elif kind == "rank1":
        E = np.outer(rng.standard_normal(m), rng.standard_normal(k))
    elif kind == "dense":
        E = rng.standard_normal(shape)
    else:
        raise ValueError(kind)
    return E * (fro / np.linalg.norm(E))


def exp_detection(n=512, trials=100):
    rng = np.random.default_rng(SEED + 1)
    m = k = n
    A = rng.standard_normal((m, n)); B = rng.standard_normal((n, k))
    A64, B64, C, _ = honest_product(A, B, "fp32")
    Cfro = np.linalg.norm(C)
    # empirical calibration: c_emp = 10 * max_i |resid_i| / M_i over 50 honest checks
    ratios = []
    for _ in range(50):
        r = rng.choice([-1.0, 1.0], size=(k, 1))
        res = np.abs(C @ r - A64 @ (B64 @ r))
        M = np.abs(A64) @ (np.abs(B64) @ np.abs(r))
        ratios.append((res / M).max())
    c_emp = 10 * max(ratios)

    def tol(mode, r):
        if mode == "emp":
            return c_emp * (np.abs(A64) @ (np.abs(B64) @ np.abs(r)))
        return float_tolerance(A64, B64, r, u=U_FP32, mode=mode, lam=6.0)

    # integer version: int8 A, B
    Ai = rng.integers(-128, 128, size=(m, n)); Bi = rng.integers(-128, 128, size=(n, k))
    Ci = int_matmul_exact(Ai, Bi)
    Cifro = np.linalg.norm(Ci.astype(np.float64))

    rel_sizes = np.logspace(-12, 0, 25)
    curves = {}
    for tk in ["single", "rank1", "dense"]:
        for mode in ["worst", "prob", "emp"]:
            det = []
            for rel in rel_sizes:
                hits = 0
                for _ in range(trials):
                    E = make_tamper(rng, (m, k), tk, rel * Cfro)
                    Ct = (C.astype(np.float32) + E.astype(np.float32)).astype(np.float64)  # stored as fp32
                    r = rng.choice([-1.0, 1.0], size=(k, 1))
                    res = np.abs(Ct @ r - A64 @ (B64 @ r))
                    hits += bool(np.any(res > tol(mode, r)))
                det.append(hits / trials)
            curves[f"float_{mode}_{tk}"] = det
            print(f"[detect] float {mode:5s} {tk:6s}: " + " ".join(f"{d:.2f}" for d in det))
        # integer: tamper rounded to integers (smallest nonzero tamper = 1)
        for name, fn in [("int_R2^20", lambda Ct: freivalds_int(Ai, Bi, Ct, R=2**20, rng=rng)),
                         ("modp61", lambda Ct: freivalds_modp(Ai, Bi, Ct, q=61, rng=rng))]:
            det = []
            for rel in rel_sizes:
                hits, valid = 0, 0
                for _ in range(trials):
                    E = make_tamper(rng, (m, k), tk, rel * Cifro)
                    Ei = np.rint(E).astype(np.int64)
                    if not Ei.any():
                        continue          # tamper rounds to zero: no change, not a tamper
                    valid += 1
                    hits += not fn(Ci + Ei)
                det.append(hits / valid if valid else float("nan"))
            curves[f"{name}_{tk}"] = det
            print(f"[detect] {name:9s} {tk:6s}: " + " ".join(f"{d:.2f}" for d in det))
    RAW["detection"] = dict(n=n, trials=trials, rel_sizes=rel_sizes.tolist(), c_emp=c_emp,
                            curves=curves, int_min_rel=float(1 / Cifro))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, tk in zip(axes, ["single", "rank1", "dense"]):
        for key, lab, st in [("float_worst", "float, worst-case γₙ bound", "-o"),
                             ("float_prob", "float, probabilistic λ√n·u bound", "-s"),
                             ("float_emp", "float, empirical 10× calibration", "-^"),
                             ("int_R2^20", "exact int, r∈[0,2²⁰)", "-x"),
                             ("modp61", "exact F_p, p=2⁶¹−1", "--+")]:
            ax.semilogx(rel_sizes, curves[f"{key}_{tk}"], st, label=lab, ms=4)
        ax.set_title(f"tamper: {tk}  (n={n}, {trials} trials/pt)")
        ax.set_xlabel("‖E‖_F / ‖C‖_F")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("detection probability (k=1)")
    axes[0].legend(fontsize=7, loc="center left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase1_detection.png"), dpi=120)
    return curves


# ---------------------------------------------------------------------------------------------
# 3. k repetitions: empirical false-accept vs bound
# ---------------------------------------------------------------------------------------------
def exp_repetitions(trials=10000):
    rng = np.random.default_rng(SEED + 2)
    A = rng.integers(-128, 128, size=(16, 16)); B = rng.integers(-128, 128, size=(16, 16))
    C = int_matmul_exact(A, B)
    rows = []
    for tk in ["single", "dense"]:
        for scheme, param in [("int", 2), ("int", 4), ("modp", 3), ("modp", 5)]:
            for k in [1, 2, 3, 4, 5]:
                acc = 0
                for _ in range(trials):
                    Ct = C.copy()
                    if tk == "single":
                        Ct[rng.integers(16), rng.integers(16)] += int(rng.integers(1, 1000))
                    else:
                        Ct += rng.integers(-3, 4, size=C.shape)
                        if np.array_equal(Ct, C):
                            Ct[0, 0] += 1
                    if scheme == "int":
                        acc += freivalds_int(A, B, Ct, k=k, R=param, rng=rng)
                        bound = (1 / param) ** k
                    else:
                        acc += freivalds_modp(A, B, Ct, k=k, q=param, rng=rng)
                        bound = (1 / mersenne(param)) ** k
                rows.append(dict(tamper=tk, scheme=scheme, param=param, k=k,
                                 false_accept=acc / trials, bound=bound, trials=trials))
                print(f"[reps] {tk:6s} {scheme}{param} k={k}: false-accept={acc/trials:.5f}  bound={bound:.5f}")
    RAW["repetitions"] = rows
    fig, ax = plt.subplots(figsize=(6, 4))
    for scheme, param, lab in [("int", 2, "Z, R=2"), ("int", 4, "Z, R=4"),
                               ("modp", 3, "F_7"), ("modp", 5, "F_31")]:
        ks = [1, 2, 3, 4, 5]
        ys = [next(r["false_accept"] for r in rows if r["tamper"] == "single" and r["scheme"] == scheme
                   and r["param"] == param and r["k"] == k) for k in ks]
        b = [next(r["bound"] for r in rows if r["scheme"] == scheme and r["param"] == param and r["k"] == k)
             for k in ks]
        l, = ax.semilogy(ks, [max(y, 1e-6) for y in ys], "o-", label=f"{lab} measured (single-entry)")
        ax.semilogy(ks, b, "--", color=l.get_color(), label=f"{lab} bound")
    ax.set_xlabel("repetitions k"); ax.set_ylabel("false-accept rate")
    ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase1_repetitions.png"), dpi=120)


def exp_float_repetitions(trials=200):
    """Float: does repeating with fresh r help against a tamper just below τ? (no, by design)"""
    rng = np.random.default_rng(SEED + 3)
    n = 512
    A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n))
    A64, B64, C, _ = honest_product(A, B, "fp32")
    rows = []
    for frac in [0.5, 0.9, 1.1, 2.0]:
        # rank-1 tamper whose expected residual is frac × median τ
        r0 = rng.choice([-1.0, 1.0], size=(n, 1))
        tau0 = np.median(float_tolerance(A64, B64, r0, mode="worst"))
        for k in [1, 4, 16]:
            hits = 0
            for _ in range(trials):
                Ct = C.copy(); i, j = rng.integers(n), rng.integers(n)
                Ct[i, j] += frac * tau0
                r = rng.choice([-1.0, 1.0], size=(n, k))
                res = np.abs(Ct @ r - A64 @ (B64 @ r))
                hits += bool(np.any(res > float_tolerance(A64, B64, r, mode="worst")))
            rows.append(dict(frac_of_tau=frac, k=k, detection=hits / trials))
            print(f"[float reps] tamper={frac}×τ k={k}: detection={hits/trials:.3f}")
    RAW["float_repetitions"] = rows


# ---------------------------------------------------------------------------------------------
# 4. timing
# ---------------------------------------------------------------------------------------------
def tmed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def exp_timing():
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for n in [512, 1024, 2048, 4096]:
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        Ai = rng.integers(-128, 128, size=(n, n)); Bi = rng.integers(-128, 128, size=(n, n))
        C = A @ B
        Ci = int_matmul_exact(Ai, Bi)
        A64, B64, C64 = A.astype(np.float64), B.astype(np.float64), C.astype(np.float64)

        def ffloat():
            r = rng.choice([-1.0, 1.0], size=(n, 1))
            res = C64 @ r - A64 @ (B64 @ r)
            return np.all(np.abs(res) <= float_tolerance(A64, B64, r))
        row = dict(n=n,
                   gemm_fp32=tmed(lambda: A @ B),
                   gemm_int_exact=tmed(lambda: int_matmul_exact(Ai, Bi)),
                   freivalds_float_with_tol=tmed(ffloat),
                   freivalds_int=tmed(lambda: freivalds_int(Ai, Bi, Ci, rng=rng)),
                   freivalds_modp61=tmed(lambda: freivalds_modp(Ai, Bi, Ci, rng=rng)))
        rows.append(row)
        print("[timing] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    RAW["timing"] = rows


if __name__ == "__main__":
    t0 = time.time()
    print(f"torch {torch.__version__}  numpy {np.__version__}  threads {torch.get_num_threads()}")
    exp_honest_residuals()
    exp_repetitions()
    exp_float_repetitions()
    exp_detection()
    exp_timing()
    RAW["wall_seconds"] = time.time() - t0
    with open(os.path.join(OUT, "phase1_raw.json"), "w") as f:
        json.dump(RAW, f, indent=1)
    print(f"done in {time.time()-t0:.1f}s")
