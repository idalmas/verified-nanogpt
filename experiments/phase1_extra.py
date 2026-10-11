"""
Phase 1 follow-ups:
 (a) mod-p false-accept for tampers that are NOT multiples of p (bound 1/p^k should hold),
     vs. tampers that are multiples of p (always accepted: blind spot).
 (b) timing after the matvec_mod limb optimization.
Writes results/phase1_extra_raw.json.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.core import freivalds_modp, freivalds_int, int_matmul_exact, mersenne, float_tolerance

SEED = 4321
RAW = {}
rng = np.random.default_rng(SEED)
A = rng.integers(-128, 128, size=(16, 16)); B = rng.integers(-128, 128, size=(16, 16))
C = int_matmul_exact(A, B)
rows = []
for q in [3, 5]:
    p = mersenne(q)
    for cls in ["non_multiple", "multiple"]:
        for k in [1, 2, 3, 4]:
            trials = 20000
            acc = 0
            for _ in range(trials):
                Ct = C.copy()
                if cls == "non_multiple":
                    d = int(rng.integers(1, 1000))
                    while d % p == 0:
                        d = int(rng.integers(1, 1000))
                else:
                    d = p * int(rng.integers(1, 100))
                Ct[rng.integers(16), rng.integers(16)] += d
                acc += freivalds_modp(A, B, Ct, k=k, q=q, rng=rng)
            rows.append(dict(p=p, tamper=cls, k=k, false_accept=acc / trials,
                             bound=(1 / p) ** k if cls == "non_multiple" else 1.0, trials=trials))
            print(f"[modp] p={p} {cls:12s} k={k}: false-accept={acc/trials:.5f} bound={rows[-1]['bound']:.5f}")
RAW["modp_classes"] = rows

def tmed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

rows = []
for n in [512, 1024, 2048, 4096]:
    A = rng.standard_normal((n, n)).astype(np.float32); B = rng.standard_normal((n, n)).astype(np.float32)
    Ai = rng.integers(-128, 128, size=(n, n)); Bi = rng.integers(-128, 128, size=(n, n))
    Ci = int_matmul_exact(Ai, Bi)
    A64, B64, C64 = A.astype(np.float64), B.astype(np.float64), (A @ B).astype(np.float64)
    def ffloat():
        r = rng.choice([-1.0, 1.0], size=(n, 1))
        res = C64 @ r - A64 @ (B64 @ r)
        return np.all(np.abs(res) <= float_tolerance(A64, B64, r))
    row = dict(n=n, gemm_fp32=tmed(lambda: A @ B), gemm_int_exact=tmed(lambda: int_matmul_exact(Ai, Bi)),
               freivalds_float_with_tol=tmed(ffloat),
               freivalds_int=tmed(lambda: freivalds_int(Ai, Bi, Ci, rng=rng)),
               freivalds_modp61=tmed(lambda: freivalds_modp(Ai, Bi, Ci, rng=rng)))
    rows.append(row)
    print("[timing] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
RAW["timing_optimized"] = rows
json.dump(RAW, open(os.path.join(os.path.dirname(__file__), "..", "results", "phase1_extra_raw.json"), "w"), indent=1)
