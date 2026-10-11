"""
Hybrid verifier: float Freivalds (catches anything above the tolerance, in any row) plus
**bitwise replay of a random subset of GEMM rows** (catches anything, however small, in the
replayed rows).

Why: Phase 4 shows that an adversary can perturb every row of a weight-gradient GEMM by up
to the Freivalds tolerance without detection. Bitwise row replay closes the sub-tolerance
band for every row it touches; a sub-tolerance perturbation spread over many rows is then
caught with probability 1 − (1 − ρ)^(#perturbed rows). What is left to the adversary is to
perturb only a few rows per step (low steering capacity, detection ≈ ρ·rows per step).

Requirement: replaying rows must reproduce the prover's bits. MKL fp32 GEMM on this CPU is
row-batch-invariant only when ≥ ~32 rows are computed together and the thread count is the
same (measured; see results/phase5_hybrid.md). So the verifier pads every replay to at
least `min_rows` rows. This is a property of the kernel, not of the math: a production
system needs batch-invariant kernels (fixed reduction order independent of m).
"""
import math

import torch


def replay_rows_check(g, rho, gen, min_rows=64):
    """
    Bitwise-replay a fraction rho of the rows of GEMM record g (C = A @ B).
    2-D GEMMs: sample rows, pad the sample to ≥ min_rows distinct rows, recompute
    A[rows] @ B and compare bitwise with C[rows]. Batched GEMMs (attention): sample
    (batch, head) slices with probability rho and recompute whole slices.
    Returns (ok, rows_replayed, macs).
    """
    A, B, C = g.A, g.B, g.C
    if A.dim() == 2:
        m = A.shape[0]
        cnt = max(1, math.ceil(rho * m)) if rho > 0 else 0
        if cnt == 0:
            return True, 0, 0
        rows = torch.randint(m, (cnt,), generator=gen).unique()
        if rows.numel() < min(min_rows, m):
            extra = torch.randperm(m, generator=gen)
            extra = extra[~torch.isin(extra, rows)][: min(min_rows, m) - rows.numel()]
            rows = torch.cat([rows, extra]).sort().values
        ok = torch.equal(A[rows] @ B, C[rows])
        return bool(ok), int(rows.numel()), int(rows.numel()) * A.shape[1] * B.shape[1]
    A3 = A.reshape(-1, A.shape[-2], A.shape[-1])
    B3 = B.reshape(-1, B.shape[-2], B.shape[-1])
    C3 = C.reshape(-1, C.shape[-2], C.shape[-1])
    nb = A3.shape[0]
    cnt = max(1, math.ceil(rho * nb)) if rho > 0 else 0
    if cnt == 0:
        return True, 0, 0
    sl = torch.randint(nb, (cnt,), generator=gen).unique()
    ok = torch.equal(A3[sl] @ B3[sl], C3[sl])
    return bool(ok), int(sl.numel()) * A3.shape[1], int(sl.numel()) * A3.shape[1] * A3.shape[2] * B3.shape[2]


def verify_step_hybrid(tr, rho=0.01, rho_rows=0.01, seed=0, mode="worst", kappa=None, min_rows=64):
    """Phase-2 verify_step (Freivalds + op sampling) plus bitwise row replay on every GEMM."""
    from freivalds.stepverify import verify_step
    res = verify_step(tr, rho=rho, seed=seed, mode=mode, kappa=kappa)
    gen = torch.Generator().manual_seed(10_000 + seed)
    bad, macs = [], 0
    for g in tr.gemms:
        ok, _, mc = replay_rows_check(g, rho_rows, gen, min_rows)
        macs += mc
        if not ok:
            bad.append(g.name)
    res["replay_failing"] = bad
    res["replay_macs"] = macs
    res["ok"] = res["ok"] and not bad
    return res
