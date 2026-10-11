"""
Sketch-verify a recorded training step (freivalds.tinygpt.Trace):

  * every GEMM: float Freivalds with the rigorous worst-case tolerance (separately per GEMM,
    or as one random linear combination across all GEMMs);
  * every non-linear / elementwise op: recompute a random fraction rho of coordinates
    (elementwise ops) or rows (row-wise ops: layer norm, softmax, CE-grad) and compare;
  * layout ops (transpose / head split): recomputed fully (no FLOPs, just copies).

Also: exact mod-p batched Freivalds over a list of integer GEMMs (used by tests and Phase 4).
"""
import math
import time

import numpy as np
import torch

from freivalds.core import U_FP32, U_FP64, gamma, matvec_mod, rand_field
from freivalds.tinygpt import OP_FLOPS_PER_ELEM


# ------------------------------------------------------------------------------------------
# float Freivalds on (batched) torch GEMM records
# ------------------------------------------------------------------------------------------

def _gemm_residual_and_tau(g, r, u=U_FP32):
    """Return residual = C r − A(B r) and worst-case τ (both float64, shape (..., m, t))."""
    A, B, C = g.A.double(), g.B.double(), g.C.double()
    resid = C @ r - A @ (B @ r)
    n, k = A.shape[-1], B.shape[-1]
    M = A.abs() @ (B.abs() @ r.abs())
    gt = gamma(n, u)
    g64 = gamma(n + k + 2, U_FP64)
    tau = (gt + g64 * (2 + gt) * 2) * M * (1 + 1e-6) + 1e-300
    return resid, tau


def _rademacher(shape, gen):
    return (torch.randint(0, 2, shape, generator=gen, dtype=torch.float64) * 2 - 1)


def check_gemm(g, gen, k=1, u=U_FP32):
    r = _rademacher(g.B.shape[:-1][:-1] + (g.B.shape[-1], k), gen)
    resid, tau = _gemm_residual_and_tau(g, r, u)
    return bool((resid.abs() <= tau).all())


def check_gemms_separate(gemms, gen, k=1):
    """Returns list of names of failing GEMMs."""
    return [g.name for g in gemms if not check_gemm(g, gen, k)]


def check_gemms_combined(gemms, gen):
    """
    One float check for all GEMMs: Σ_i α_i s_iᵀ (C_i r_i − A_i B_i r_i) vs Σ_i |α_i| |s_i|ᵀ τ_i.
    α_i, s_i, r_i Rademacher. Returns (ok, value, tolerance).
    """
    tot, tol = 0.0, 0.0
    for g in gemms:
        r = _rademacher(g.B.shape[:-2] + (g.B.shape[-1], 1), gen)
        resid, tau = _gemm_residual_and_tau(g, r)
        s = _rademacher(resid.shape, gen)
        a = float(_rademacher((1,), gen))
        tot += a * float((s * resid).sum())
        tol += float(tau.sum())
    return abs(tot) <= tol, tot, tol


# ------------------------------------------------------------------------------------------
# exact mod-p batched Freivalds for integer GEMMs (list of (A, B, C) numpy int arrays, 2-D)
# ------------------------------------------------------------------------------------------

def modp_combined_check(int_gemms, rng, q=61):
    """
    Accept iff Σ_i α_i s_iᵀ (C_i r_i − A_i (B_i r_i)) ≡ 0 (mod p), p = 2^q − 1, with all
    α_i, s_i, r_i uniform in F_p. For any nonzero error the left side is a nonzero polynomial
    of total degree 3 in the random variables, so false accept ≤ 3/p (Schwartz–Zippel).
    """
    p = (1 << q) - 1
    total = 0
    for A, B, C in int_gemms:
        r = rand_field(rng, (B.shape[1],), q)
        d = (matvec_mod(C, r, q) - matvec_mod(A, matvec_mod(B, r, q), q)) % p
        s = rand_field(rng, (1, d.shape[0]), q)
        sd = int(matvec_mod(s, d, q)[0])
        a = int(rng.integers(0, p))
        total = (total + a * sd) % p
    return total == 0


def modp_separate_check(int_gemms, rng, q=61):
    p = (1 << q) - 1
    bad = []
    for i, (A, B, C) in enumerate(int_gemms):
        r = rand_field(rng, (B.shape[1],), q)
        if not np.array_equal(matvec_mod(C, r, q), matvec_mod(A, matvec_mod(B, r, q), q)):
            bad.append(i)
    return bad


# ------------------------------------------------------------------------------------------
# coordinate sampling for non-linear ops
# ------------------------------------------------------------------------------------------

def _close(a, b, rtol=1e-5, atol_rel=1e-6):
    scale = max(float(b.abs().max()) if b.numel() else 0.0, 1e-30)
    return bool(((a - b).abs() <= rtol * b.abs() + atol_rel * scale).all())


def check_op(o, rho, gen, stats=None):
    """
    Recompute a random fraction rho of op o's outputs. Returns True if consistent.
    For 'elem' ops, sampling unit = element; for 'row' ops = row of last dim.
    stats (dict) accumulates sampled units / exact-bitwise matches / flops.
    """
    out0 = o.outputs[0]
    fl = OP_FLOPS_PER_ELEM[o.fn_name]
    if o.kind == "layout":
        new = o.fn(**o.inputs)
        new = new if isinstance(new, tuple) else (new,)
        return all(torch.equal(a, b) for a, b in zip(new, o.outputs))
    if o.kind == "elem":
        N = out0.numel()
        cnt = max(1, math.ceil(rho * N)) if rho > 0 else 0
        if cnt == 0:
            return True
        idx = torch.randperm(N, generator=gen)[:cnt]
        sub = {k: (v.reshape(-1)[idx] if torch.is_tensor(v) and v.numel() == N else v)
               for k, v in o.inputs.items()}
        new = o.fn(**sub)
        new = new if isinstance(new, tuple) else (new,)
        refs = [t.reshape(-1)[idx] for t in o.outputs]
        units = cnt
        flops = cnt * fl
    else:  # row
        L = out0.shape[-1]
        rows = out0.numel() // L
        cnt = max(1, math.ceil(rho * rows)) if rho > 0 else 0
        if cnt == 0:
            return True
        idx = torch.randperm(rows, generator=gen)[:cnt]
        sub = {k: (v.reshape(rows, L)[idx] if torch.is_tensor(v) and v.numel() == rows * L else v)
               for k, v in o.inputs.items()}
        new = o.fn(**sub)
        new = new if isinstance(new, tuple) else (new,)
        refs = [t.reshape(rows, L)[idx] for t in o.outputs]
        units = cnt
        flops = cnt * L * fl
    ok = all(_close(a, b) for a, b in zip(new, refs))
    if stats is not None:
        stats["units"] = stats.get("units", 0) + units
        stats["exact"] = stats.get("exact", 0) + sum(int(torch.equal(a, b)) for a, b in zip(new, refs))
        stats["outputs"] = stats.get("outputs", 0) + len(refs)
        stats["flops"] = stats.get("flops", 0) + flops
    return ok


# ------------------------------------------------------------------------------------------
# whole-step verification and cost accounting
# ------------------------------------------------------------------------------------------

def step_flops(tr):
    """FLOPs of a full replay: 2·batch·m·n·k per GEMM + op-cost model per element."""
    gem = sum(2 * b * m * n * k for b, m, n, k in (g.mnk for g in tr.gemms))
    ops = sum(OP_FLOPS_PER_ELEM[o.fn_name] * o.outputs[0].numel() for o in tr.ops)
    return gem, ops


def verifier_flops(tr, rho, k=1, combined=False):
    """FLOPs of sketch verification (Freivalds incl. tolerance computation + sampled ops)."""
    gem = 0
    for b, m, n, kk in (g.mnk for g in tr.gemms):
        # C r, B r, A(Br): 2(mk + nk + mn); tolerance |A|(|B||r|): 2(nk + mn)
        gem += k * b * (2 * (m * kk + n * kk + m * n) + 2 * (n * kk + m * n))
        if combined:
            gem += 2 * b * m
    ops = 0
    for o in tr.ops:
        ops += math.ceil(rho * o.outputs[0].numel()) * OP_FLOPS_PER_ELEM[o.fn_name]
    return gem, ops


def verify_step(tr, rho=0.01, k=1, combined=False, seed=0):
    """
    Returns dict: ok, failing (list of record names), timings, sampling stats.
    """
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    if combined:
        ok_g, _, _ = check_gemms_combined(tr.gemms, gen)
        bad_g = [] if ok_g else ["<combined GEMM check>"]
    else:
        bad_g = check_gemms_separate(tr.gemms, gen, k)
    t1 = time.perf_counter()
    stats = {}
    bad_o = [o.name for o in tr.ops if not check_op(o, rho, gen, stats)]
    t2 = time.perf_counter()
    return dict(ok=not bad_g and not bad_o, failing=bad_g + bad_o, t_gemm=t1 - t0, t_ops=t2 - t1,
                stats=stats)
