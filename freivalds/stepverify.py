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

from freivalds.core import U_BF16, U_FP32, U_FP64, gamma, matvec_mod, rand_field
from freivalds.tinygpt import OP_FLOPS_PER_ELEM


# ------------------------------------------------------------------------------------------
# float Freivalds on (batched) torch GEMM records
# ------------------------------------------------------------------------------------------

def tol_coef(n, mode="worst", u=U_FP32, kappa=None, lam=6.0):
    """
    Coefficient κ with τ_i = κ · (|A||B||r|)_i for the verifier's tolerance setting:
      worst   — γ_n(u): rigorous for fp32 accumulation and fp32 output.
      bf16out — γ_n(u) + u_bf16 (1 + γ_n): rigorous if the recipe rounds C to bf16.
      prob    — λ √n u (Higham–Mary probabilistic bound; not a guarantee).
      emp     — kappa, calibrated from honest residuals (not a guarantee).
    """
    if mode == "worst":
        return gamma(n, u)
    if mode == "bf16out":
        g = gamma(n, u)
        return g + U_BF16 * (1 + g)
    if mode == "prob":
        return lam * math.sqrt(n) * u
    if mode == "emp":
        return kappa
    raise ValueError(mode)


ETA_FP32 = 2.0 ** -126   # smallest normal fp32: absolute error of a product with a flushed
                         # (FTZ/DAZ) or underflowing operand/result


def _underflow_term(A, B, r, mult, extra_k=0, Babs_r=None):
    """
    Absolute term for gradual/flush-to-zero underflow, which the relative γ_n model ignores:
    each product a·b may be off by ≤ η(|a| + |b| + 1) (a subnormal operand flushed to zero,
    or a subnormal product). Summed over the inner dimension and weighted by |r|:
        η · ( ‖A_i‖₁ ‖r‖₁ + Σ_k (|B||r|)_k + 2(n + extra_k) ‖r‖₁ )   (the 2: products and
    flushed partial sums).
    Found necessary in Phase 4: peaked attention makes softmax outputs subnormal, MKL sgemm
    flushes them, and honest attᵀ·dY GEMMs exceeded the pure γ_n bound.
    """
    n = A.shape[-1]
    r1 = r.abs().sum(-2, keepdim=True)                         # (..., 1, t)
    t1 = A.sum(-1, keepdim=True) * r1                          # (..., m, t); A is |A| here
    t2 = (B @ r.abs() if Babs_r is None else Babs_r).sum(-2, keepdim=True)   # B is |B| here
    return mult * ETA_FP32 * (t1 + t2 + 2 * (n + extra_k) * r1) * 1.01 + 1e-300


def _gemm_residual_and_tau(g, r, u=U_FP32, vprec="fp64", mode="worst", kappa=None):
    """
    Return residual = C r − A(B r) and a rigorous τ with |residual| ≤ τ for an honest
    C = fl(A B) computed with unit roundoff u (shape (..., m, t)).

    vprec='fp64': verifier computes in float64; τ = γ_n(u)·|A||B||r| + float64 slack.
    vprec='fp32': verifier computes in float32 (half the memory traffic). τ then also bounds
        the verifier's own rounding: fl(Br) (γ_k), fl(A·w) (γ_n), fl(Cr) (γ_k) and the final
        subtraction (u), using |C| ≤ (1+γ_n)|A||B| for the honest C; M = |A|(|B||r|) is itself
        computed in fp32 and inflated by (1 + 2γ_{n+k}) to stay an upper bound.
    """
    n, k = g.A.shape[-1], g.B.shape[-1]
    if vprec == "fp64":
        A, B, C = g.A.double(), g.B.double(), g.C.double()
        r = r.double()
        resid = C @ r - A @ (B @ r)
        Aa, Ba = A.abs(), B.abs()
        Bar = Ba @ r.abs()
        M = Aa @ Bar
        gt = tol_coef(n, mode, u, kappa)
        g64 = gamma(n + k + 2, U_FP64)
        tau = (gt + g64 * (2 + gt) * 2) * M * (1 + 1e-6) + _underflow_term(Aa, Ba, r, 1.0, Babs_r=Bar)
        return resid, tau
    A, B, C = g.A.float(), g.B.float(), g.C.float()
    r = r.float()
    resid = C @ r - A @ (B @ r)
    Aa, Ba = A.abs(), B.abs()
    Bar = Ba @ r.abs()
    M = Aa @ Bar
    gn, gk = gamma(n, u), gamma(k, u)
    coef = gn + gk * (1 + gn) + gn * (1 + gk) + gk + u * (3 + 2 * gn)
    tau = coef * M * (1 + 2 * gamma(n + k, u)) * 1.01
    # underflow: prover's GEMM plus the verifier's three fp32 products (Br, A·w, C·r)
    tau = tau.double() + _underflow_term(Aa, Ba, r, 3.0, extra_k=k, Babs_r=Bar).double()
    return resid.double(), tau


def _rademacher(shape, gen):
    return (torch.randint(0, 2, shape, generator=gen, dtype=torch.float64) * 2 - 1)


def check_gemm(g, gen, k=1, u=U_FP32, vprec="fp64", mode="worst", kappa=None):
    r = _rademacher(g.B.shape[:-1][:-1] + (g.B.shape[-1], k), gen)
    resid, tau = _gemm_residual_and_tau(g, r, u, vprec, mode, kappa)
    return bool((resid.abs() <= tau).all())


def check_gemms_separate(gemms, gen, k=1, vprec="fp64", mode="worst", kappa=None):
    """Returns list of names of failing GEMMs."""
    return [g.name for g in gemms if not check_gemm(g, gen, k, vprec=vprec, mode=mode, kappa=kappa)]


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

EXACT_OPS = {"on": False}   # set by verify_step(exact_ops=True): bitwise op recomputation


def _close(a, b, rtol=1e-5, atol_rel=1e-6):
    if EXACT_OPS["on"]:
        # bit-level equality: torch.equal treats −0 == +0 and fails on NaN
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype == torch.float32:
            return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
        if a.dtype == torch.float64:
            return bool(torch.equal(a.view(torch.int64), b.view(torch.int64)))
        return bool(torch.equal(a, b))
    scale = max(float(b.abs().max()) if b.numel() else 0.0, 1e-30)
    return bool(((a - b).abs() <= rtol * b.abs() + atol_rel * scale).all())


_LAYOUT_MAPS = {}


def _layout_map(o):
    """
    Index map of a layout op: out.flatten()[i] == concat(inputs.flatten())[map[i]].
    Depends only on the op and the shapes, so the verifier builds it once per architecture
    (cached) by running the layout function on index tensors.
    """
    names = [k for k, v in o.inputs.items() if torch.is_tensor(v)]
    key = (o.name, tuple(tuple(o.inputs[k].shape) for k in names), tuple(o.outputs[0].shape))
    if key not in _LAYOUT_MAPS:
        off, idx_inputs = 0, {}
        for k in names:
            n = o.inputs[k].numel()
            idx_inputs[k] = torch.arange(off, off + n, dtype=torch.float64).view(o.inputs[k].shape)
            off += n
        m = o.fn(**idx_inputs)
        _LAYOUT_MAPS[key] = (names, m.reshape(-1).long(),
                             torch.tensor([0] + [o.inputs[k].numel() for k in names]).cumsum(0))
    return _LAYOUT_MAPS[key]


def _check_layout(o, rho, gen):
    if rho <= 0:
        return True
    names, m, offs = _layout_map(o)
    out = o.outputs[0].reshape(-1)
    if rho >= 1.0:
        idx = torch.arange(out.numel())
    else:
        idx = torch.randint(out.numel(), (max(1, math.ceil(rho * out.numel())),), generator=gen)
    src = m[idx]
    which = torch.searchsorted(offs, src, right=True) - 1
    vals = torch.empty(idx.numel(), dtype=out.dtype)
    for j, k in enumerate(names):
        sel = which == j
        if sel.any():
            vals[sel] = o.inputs[k].reshape(-1)[src[sel] - offs[j]]
    return bool(torch.equal(vals, out[idx]))


def check_op(o, rho, gen, stats=None):
    """
    Recompute a random fraction rho of op o's outputs. Returns True if consistent.
    For 'elem' ops, sampling unit = element; for 'row' ops = row of last dim.
    stats (dict) accumulates sampled units / exact-bitwise matches / flops.
    """
    out0 = o.outputs[0]
    fl = OP_FLOPS_PER_ELEM[o.fn_name]
    if o.kind == "layout":
        return _check_layout(o, rho, gen)
    if o.kind == "elem":
        N = out0.numel()
        cnt = max(1, math.ceil(rho * N)) if rho > 0 else 0
        if cnt == 0:
            return True
        idx = torch.randint(N, (cnt,), generator=gen)   # with replacement
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
        idx = torch.randint(rows, (cnt,), generator=gen)
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


def verify_step(tr, rho=0.01, k=1, combined=False, seed=0, vprec="fp64", mode="worst", kappa=None,
                exact_ops=False):
    """
    Returns dict: ok, failing (list of record names), timings, sampling stats.
    Float GEMMs use tolerance `mode` (see tol_coef); int GEMMs (tr.int_gemms) are checked
    exactly over F_p, p = 2^61 − 1.
    """
    gen = torch.Generator().manual_seed(seed)
    EXACT_OPS["on"] = exact_ops
    t0 = time.perf_counter()
    cr = [g for g in tr.gemms if g.R is not None]
    if cr:
        from freivalds.certified import certificate_check
        from freivalds.certified_fast import certificate_check_fast
        crng = np.random.default_rng(int(torch.randint(2**62, (1,), generator=gen)))

        def cert_ok(g):
            # fused numba kernel (r uniform in F_p), slice by slice for batched (attention) GEMMs
            A3 = g.A.reshape(-1, *g.A.shape[-2:]).numpy(); B3 = g.B.reshape(-1, *g.B.shape[-2:]).numpy()
            C3 = g.C.reshape(-1, *g.C.shape[-2:]).numpy(); R3 = g.R.reshape(-1, *g.R.shape[-2:]).numpy()
            return all(certificate_check_fast(A3[b], B3[b], C3[b], R3[b], g.K, crng)[0] for b in range(A3.shape[0]))
        bad_cr = [g.name for g in cr if not cert_ok(g)]
        tr_gemms = [g for g in tr.gemms if g.R is None]
    else:
        bad_cr, tr_gemms = [], tr.gemms
    if combined:
        ok_g, _, _ = check_gemms_combined(tr_gemms, gen)
        bad_g = [] if ok_g else ["<combined GEMM check>"]
    else:
        bad_g = check_gemms_separate(tr_gemms, gen, k, vprec, mode, kappa)
    bad_g = bad_cr + bad_g
    if tr.int_gemms:
        rng = np.random.default_rng(int(torch.randint(2**62, (1,), generator=gen)))
        ig = [(g.A.numpy(), g.B.numpy(), g.C.numpy()) for g in tr.int_gemms]
        bad_g += [tr.int_gemms[i].name for i in modp_separate_check(ig, rng)]
    t1 = time.perf_counter()
    stats = {}
    bad_o = [o.name for o in tr.ops if not check_op(o, rho, gen, stats)]
    t2 = time.perf_counter()
    EXACT_OPS["on"] = False
    return dict(ok=not bad_g and not bad_o, failing=bad_g + bad_o, t_gemm=t1 - t0, t_ops=t2 - t1,
                stats=stats)
