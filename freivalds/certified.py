"""
Rounding-certificate matmul ("CR-GEMM"): float in, float out, checkable *exactly*.

Spec (what the lab must compute, deterministic and order-independent):
  1. Inputs are rounded to bf16 (RNE). Per row of A (per column of B), entries with
     |x| < 2^(emax − K) are flushed to zero, emax = floor(log2 max|x|) of that row/column
     (a block-floating-point rule, like MX formats). Every kept entry is then an integer
     multiple of the row unit 2^(emax − K − 7), with |integer| < 2^(K + 8).
  2. The exact product E = A_int · B_int (an integer matrix, |E| < 2^58) is rounded ONCE to
     fp32 (round-to-nearest-even) and scaled by the row/column units: C = RNE(E) · 2^(ua_i+ub_j).
     Underflow is part of the spec: if the exact value |E|·2^(ua_i+ub_j) < 2^-126, C = 0
     (flush-to-zero on the exact value, so C is never subnormal). Found necessary when a
     certified training run produced subnormal attention products at step ~200.
     Zeros are +0 (a −0 would be a free covert bit per zero entry; found by fuzzing).
Because the reference value is the exact product rounded once, it does not depend on
summation order, blocking, threads or hardware: any correct kernel produces identical bits.

Certificate: alongside C the lab publishes the integer residual R = E − C_s, where
C_s = C · 2^−(ua_i+ub_j). The verifier, without recomputing the product:
  * recomputes A_int, B_int and the units from A, B (O(size), exact),
  * checks C_s is an integer with |C_s| < 2^59 and that every R_ij lies in the
    round-to-nearest-even interval of C_s (|R| ≤ ½ulp, ¼ulp on the low side of a power of
    two, ties only to even mantissas),
  * checks A_int (B_int r) ≡ C_s r + R r (mod p = 2^61 − 1) for random r (Freivalds).
If all pass, C is *the* spec output except with probability ≤ 1/p per check: |C_s + R − E| < p
by the range checks, so a mismatch over ℤ cannot vanish mod p, and given C_s + R = E the
range check pins C_s = RNE(E) uniquely. There is no tolerance band.
"""
import math

import numpy as np
import torch

from freivalds.core import _mul_pow2_mod, matvec_mod, rand_field

MANT_BITS = 24          # fp32 significand
LIMB = 12               # limb width for the exact product in float64


def spec_K(n):
    """Flush threshold K for inner dimension n, chosen so that |E| < n·2^(2K+16) ≤ 2^57."""
    return int(min(16, (57 - math.ceil(math.log2(max(n, 2))) - 16) // 2))


def quantize(X, axis, K):
    """
    Spec step 1 along `axis` (-1: per row of A; -2: per column of B).
    Returns (Xq float32 tensor, X_int int64 tensor, unit exponent tensor (keepdim, int64)).
    """
    xb = X.detach().float().bfloat16().float()
    ax = xb.abs()
    amax = ax.amax(axis, keepdim=True)
    _, e = torch.frexp(amax)                      # amax = m·2^e, m ∈ [0.5, 1)
    emax = (e - 1).long()                         # floor(log2 amax)
    zero = amax == 0
    keep = ax >= torch.ldexp(torch.ones_like(amax), (emax - K).int())
    xq = torch.where(keep & ~zero, xb, torch.zeros_like(xb))
    unit = torch.where(zero, torch.zeros_like(emax), emax - K - 7)
    xi = torch.ldexp(xq.double(), (-unit).int()).round().long()
    return xq, xi, unit


def _exact_matmul_int(ai, bi):
    """Exact int64 A_int @ B_int for |entries| < 2^24 via 12-bit limbs and float64 matmuls."""
    def split(x):
        s = torch.sign(x)
        a = x.abs()
        return (s * (a >> LIMB)).double(), (s * (a & ((1 << LIMB) - 1))).double()
    a1, a0 = split(ai)
    b1, b0 = split(bi)
    n = ai.shape[-1]
    assert n * (1 << (2 * LIMB)) < 2 ** 53
    p11 = (a1 @ b1).long(); p10 = (a1 @ b0).long(); p01 = (a0 @ b1).long(); p00 = (a0 @ b0).long()
    return (p11 << (2 * LIMB)) + ((p10 + p01) << LIMB) + p00


def _rne_int(E):
    """Round int64 E to a 24-bit significand (RNE). Returns (C_s int64, shift int64)."""
    s = torch.sign(E)
    a = E.abs()
    L = torch.where(a > 0, torch.floor(torch.log2(a.double())).long() + 1, torch.zeros_like(a))
    # log2 in float64 can be off by one near powers of two: fix exactly
    L = torch.where((a > 0) & ((a >> (L - 1).clamp_min(0)) == 0), L - 1, L)
    L = torch.where((a > 0) & ((a >> L.clamp_min(0)) > 0), L + 1, L)
    shift = (L - MANT_BITS).clamp_min(0)
    q = a >> shift
    rem = a - (q << shift)
    half = torch.where(shift > 0, torch.ones_like(shift) << (shift - 1).clamp_min(0), torch.zeros_like(shift))
    up = (shift > 0) & ((rem > half) | ((rem == half) & (q % 2 == 1)))
    q = q + up.long()
    return s * (q << shift), shift


def _below_normal(X, expo):
    """Exact test |X|·2^expo < 2^-126 for int64 X: |X| < 2^t with t = −126 − expo."""
    t = -126 - expo
    a = X.abs()
    return torch.where(t <= 0, a == 0,
                       torch.where(t >= 62, torch.ones_like(a, dtype=torch.bool), a < (torch.ones_like(t) << t.clamp(0, 61))))


def cr_matmul(A, B, K=None):
    """
    Spec CR-GEMM. A (..., m, n), B (..., n, k) float tensors.
    Returns (C float32, R int64, meta) where meta holds K. Raises if C would be subnormal.
    """
    n = A.shape[-1]
    K = spec_K(n) if K is None else K
    _, ai, ua = quantize(A, -1, K)
    _, bi, ub = quantize(B, -2, K)
    E = _exact_matmul_int(ai, bi)
    Cs, _ = _rne_int(E)
    expo = ua + ub                                   # (..., m, 1) + (..., 1, k)
    Cs = torch.where(_below_normal(E, expo), torch.zeros_like(Cs), Cs)
    R = E - Cs
    C = torch.ldexp(Cs.double(), expo.int())
    Cf = C.float()
    nz = Cs != 0
    assert torch.equal(Cf.double()[nz], C[nz]), "fp32 conversion not exact (overflow?)"
    assert bool((Cf.abs()[nz] >= torch.finfo(torch.float32).tiny).all()), "subnormal output: outside spec"
    return Cf, R, dict(K=K)


def matvec_mod_t(M, v, q=61):
    """
    Exact (M @ v) mod p for an int64 torch matrix M (|M| < 2^62) and v (n, t) in [0, p), using
    multithreaded torch float64 matmuls on limbs (every partial sum < 2^53). Same math as
    freivalds.core.matvec_mod, faster for large integer matrices.
    """
    p = (1 << q) - 1
    n = M.shape[-1]
    budget = 52 - math.ceil(math.log2(max(n, 2)))
    mbits = int(M.abs().max()).bit_length() if M.numel() else 1
    if mbits <= budget - 8:
        bm, limbs, shifts = mbits, [M.double()], [0]
    else:
        bm = budget // 2
        Mp = torch.remainder(M, p)
        nl = math.ceil(q / bm)
        mask = (1 << bm) - 1
        limbs = [((Mp >> (bm * i)) & mask).double() for i in range(nl)]
        shifts = [bm * i for i in range(nl)]
    bv = budget - bm
    nv = math.ceil(q / bv)
    vt = torch.from_numpy(np.ascontiguousarray(v.reshape(n, -1)))
    t = vt.shape[1]
    Vcat = torch.cat([((vt >> (bv * j)) & ((1 << bv) - 1)) for j in range(nv)], 1).double()
    acc = None
    for L, ms in zip(limbs, shifts):
        Pm = torch.remainder((L @ Vcat).long(), p).numpy()
        for j in range(nv):
            part = _mul_pow2_mod(np.ascontiguousarray(Pm[:, j * t:(j + 1) * t]), ms + bv * j, q)
            acc = part if acc is None else np.mod(acc + part, p)
    return acc


def certificate_check(A, B, C, R, K, rng, reps=1, q=61):
    """
    Verify (A, B, C, R) against the spec without recomputing the product.
    Returns (ok, reason). Exact: no tolerance anywhere.

    Works directly on C's fp32 bit fields: C = ±mant·2^(eC−150), so in the product's scaled
    units C_s = ±mant·2^sh with sh = eC − 150 − (ua_i + ub_j). For sh ≥ 0, C_s has a 24-bit
    significand and its ulp is 2^sh; for sh < 0, C_s must be an exact integer below 2^24
    (then rounding was exact and R must be 0).
    """
    _, ai, ua = quantize(A, -1, K)
    _, bi, ub = quantize(B, -2, K)
    expo = (ua + ub)
    bits = C.contiguous().view(torch.int32).long()
    eC = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    neg = (bits >> 31) & 1
    zero = (eC == 0) & (frac == 0)
    if bool((zero & (neg == 1)).any()):
        return False, "negative zero (spec requires +0: no covert sign bits)"
    if bool(((eC == 0) & (frac != 0)).any()) or bool((eC == 255).any()):
        return False, "C subnormal/inf/nan (outside spec)"
    mant = torch.where(zero, torch.zeros_like(frac), frac | (1 << 23))
    sh = eC - 150 - expo
    if bool(((sh > 59 - MANT_BITS) & ~zero).any()):
        return False, "C out of range"
    neg_sh = (-sh).clamp(0, 62)
    if bool((((mant & ((torch.ones_like(neg_sh) << neg_sh) - 1)) != 0) & (sh < 0)).any()):
        return False, "C not on the spec grid"
    Cs = torch.where(sh >= 0, mant << sh.clamp(0, 62), mant >> neg_sh)
    Cs = torch.where(neg == 1, -Cs, Cs)
    R = R.long()
    # rounding interval: |R| ≤ ½ulp (¼ulp below a power of two), ties to even
    twice_bound = torch.where(sh > 0, torch.ones_like(sh) << sh.clamp(0, 62), torch.zeros_like(sh))
    low_side = torch.sign(R) == -torch.sign(Cs)
    twice_bound = torch.where((mant == (1 << 23)) & low_side, twice_bound >> 1, twice_bound)
    two_r = 2 * R.abs()
    if bool(((two_r > twice_bound) & ~zero).any()):
        return False, "R outside rounding interval"
    if bool(((two_r == twice_bound) & (twice_bound > 0) & (mant % 2 == 1) & ~zero).any()):
        return False, "tie not rounded to even"
    # underflow rule: C = 0 iff the exact value is below 2^-126; |R| < 2^59 keeps |C_s+R−E| < p
    if bool((R.abs() >= (1 << 59)).any()):
        return False, "R out of range"
    below = _below_normal(Cs + R, expo)
    if bool((zero & ~below).any()):
        return False, "R outside rounding interval"
    if bool((~zero & below).any()):
        return False, "value below normal range must be flushed"
    # Freivalds over F_p: A_int (B_int r) ≡ (C_s + R) r. |C_s + R − E| < 2^60 < p by the checks above.
    D = Cs + R
    ai3 = ai.reshape(-1, ai.shape[-2], ai.shape[-1])
    bi3 = bi.reshape(-1, bi.shape[-2], bi.shape[-1])
    d3 = D.reshape(-1, D.shape[-2], D.shape[-1])
    for b in range(ai3.shape[0]):
        rv = rand_field(rng, (bi3.shape[2], reps), q)
        lhs = matvec_mod_t(ai3[b], matvec_mod_t(bi3[b], rv, q), q)
        rhs = matvec_mod_t(d3[b], rv, q)
        if not np.array_equal(lhs, rhs):
            return False, "Freivalds mod p failed"
    return True, "ok"
