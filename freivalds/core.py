"""
Core Freivalds checks: exact integer, prime field (mod a Mersenne prime), and float with a
rigorous rounding-error tolerance.

Conventions: A is m×n, B is n×k, the claim is C = A·B (m×k). All functions take numpy arrays.

Exactness of the integer and mod-p paths: every matrix-vector product is computed in float64
on limbs small enough that each partial sum is an integer below 2^53, so BLAS float64 gives the
exact integer result. No floating rounding enters the integer checks.
"""
import math

import numpy as np

U_FP32 = 2.0 ** -24   # unit roundoff, round-to-nearest fp32
U_FP64 = 2.0 ** -53
U_BF16 = 2.0 ** -8

MERSENNE_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61)


# ---------------------------------------------------------------------------------------------
# exact integer helpers
# ---------------------------------------------------------------------------------------------

def int_matmul_exact(A, B):
    """
    Exact integer product of two integer matrices, via float64 BLAS.

    Inputs: A (m×n), B (n×k) integer arrays.
    Outputs: int64 array A·B.
    Raises AssertionError if any partial sum could exceed 2^53 (then float64 would round).
    """
    amax = int(np.abs(A).max(initial=0))
    bmax = int(np.abs(B).max(initial=0))
    assert amax * bmax * A.shape[1] < 2 ** 53, "int_matmul_exact would lose exactness"
    return (A.astype(np.float64) @ B.astype(np.float64)).astype(np.int64)


def _int64_matmul_checked(A, B):
    """Exact int64 matmul (numpy integer path, no BLAS); asserts no int64 overflow."""
    amax = int(np.abs(A).max(initial=0))
    bmax = int(np.abs(B).max(initial=0))
    assert amax * bmax * A.shape[1] < 2 ** 62, "int64 overflow possible"
    return A.astype(np.int64) @ B.astype(np.int64)


def _bits(x):
    return max(1, int(x).bit_length())


def _limbs(M, b, nlimbs):
    """Split a nonnegative int64 array into nlimbs limbs of b bits, low limb first."""
    mask = (1 << b) - 1
    return [((M >> (b * i)) & mask) for i in range(nlimbs)]


# ---------------------------------------------------------------------------------------------
# arithmetic mod a Mersenne prime p = 2^q - 1
# ---------------------------------------------------------------------------------------------

def mersenne(q):
    assert q in MERSENNE_EXPONENTS, f"2^{q}-1 is not a supported Mersenne prime"
    return (1 << q) - 1


def _mul_pow2_mod(x, s, q):
    """x * 2^s mod (2^q - 1), for x an int64 array in [0, p). Uses 2^q ≡ 1 (rotation)."""
    p = (1 << q) - 1
    s %= q
    if s == 0:
        return x.copy()
    xu = x.astype(np.uint64)
    lo = (xu << np.uint64(s)) & np.uint64(p)
    hi = xu >> np.uint64(q - s)
    r = (lo | hi).astype(np.int64)
    r[r == p] = 0
    return r


def matvec_mod(M, v, q=61):
    """
    Exact (M @ v) mod p, p = 2^q - 1.

    Inputs:
        M: integer array (m×n), any sign, |M| < 2^62.
        v: int64 array (n,) or (n, t) with entries in [0, p).
    Outputs: int64 array in [0, p), shape (m,) or (m, t).
    Method: limb-split M and v so every limb product summed over n stays < 2^53, do float64
    matmuls (exact), recombine with shifts mod p.
    """
    p = (1 << q) - 1
    n = M.shape[1]
    budget = 52 - math.ceil(math.log2(max(n, 2)))   # bits allowed for one limb-product
    mmax = int(np.abs(M).max(initial=0))
    bm_signed = _bits(mmax)
    # option 1: keep M signed as a single limb; option 2: reduce mod p and split M into limbs.
    # pick whichever needs fewer float64 matmuls.
    cost1 = math.ceil(q / (budget - bm_signed)) if bm_signed < budget else None
    best2 = min(((math.ceil(q / b) * math.ceil(q / (budget - b)), b) for b in range(1, budget)))
    if cost1 is not None and cost1 <= best2[0]:
        bm = bm_signed
        Mlimbs, Mshift = [M.astype(np.int64)], [0]
    else:
        bm = best2[1]
        Mp = np.mod(M.astype(np.int64), p)
        nl = math.ceil(q / bm)
        Mlimbs = _limbs(Mp, bm, nl)
        Mshift = [bm * i for i in range(nl)]
    bv = budget - bm
    assert bv >= 1
    nv = math.ceil(q / bv)
    vlimbs = _limbs(v.astype(np.int64), bv, nv)
    vshift = [bv * j for j in range(nv)]

    # all v-limbs as columns of one matrix: a single pass over each M-limb (memory-bound)
    t = 1 if v.ndim == 1 else v.shape[1]
    Vcat = np.concatenate([vl.reshape(n, t) for vl in vlimbs], axis=1).astype(np.float64)
    acc = None
    for Ml, ms in zip(Mlimbs, Mshift):
        P = (Ml.astype(np.float64) @ Vcat).astype(np.int64)   # exact integers
        P = np.mod(P, p)
        for j, vs in enumerate(vshift):
            part = _mul_pow2_mod(np.ascontiguousarray(P[:, j * t:(j + 1) * t]), ms + vs, q)
            acc = part if acc is None else np.mod(acc + part, p)
    return acc[:, 0] if v.ndim == 1 else acc


def rand_field(rng, shape, q=61):
    p = (1 << q) - 1
    return rng.integers(0, p, size=shape, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Freivalds checks
# ---------------------------------------------------------------------------------------------

def freivalds_int(A, B, C, k=1, R=2 ** 20, rng=None):
    """
    Exact integer Freivalds over Z with r uniform in {0..R-1}^k (columns), k repetitions.

    Inputs: A (m×n), B (n×k) int8-range ints; C claimed product (int32/int64).
    Outputs: True iff C·r == A·(B·r) exactly for every repetition.
    Soundness: for C ≠ AB, accept prob ≤ (1/R)^k (Schwartz–Zippel, degree 1 per repetition).
    Overflow: checks every float64 partial sum stays below 2^53, so the check is exact.
    """
    rng = np.random.default_rng() if rng is None else rng
    r = rng.integers(0, R, size=(B.shape[1], k), dtype=np.int64)
    Br = _int64_matmul_checked(B, r)
    ABr = _int64_matmul_checked(A, Br)
    Cr = _int64_matmul_checked(C, r)
    return bool(np.array_equal(Cr, ABr))


def freivalds_modp(A, B, C, k=1, q=61, rng=None):
    """
    Freivalds over F_p, p = 2^q - 1, with k independent uniform r ∈ F_p^n.

    Outputs: True iff (C r - A (B r)) ≡ 0 mod p for each repetition.
    Soundness: if C ≢ AB (mod p), accept prob ≤ (1/p)^k. If the integer error E = C - AB
    has |E| < p entrywise (true when C is int32 and AB fits int32, and q = 61), then
    C ≠ AB  ⇒  C ≢ AB mod p, so the bound is unconditional.
    """
    rng = np.random.default_rng() if rng is None else rng
    r = rand_field(rng, (B.shape[1], k), q)
    Br = matvec_mod(B, r, q)
    ABr = matvec_mod(A, Br, q)
    Cr = matvec_mod(C, r, q)
    return bool(np.array_equal(Cr, ABr))


def gamma(n, u):
    """Higham's γ_n = n u / (1 - n u)."""
    assert n * u < 1
    return n * u / (1 - n * u)


def float_tolerance(A, B, r, n_acc=None, u=U_FP32, out_round_u=0.0, mode="worst", lam=6.0):
    """
    Per-row tolerance τ for |C r - A (B r)| where C = fl(A B) was computed honestly.

    Inputs:
        A, B: the GEMM inputs as float64 (exact values of the low-precision inputs).
        r: test vectors (k, t).
        n_acc: accumulation length (default A.shape[1]).
        u: unit roundoff of the accumulator.
        out_round_u: unit roundoff of a final rounding of C to a narrower type (e.g. bf16
            output), 0 if C is kept in the accumulator precision.
        mode: 'worst' -> deterministic bound γ_n (holds for any summation order, with or
              without FMA). 'prob' -> Higham–Mary probabilistic bound λ·√n·u (holds w.h.p.
              under a random-rounding model; not a guarantee).
        lam: λ for 'prob'.
    Outputs: τ (m, t) such that, for the honest C, |C r − A(B r)| ≤ τ (always for 'worst',
        w.h.p. for 'prob'), including a bound on the verifier's own float64 rounding.
    """
    n = A.shape[1] if n_acc is None else n_acc
    k = B.shape[1]
    absA, absB, absr = np.abs(A), np.abs(B), np.abs(r)
    M = absA @ (absB @ absr)                  # |A||B||r|
    if mode == "worst":
        g = gamma(n, u)
    elif mode == "prob":
        g = lam * math.sqrt(n) * u
    else:
        raise ValueError(mode)
    # |fl(AB) - AB| <= g |A||B|; output rounding adds out_round_u (1+g) |A||B|
    g_total = g + out_round_u * (1 + g)
    # verifier float64 error: C r (len k), B r (len k), A(Br) (len n); bound generously with
    # γ in fp64 on |C||r| and |A||B||r| (|C| <= (1+g_total)|A||B| for honest C)
    g64 = gamma(n + k + 2, U_FP64)
    tau = g_total * M + g64 * (2 + g_total) * M * 2
    return tau * (1 + 1e-6) + 1e-300


def freivalds_float(A, B, C, k=1, rng=None, u=U_FP32, out_round_u=0.0, mode="worst",
                    lam=6.0, tau_override=None, return_details=False):
    """
    Float Freivalds with a tolerance. r is Rademacher (±1).

    Inputs: A, B, C float arrays (any dtype; converted to float64 for the verifier).
        tau_override: if given, a callable (resid_abs, M) -> tau used instead of the analytic
            bound (used for empirically calibrated tolerances).
    Outputs: bool accept (and details dict if return_details).
    """
    rng = np.random.default_rng() if rng is None else rng
    A64, B64, C64 = (np.asarray(x, dtype=np.float64) for x in (A, B, C))
    r = rng.choice(np.array([-1.0, 1.0]), size=(B64.shape[1], k))
    resid = C64 @ r - A64 @ (B64 @ r)
    if tau_override is None:
        tau = float_tolerance(A64, B64, r, u=u, out_round_u=out_round_u, mode=mode, lam=lam)
    else:
        tau = tau_override(A64, B64, r)
    ok = bool(np.all(np.abs(resid) <= tau))
    if return_details:
        return ok, {"resid": resid, "tau": tau, "r": r}
    return ok


def freivalds_cost_macs(m, n, k, reps=1, with_tolerance=False):
    """Multiply-adds for a Freivalds check vs. the full product (m n k)."""
    base = reps * (n * k + m * n + m * k)
    if with_tolerance:
        base += reps * (n * k + m * n)
    return base
