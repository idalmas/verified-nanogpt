"""
Freivalds modulo a *random* prime chosen after commitment (Fiat–Shamir), instead of a fixed p.

Motivation: with a fixed public modulus p, any integer error E with every entry ≡ 0 (mod p) is
invisible (Phase 1: tampers that are multiples of p are always accepted). When errors can be
larger than p (wide accumulators, big-integer claims, or checks without range bounds) the
prover can exploit that. If instead p is drawn uniformly from the primes in [2^30, 2^31) after
the prover commits, a nonzero error entry |e| < 2^B is divisible by at most B/30 of those
primes, so it vanishes mod p with probability ≤ (B/30) / #primes (≈ 5·10^7 primes, so ~2^-24
for B = 64 per repetition), on top of the usual 1/p Freivalds term.

Arithmetic: p < 2^31, so (M mod p)·(v_limb < 2^16) < 2^47 and n·2^47 < 2^63 for n < 2^16: exact
int64 matmuls, two 16-bit limbs of v.
"""
import hashlib

import numpy as np


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):   # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_from_seed(seed: bytes, lo=1 << 30, hi=1 << 31):
    """Deterministic (Fiat–Shamir) prime in [lo, hi): hash → candidate → next prime."""
    c = lo + int.from_bytes(hashlib.sha256(b"prime" + seed).digest()[:8], "little") % (hi - lo)
    while not is_prime(c):
        c += 1
        if c >= hi:
            c = lo
    return c


def matvec_mod_small(M, v, p):
    """Exact (M @ v) mod p for p < 2^31, M int64 (any size), v in [0, p)."""
    assert p < (1 << 31) and M.shape[1] < (1 << 16)
    Mp = np.mod(M, p).astype(np.int64)
    lo = (v & 0xFFFF).astype(np.int64)
    hi = (v >> 16).astype(np.int64)
    return (np.mod(Mp @ lo, p) + np.mod(np.mod(Mp @ hi, p) * (1 << 16), p)) % p


def freivalds_randprime(A, B, C, commitment: bytes, reps=1):
    """Integer Freivalds; modulus and r both derived from the commitment."""
    for t in range(reps):
        seed = commitment + t.to_bytes(4, "little")
        p = prime_from_seed(seed)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(b"r" + seed).digest()[:8], "little"))
        r = rng.integers(0, p, size=B.shape[1], dtype=np.int64)
        if not np.array_equal(matvec_mod_small(C, r, p), matvec_mod_small(A, matvec_mod_small(B, r, p), p)):
            return False
    return True


def freivalds_fixedprime(A, B, C, rng, p=(1 << 31) - 1):
    r = rng.integers(0, p, size=B.shape[1], dtype=np.int64)
    return bool(np.array_equal(matvec_mod_small(C, r, p), matvec_mod_small(A, matvec_mod_small(B, r, p), p)))
