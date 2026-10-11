"""
Phase 8: two small creative checks.

 B. Random-prime Freivalds vs the multiple-of-p attack.
    The adversary knows the verifier works modulo p0 = 2^31 − 1 (fixed) and adds k·p0 to one
    entry of C (a huge, real change). Fixed modulus: always accepted. Modulus chosen by
    Fiat–Shamir from the commitment (here: hash of C) among primes in [2^30, 2^31): the attacker
    would have to guess the prime; we also let the attacker grind: try many tampered C's until
    the FS prime divides its error.
 C. Cross-implementation bit-exactness of the non-GEMM ops (numpy vs torch, same formulas,
    float32): AdamW (+, −, ×, ÷, √ only — IEEE correctly rounded), GELU (tanh), softmax (exp),
    layer norm (mean/var reductions). Which can a verifier recompute bit-exactly on its own
    software stack, and which need a pinned implementation?
Writes results/phase8_raw.json.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freivalds.core import int_matmul_exact
from freivalds.randprime import freivalds_fixedprime, freivalds_randprime, prime_from_seed

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED = 8008
RAW = {}


def exp_randprime(trials=2000, grind=2000):
    rng = np.random.default_rng(SEED)
    A = rng.integers(-128, 128, (32, 64)); B = rng.integers(-128, 128, (64, 16))
    C = int_matmul_exact(A, B)
    p0 = (1 << 31) - 1
    acc_fixed = acc_rand = 0
    for t in range(trials):
        Ct = C.copy(); Ct[rng.integers(32), rng.integers(16)] += p0 * int(rng.integers(1, 1000))
        acc_fixed += freivalds_fixedprime(A, B, Ct, rng, p0)
        acc_rand += freivalds_randprime(A, B, Ct, hashlib.sha256(Ct.tobytes()).digest())
    # grinding attacker: tries tampers e = (random multiple of some prime q in range) until the
    # FS prime (derived from the tampered C) divides e
    succ = 0
    for t in range(grind):
        Ct = C.copy()
        q_guess = prime_from_seed(rng.bytes(16))             # attacker bets on one prime
        Ct[0, 0] += q_guess * int(rng.integers(1, 1000))
        succ += freivalds_randprime(A, B, Ct, hashlib.sha256(Ct.tobytes()).digest())
    RAW["randprime"] = dict(trials=trials, fixed_p_accepts=acc_fixed, random_p_accepts=acc_rand,
                            grind_attempts=grind, grind_successes=succ)
    print(f"[randprime] tamper = k·(2^31−1): fixed-modulus accepts {acc_fixed}/{trials}; "
          f"FS-random-prime accepts {acc_rand}/{trials}; grinding attacker {succ}/{grind}")


def exp_crosslib(n=1 << 20):
    rng = np.random.default_rng(SEED)
    f32 = np.float32
    th = rng.standard_normal(n).astype(f32) * f32(0.02)
    m = rng.standard_normal(n).astype(f32) * f32(1e-3)
    v = np.abs(rng.standard_normal(n).astype(f32)) * f32(1e-6)
    g = rng.standard_normal(n).astype(f32) * f32(1e-3)
    lr, b1, b2, eps, t = f32(3e-3), f32(0.9), f32(0.99), f32(1e-8), 7
    c1, c2 = f32(1 - 0.9 ** t), f32(1 - 0.99 ** t)

    def adam_np():
        m2 = b1 * m + (f32(1) - b1) * g
        v2 = b2 * v + (f32(1) - b2) * g * g
        return th - lr * ((m2 / c1) / (np.sqrt(v2 / c2) + eps))

    def adam_t():
        T = lambda x: torch.from_numpy(x)
        m2 = float(b1) * T(m) + (1 - float(b1)) * T(g)
        m2 = T(b1 * m + (f32(1) - b1) * g)   # same op order as numpy, see note in report
        v2 = T(b2 * v + (f32(1) - b2) * g * g)
        return (T(th) - torch.tensor(lr) * ((m2 / torch.tensor(c1)) / (torch.sqrt(v2 / torch.tensor(c2)) + torch.tensor(eps)))).numpy()

    def adam_t_pure():
        T = lambda x: torch.from_numpy(x)
        s = lambda x: torch.tensor(x, dtype=torch.float32)
        m2 = s(b1) * T(m) + (s(1) - s(b1)) * T(g)
        v2 = s(b2) * T(v) + (s(1) - s(b2)) * T(g) * T(g)
        return (T(th) - s(lr) * ((m2 / s(c1)) / (torch.sqrt(v2 / s(c2)) + s(eps)))).numpy()

    x = rng.standard_normal(n).astype(f32) * f32(3)
    C0 = f32(math.sqrt(2 / math.pi))
    gelu_np = lambda: f32(0.5) * x * (f32(1) + np.tanh(C0 * (x + f32(0.044715) * x ** 3)))
    gelu_t = lambda: (0.5 * torch.from_numpy(x) * (1 + torch.tanh(torch.tensor(C0) * (torch.from_numpy(x) + 0.044715 * torch.from_numpy(x) ** 3)))).numpy()
    S = rng.standard_normal((n // 64, 64)).astype(f32) * f32(4)

    def softmax_np():
        e = np.exp(S - S.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

    def softmax_t():
        s = torch.from_numpy(S); e = torch.exp(s - s.max(-1, keepdim=True).values); return (e / e.sum(-1, keepdim=True)).numpy()
    X = rng.standard_normal((n // 128, 128)).astype(f32)

    def ln_np():
        mu = X.mean(-1, keepdims=True); var = ((X - mu) ** 2).mean(-1, keepdims=True); return (X - mu) / np.sqrt(var + f32(1e-5))

    def ln_t():
        t_ = torch.from_numpy(X); mu = t_.mean(-1, keepdim=True); var = ((t_ - mu) ** 2).mean(-1, keepdim=True)
        return ((t_ - mu) / torch.sqrt(var + 1e-5)).numpy()
    exp_np = lambda: np.exp(x)
    exp_t = lambda: torch.exp(torch.from_numpy(x)).numpy()
    rows = {}
    for name, a, b in [("adamw (+−×÷√ only)", adam_np, adam_t_pure), ("exp", exp_np, exp_t),
                       ("gelu (tanh)", gelu_np, gelu_t), ("softmax (exp + row sum)", softmax_np, softmax_t),
                       ("layernorm (row mean/var)", ln_np, ln_t)]:
        ra, rb = a(), b()
        diff = ra.view(np.int32).astype(np.int64) - rb.view(np.int32).astype(np.int64)
        rows[name] = dict(elements=int(ra.size), bit_mismatches=int((diff != 0).sum()), max_ulp=int(np.abs(diff).max()))
        print(f"[crosslib] {name:26s}: numpy vs torch bit mismatches {rows[name]['bit_mismatches']}/{ra.size}, max {rows[name]['max_ulp']} ulp")
    # diagnose the AdamW mismatch: which library's sqrt is correctly rounded? (fp32 sqrt computed
    # in float64 then rounded is correctly rounded), and is torch's sqrt sample-size invariant?
    a = np.abs(rng.standard_normal(n).astype(f32))
    ref = np.sqrt(a.astype(np.float64)).astype(f32)
    npw = int((np.sqrt(a).view(np.int32) != ref.view(np.int32)).sum())
    full = torch.sqrt(torch.from_numpy(a))
    tw = int((full.numpy().view(np.int32) != ref.view(np.int32)).sum())
    inv = 0
    for k in [1, 7, 16, 656]:
        for _ in range(100):
            idx = torch.randint(n, (k,))
            inv += int((torch.sqrt(torch.from_numpy(a)[idx]).view(torch.int32) != full[idx].view(torch.int32)).sum())
    rows["sqrt_diagnosis"] = dict(numpy_not_correctly_rounded=npw, torch_not_correctly_rounded=tw, elements=n,
                                  torch_subset_vs_full_mismatches=inv)
    print(f"[crosslib] sqrt vs correctly-rounded reference: numpy wrong {npw}/{n}, torch wrong {tw}/{n}; "
          f"torch recompute on subsets (1..656 elems) vs full tensor: {inv} mismatches")
    RAW["crosslib"] = rows


if __name__ == "__main__":
    exp_randprime()
    exp_crosslib()
    json.dump(RAW, open(os.path.join(OUT, "phase8_raw.json"), "w"), indent=1)
