# Phase 8 — random-prime Freivalds; which ops can be recomputed bit-exactly

Code: `freivalds/randprime.py`, `experiments/phase8_misc.py`. Raw: `results/phase8_raw.json`,
log `results/phase8_stdout.txt`.

## B. Random-prime Freivalds (modulus chosen by Fiat–Shamir after commitment)

With a fixed public modulus p, any error that is a multiple of p is invisible (Phase 1 showed
this with small p). If the modulus is instead derived from the commitment, as a prime in
[2³⁰, 2³¹) chosen by hash, the prover can't target it. A nonzero error below 2^B is divisible by
at most B/30 of the ~5×10⁷ primes in range.

| attack: add k·(2³¹−1) to one entry of C | accepted |
|---|---|
| fixed modulus p = 2³¹−1 | 2000/2000 |
| modulus = Fiat–Shamir prime from the commitment | 0/2000 |
| grinding attacker (bets on one prime per attempt, error = multiple of it) | 0/2000 |

This is useful wherever errors can exceed the modulus: wide accumulators, or certificates
without range checks. Phase 7 avoids needing it by bounding every value below 2⁶¹.

## C. Can a verifier recompute non-matmul ops bit-exactly with its own library?

Same formulas, float32, numpy vs torch (CPU), 2²⁰ elements:

| op | bit mismatches | max ulp |
|---|---|---|
| AdamW (only +, −, ×, ÷, √) | 1313 | 65 536 (near-zero outputs) |
| exp | 412 635 | 2 |
| GELU (tanh) | 196 567 | large (near-zero outputs) |
| softmax (exp + row sum) | 654 253 | 10 |
| layer norm (row mean/var) | 392 531 | large (near-zero outputs) |

The AdamW mismatches trace to **torch's sqrt not being correctly rounded on this CPU build**:
6740 of 2²⁰ results differ from the correctly rounded value, while numpy's sqrt is always
correct. torch's sqrt is self-consistent, though: recomputing subsets of 1–656 elements gave 0
mismatches against the full-tensor results. So:
- bit-exact op checks work *within one pinned implementation*, as in Phase 2 and Phase 7;
- across implementations, even "IEEE-exact" ops can differ, and transcendental and reduction
  ops differ massively.

A verifiable recipe has to specify the op implementations (e.g. correctly rounded libm such as
CORE-MATH, fixed reduction order), not just the formulas.
