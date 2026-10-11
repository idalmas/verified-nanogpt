# Phase 7 — rounding certificates: float training, exact verification

Idea: keep float training, but define each matmul as "exact product, rounded once". The lab
publishes the integer rounding residual R. The verifier then checks A·B = C + R exactly (mod p)
plus a range check on R, which leaves no tolerance band at all.

Code: `freivalds/certified.py` (spec, prover, reference checker), `freivalds/certified_fast.py`
(fused numba checker), `Config(cr=True)` / `Trace(cr=True)` in `freivalds/tinygpt.py`, the
certificate path in `freivalds/stepverify.py` (`verify_step(..., exact_ops=True)` for bitwise
op recomputation). Experiment: `experiments/phase7_certified.py`. Raw: `results/phase7_raw.json`;
logs `results/phase7_stdout.txt` (final run), `phase7_stdout_a/b/c/d.txt` (earlier runs, see
"Bugs" below). Tests: `tests/test_certified.py` (6 tests).

## The spec (what the lab must compute)

1. Matmul inputs are rounded to bf16. Within each row of A and each column of B, entries
   below 2^−K of that row's/column's largest magnitude are flushed to zero (block floating
   point, like the MX formats). K = 14–16 depending on the inner dimension. Every kept entry is
   then an integer below 2^(K+8) times a power-of-two unit.
2. The exact integer product E = A_int·B_int (|E| < 2^58) is rounded **once** to fp32 (round to
   nearest, ties to even) and scaled back. If the exact value is below 2^−126, the output is 0
   (flush-to-zero on the exact value).
3. The certificate is R = E − C (in integer units), at most half an ulp per entry.

Because the reference is "exact, then round once", the output does not depend on summation
order, blocking, threads or hardware. Any correct kernel gives the same bits. The prover here
computes E with four float64 matmuls on 12-bit limbs; on GPUs, int8 tensor-core slicing (the
Ozaki scheme) would be the natural implementation.

Verifier, without recomputing the product:
- re-derive A_int, B_int from A and B;
- read C's exponent and mantissa bits;
- check R lies in C's rounding interval (½ ulp, ¼ ulp on the low side of a power of two, ties
  to even) and check the underflow rule;
- check A_int·(B_int·r) ≡ (C + R)·r mod 2⁶¹−1.

The range checks bound |C + R − E| < 2⁶¹, so a mismatch cannot vanish mod p. Given C + R = E,
the range check pins C to the unique correctly rounded value. False accept ≤ 2⁻⁶¹ per check
(2⁻³⁰ with the faster small-r option).

## Results

| check | result |
|---|---|
| exactness vs exact rational arithmetic + correct rounding (40 random cases, wide dynamic range) | 0 mismatches in 480 entries |
| inputs flushed by the block-float rule (real training step, all matmul operands) | 3091 of 1.3e7 nonzero entries (2.4e−4) |
| distance to ordinary torch fp32 matmul on the same bf16 inputs | 59.5% bit-identical; median 0 ulp, p99 29 ulp, p99.9 484 ulp (cancellation-heavy entries) |
| forged C' = C ± 1, 2, 3 ulp with a *matching* forged R' (3000 tries) | accepted 0/2500; the true C accepted 500/500 |
| training, 400 steps, same seeds: fp32 vs all 43 matmuls certified | held-out loss 2.394 vs **2.417**; 8/8 certified checkpoints fully verified |
| adversary vs certified dW: naive tamper (backdoor-sized perturbation, honest R) | caught 40/40 steps |
| forged matching certificate (R' = E − C') | caught 40/40 steps |
| minimal forgery: 1 ulp in one entry per dW, with matching R' | caught 40/40 steps |

Cost (square n×n, median of 3 on 4 CPU cores, single matmul):

| n | fp32 matmul | certified prover (prototype) | check, torch | check, fused numba (r ∈ F_p) | fused, small-r (2⁻³⁰ per rep) |
|---|---|---|---|---|---|
| 256 | 0.07 ms | 7.0 ms | 5.7 ms | 1.3 ms (17.6×) | 0.7 ms (10.2×) |
| 1024 | 4.7 ms | 116 ms | 75.5 ms | 24.1 ms (5.1×) | 12.3 ms (2.6×) |
| 2048 | 37.1 ms | 1033 ms | 663 ms | 147 ms (4.0×) | 75.8 ms (2.0×) |
| 4096 | 281 ms | 6218 ms | 3672 ms | 464 ms (1.65×) | 355 ms (1.26×) |

Whole training step, d = 128 (d = 256): plain fp32 replay 28 ms (59 ms); certified prover
477 ms (1005 ms); **complete exact verification** of the certified step, i.e. every certificate
plus every op recomputed bit-exactly at ρ = 1, took 318 ms (657 ms) and passed.

## What this shows

1. **The tolerance band is gone without integer training.** The model trains in float with
   ordinary-looking fp32 matmul outputs (held-out loss within 0.023 of fp32 after 400 steps,
   mostly from bf16 inputs). Yet every matmul is verified exactly: no adversary variant got a
   single step through, including one that forged a self-consistent residual.
2. **Uniqueness is the key property.** Given the exact product, correct rounding leaves no
   choice, so "within tolerance" stops being a set of acceptable outputs and becomes exactly one.
   It is also order-independent, so it doesn't need batch-invariant kernels (unlike Phase 5's
   row replay).
3. **Cost reality on CPU.** The check is O(n²) but integer-heavy. It only overtakes a
   near-peak sgemm around n ≈ 4–8k on this machine, and at tiny-model shapes complete
   verification is ~11× slower than plain fp32 replay (though 1.5× faster than certified
   replay). My prover prototype is 22–95× slower than sgemm, which says little about a real
   GPU implementation (not measured).
4. **Combined with Phase 6**: at realistic batch sizes the float tolerance leaves the adversary
   ~0.4–0.9× the gradient per step. A certified recipe leaves 0.

## Bugs found on the way (kept in the record)

- The first certified training run crashed at step ~200 on subnormal outputs, which my spec
  forbade. Peaked attention again produced tiny products (the same phenomenon as Phase 4's
  false alarms). Fix: an explicit underflow rule in the spec (flush on the exact value) and
  matching verifier checks, plus |R| < 2⁵⁹ so that mismatches stay below p. Test:
  `test_underflow_rule`.
- The first verifier implementations were 25–30× slower than sgemm (many int64 passes, separate
  products for C and R). A fused numba kernel fixed this. Its decisions match the reference on
  240/240 honest and forged cases.

## Limitations

- It changes the recipe in two small ways: bf16 inputs with a block-float flush (2.4e−4 of
  entries), and correctly rounded outputs, which differ from cuBLAS in the last bits.
- It needs exact accumulation, which my CPU prototype does slowly. GPU int8 slicing is the
  plausible fast path, but untested here.
- Non-matmul ops still need a pinned implementation (Phase 8: numpy and torch disagree even on
  sqrt). Within one implementation, bitwise recomputation worked.
- In the commit/open protocol of Phase 3, the range check on R must cover every entry. Done by
  sampling, a single out-of-range R entry is caught only at the sampling rate; closing that
  needs range proofs (bit decomposition, SNARK-style).
