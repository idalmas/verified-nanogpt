# Phase 1 — exact vs float Freivalds: tolerance, detection, repetitions, cost

Code: `freivalds/core.py`, `experiments/phase1_detection.py`, `experiments/phase1_extra.py`.
Raw outputs: `results/phase1_raw.json`, `results/phase1_stdout.txt`,
`results/phase1_extra_raw.json`, `results/phase1_extra_stdout.txt`.
Plots: `results/phase1_detection.png`, `results/phase1_repetitions.png`.
Tests: `tests/test_core.py` (honest passes / tamper fails for int, F_p, float; F_p blind spot;
empirical false-accept = 1/R; bf16-output rounding needs its own term).
Machine: 4-core CPU, numpy 2.5.3 (OpenBLAS), torch 2.14.1 CPU. Seeds fixed in each script.

## What was built

* `freivalds_int`: exact over ℤ, r ∈ {0..R−1}ⁿ, int64 arithmetic with an overflow assertion.
  Bound: a wrong C is accepted with probability ≤ (1/R)ᵏ for k repetitions.
* `freivalds_modp`: over F_p with p = 2^q − 1 (Mersenne). Matrix–vector products mod p are
  done **exactly** in float64 BLAS by splitting matrix and vector into limbs so every partial
  sum stays below 2⁵³, then recombining with rotations (2^q ≡ 1 mod p). Bound: accept
  probability ≤ (1/p)ᵏ **for errors E ≢ 0 (mod p)**. With p = 2⁶¹−1 and int32 outputs,
  |E| < 2³³ < p, so any wrong C is caught except with probability ≤ 2⁻⁶¹ per repetition.
* `freivalds_float`: Rademacher r, verifier computes in float64, tolerance per row
  τ = γₙ·|A|(|B||r|) (+ output-rounding term u_out(1+γₙ)|A||B||r| when C is rounded to bf16,
  + a bound on the verifier's own float64 error). γₙ = nu/(1−nu), u = 2⁻²⁴. This is the
  textbook bound |fl(AB) − AB| ≤ γₙ|A||B|, valid for any summation order and FMA.
  Also a 'prob' mode (Higham–Mary λ√n·u, λ = 6, not a guarantee) and an empirical mode
  (10× the largest honest residual over 50 calibration checks).

## 1. Honest residuals vs the rigorous bound (m = k = 256, Gaussian inputs, 16 r's per case)

| recipe | n | max honest \|res\|/τ_worst | slack (τ_worst / max res) | max \|res\|/τ_prob (λ=6) | honest fails (worst / prob) |
|---|---|---|---|---|---|
| fp32 | 64 | 1.68e-03 | 596× | 2.24e-03 | 0 / 0 of 4096 |
| fp32 | 256 | 4.01e-04 | 2493× | 1.07e-03 | 0 / 0 of 4096 |
| fp32 | 1024 | 6.83e-05 | 14632× | 3.65e-04 | 0 / 0 of 4096 |
| fp32 | 4096 | 8.02e-06 | 124744× | 8.55e-05 | 0 / 0 of 4096 |
| bf16in_fp32out | 64 | 6.42e-04 | 1557× | 8.56e-04 | 0 / 0 of 4096 |
| bf16in_fp32out | 256 | 1.64e-04 | 6080× | 4.39e-04 | 0 / 0 of 4096 |
| bf16in_fp32out | 1024 | 2.84e-05 | 35226× | 1.51e-04 | 0 / 0 of 4096 |
| bf16in_fp32out | 4096 | 4.46e-06 | 224168× | 4.76e-05 | 0 / 0 of 4096 |
| bf16in_bf16out | 64 | 1.79e-02 | 56× | 1.79e-02 | 0 / 0 of 4096 |
| bf16in_bf16out | 256 | 8.97e-03 | 111× | 8.99e-03 | 0 / 0 of 4096 |
| bf16in_bf16out | 1024 | 4.81e-03 | 208× | 4.87e-03 | 0 / 0 of 4096 |
| bf16in_bf16out | 4096 | 2.26e-03 | 442× | 2.39e-03 | 0 / 0 of 4096 |

* The worst-case bound **never** failed an honest product (0 of 4096 row-checks in every
  case), as the theory says it must.
* It is very loose: real residuals are 600× (n = 64) to 125 000× (n = 4096) below τ in fp32.
  The slack grows ~linearly with n because real rounding errors grow like √n (random-walk)
  while the bound grows like n, and because |A||B| ignores cancellation.
* For bf16 outputs the final rounding to bf16 dominates; slack drops to 56–442×.
* The probabilistic λ√n·u bound is 1–10× tighter here and also had 0 honest failures,
  but it is a statistical model, not a proof.

## 2. Detection curves (n = m = k = 512, k = 1 repetition, 100 trials per point)

Tamper E added to C, with ‖E‖_F / ‖C‖_F on the x-axis (`results/phase1_detection.png`).
For integer checks the tamper is rounded to integers; points where it rounds to zero are
not tampers and are skipped (the smallest possible integer tamper is ‖E‖_F/‖C‖_F ≈ 1.6e−8).

| tamper | float worst-case: largest size never detected → first size detected ≥50% | float λ√n·u | float empirical 10× | exact ℤ (R=2²⁰) and F_p (p=2⁶¹−1) |
|---|---|---|---|---|
| single entry | 3.2e−4 → 1e−3 | 1e−4 → 3.2e−4 | 3.2e−7 → 1e−6 | every non-zero tamper detected (1.00 at all sizes) |
| rank-1 | 1e−3 → 1e−2 | 3.2e−4 → 3.2e−3 | 1e−6 → 1e−5 | 1.00 at all sizes |
| dense noise | 1e−3 → 3.2e−3 | 3.2e−4 → 1e−3 | 1e−6 → 1e−5 | 1.00 at all sizes |

Concretely, for the single-entry case: ‖C‖_F ≈ 11 600 and typical |C_ij| ≈ 22.6, so the
worst-case bound lets an adversary move **one output entry by ~3.7 (≈16% of a typical entry)
with probability 0 of detection** (100/100 trials passed); at ~11.6 (‖E‖/‖C‖ = 1e−3) it was caught 100/100, so the threshold lies in between.
The exact checks catch a change of 1 integer unit in one entry with probability 1 − 2⁻²⁰
(ℤ) or 1 − 2⁻⁶¹ (F_p).

## 3. Repetitions k

Over ℤ with tiny R, a single-entry tamper is accepted iff r_j = 0, so the bound (1/R)ᵏ is
tight; dense tampers are essentially never accepted (0 in 10 000 trials) because every row's
error must be orthogonal to r at once:

| tamper | scheme | k | measured false-accept (10k trials) | bound |
|---|---|---|---|---|
| single | Z, R=2 | 1 | 0.49920 | 0.50000 |
| single | Z, R=2 | 2 | 0.24820 | 0.25000 |
| single | Z, R=2 | 3 | 0.12510 | 0.12500 |
| single | Z, R=2 | 4 | 0.06330 | 0.06250 |
| single | Z, R=2 | 5 | 0.03270 | 0.03125 |
| single | Z, R=4 | 1 | 0.25330 | 0.25000 |
| single | Z, R=4 | 2 | 0.06350 | 0.06250 |
| single | Z, R=4 | 3 | 0.01610 | 0.01562 |
| single | Z, R=4 | 4 | 0.00360 | 0.00391 |
| single | Z, R=4 | 5 | 0.00080 | 0.00098 |
| dense | Z, R=2 | 1 | 0.00000 | 0.50000 |
| dense | Z, R=2 | 2 | 0.00000 | 0.25000 |
| dense | Z, R=2 | 3 | 0.00000 | 0.12500 |
| dense | Z, R=2 | 4 | 0.00000 | 0.06250 |
| dense | Z, R=2 | 5 | 0.00000 | 0.03125 |
| dense | Z, R=4 | 1 | 0.00000 | 0.25000 |
| dense | Z, R=4 | 2 | 0.00000 | 0.06250 |
| dense | Z, R=4 | 3 | 0.00000 | 0.01562 |
| dense | Z, R=4 | 4 | 0.00000 | 0.00391 |
| dense | Z, R=4 | 5 | 0.00000 | 0.00098 |

Over F_p with tiny p (7, 31), the bound holds only for tampers that are not multiples of p;
a tamper that is a multiple of p is invisible and **no number of repetitions helps** — the
reason the real check uses p = 2⁶¹−1 > any int32 error (test
`test_modp_blind_spot_multiple_of_p` shows the same tamper accepted with p = 2³¹−1 and caught
with p = 2⁶¹−1). 20 000 trials per row:

| tamper | p | k | measured false-accept | bound |
|---|---|---|---|---|
| non_multiple | 7 | 1 | 0.14550 | 0.14286 |
| non_multiple | 7 | 2 | 0.01985 | 0.02041 |
| non_multiple | 7 | 3 | 0.00235 | 0.00292 |
| non_multiple | 7 | 4 | 0.00050 | 0.00042 |
| multiple | 7 | 1 | 1.00000 | 1.00000 |
| multiple | 7 | 2 | 1.00000 | 1.00000 |
| multiple | 7 | 3 | 1.00000 | 1.00000 |
| multiple | 7 | 4 | 1.00000 | 1.00000 |
| non_multiple | 31 | 1 | 0.03205 | 0.03226 |
| non_multiple | 31 | 2 | 0.00110 | 0.00104 |
| non_multiple | 31 | 3 | 0.00005 | 0.00003 |
| non_multiple | 31 | 4 | 0.00000 | 0.00000 |
| multiple | 31 | 1 | 1.00000 | 1.00000 |
| multiple | 31 | 2 | 1.00000 | 1.00000 |
| multiple | 31 | 3 | 1.00000 | 1.00000 |
| multiple | 31 | 4 | 1.00000 | 1.00000 |

Float checks: repetition gives nothing against a tamper below the tolerance. Single-entry
tamper of size f·τ (n = 512, 200 trials): detection 0.000 for f = 0.5 and 0.9 at k = 1, 4, 16;
1.000 for f = 1.1 and 2.0 at every k. With Rademacher r, |Er|ᵢ is deterministic for a
single entry, so the float check is a hard threshold.

## 4. Cost (square n×n, median of 3, wall seconds, single process)

| n | fp32 GEMM (s) | exact int GEMM via fp64 (s) | float Freivalds + τ (s) | int Freivalds (s) | F_p Freivalds p=2⁶¹−1 (s) |
|---|---|---|---|---|---|
| 512 | 0.0008 | 0.0046 | 0.0027 | 0.0019 | 0.0071 |
| 1024 | 0.0052 | 0.0297 | 0.0031 | 0.0080 | 0.0094 |
| 2048 | 0.0426 | 0.1348 | 0.0224 | 0.0562 | 0.0760 |
| 4096 | 0.2490 | 0.9823 | 0.1449 | 0.3992 | 0.4965 |

* In MACs Freivalds costs 3n² (plus 2n² for the float tolerance) vs n³ for the GEMM: a
  ~800× (n = 4096) FLOP saving. **In wall time on this CPU the saving is only 0.6–3×**
  (float: 0.145 s vs 0.249 s at n = 4096). Matrix–vector products are memory-bound: the
  verifier must stream A, B and C (in float64 here) through memory once, while sgemm runs
  near peak FLOPs. A verifier that keeps fp32/int8 operands and fuses the passes would be
  faster; the honest lesson is that Freivalds' cost is "read every operand once", which is
  O(data), not O(compute).
* Exact integer GEMMs in numpy have no int BLAS, so the exact-integer *product* is done via
  float64 BLAS (exact for n·127² < 2⁵³); the int and F_p Freivalds checks are 2–2.5× cheaper
  than that product at n = 4096. The F_p check needed a limb-layout optimization: the
  first version (`results/phase1_raw.json` → `timing`) took 2.19 s at n = 4096; after picking the
  cheapest limb split and doing all limbs in one matmul it takes 0.50 s.

## Takeaways

1. Exact Freivalds (ℤ or F_p) is a real proof: zero tolerance, failure probability set by R or
   p and confirmed empirically where measurable (tight for single-entry tampers).
2. Float Freivalds with a *rigorous* tolerance is sound for honest provers but leaves a
   wide band (here ~10⁻³ relative in Frobenius norm, i.e. tens of percent on individual
   entries) where tampering is undetectable with certainty, not just with some probability.
3. Tighter tolerances (probabilistic or empirical) shrink the band by 3× to 1000× but give up
   the guarantee that honest provers always pass, and they depend on hardware/kernels.
