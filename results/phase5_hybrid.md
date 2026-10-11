# Phase 5 — hybrid verifier: Freivalds + bitwise row replay, and whole-run detection

Idea (IDEAS.md #17, added tonight): Freivalds catches anything *above* the float tolerance
anywhere; **bitwise replay of a random subset of GEMM rows** catches anything *at all* in the
replayed rows. Together they force the Phase-4 adversary to choose between exceeding the
tolerance (caught by Freivalds) and perturbing only a few rows (caught by replay with
probability ≈ ρ per perturbed row, and with much less steering power).

Code: `freivalds/hybrid.py`, `experiments/phase5_hybrid.py`, `experiments/phase5_analysis.py`.
Raw: `results/phase5_raw.json` (survey, honest, dense), `results/phase5_stdout.txt` (all runs,
including the sparse runs), `results/phase5_analysis.json` (sparse runs parsed from stdout +
formula tables). Note: the Phase-5 job hit its 2-hour limit during the s = 4 run (at step 351
of 400), before writing the sparse runs to json; the s = 16 run never started. The sparse numbers
below come from the stdout log of that job.

## A. Can a verifier replay rows bit-exactly? (MKL fp32 sgemm, this CPU)

Rows of `A[rows] @ B` vs the same rows of the full `A @ B`, 10 random row sets per cell:

| shape m×n×k | 1 row | 2 | 8 | 16 | 32 | 64 rows | 1 vs 4 threads identical |
|---|---|---|---|---|---|---|---|
| 128×1024×512 | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 | no |
| 512×1024×128 | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 | no |
| 128×1024×384 | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 | no |
| 1024×128×512 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | yes |
| 1024×512×128 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | yes |
| 65×1024×128 | 0/10 | 0/10 | 0/10 | 0/10 | 10/10 | 10/10 | no |

Single rows go through a different (GEMV) path and never match; ≥ 32 rows always matched; the
result can depend on the thread count. The hybrid verifier therefore pads every replay to
≥ 64 distinct rows (or replays whole attention (batch, head) slices) and uses the prover's
thread count. This is a property of one library on one CPU, not a guarantee — a real
deployment needs batch-invariant kernels whose per-row result is independent of how many rows
are computed together.

## B. Honest steps

60 hybrid checks (20 training steps × 3 seeds, ρ_ops = ρ_rows = 0.01): **0 failures**.
Replay cost: 2.26e8 MACs per step = **16.5% of the step's GEMM MACs** (1.37e9). The 64-row
padding dominates: the weight-gradient GEMMs have only 64–512 rows, so 64 rows is 12–100% of
them. With batch-invariant kernels (no padding) replay at ρ = 0.01 would cost ~1% of GEMM MACs.

## C. Phase-4 dense sub-tolerance adversary vs the hybrid verifier

30 steps each, 3 verifier seeds per step (90 checks):

| adversary (every row of every dW GEMM perturbed inside τ) | Freivalds detection | replay detection | any |
|---|---|---|---|
| tol_worst | 0.000 | **1.000** | 1.000 |
| tol_bf16out (the one that implanted the backdoor in Phase 4) | 0.000 | **1.000** | 1.000 |

Freivalds alone never fires (as designed by the adversary); bit-exact replay of ≥ 64 rows of a
GEMM in which *every* row was perturbed always finds one.

## D. Best response: perturb only s rows per dW GEMM (inside τ, bf16-output tolerance)

The adversary keeps the s rows with the highest gain τ_i‖G_i‖₂²/‖G_i‖₁ per GEMM.

| s rows / dW GEMM | steps | measured per-step detection (padded replay) | log p(target) at end | greedy success |
|---|---|---|---|---|
| all (Phase 4 tol_bf16out_800) | 800 | 1.000 (C above) | −0.25 / −1.12 (2nd-half means) | 1.0 |
| 16 | — | not run (time limit) | — | — |
| 4 | 351 (stopped) | 1.000 | −8.04 at step 351 | 0.0 |
| 1 | 400 | 1.000 | −7.78 | 0.0 |
| honest (Phase 4) | 400 | — | −9.49 (−9.28 2nd-half mean) | 0.0 |

* Restricting the perturbation to 1 or 4 rows per GEMM removed most of the steering: at every
  checkpoint log p is within 1.7 nats of the honest run (honest −6.7…−9.8 over steps 51–400;
  s = 1: −6.5…−9.3; s = 4: −6.6…−8.1), versus 4–6 nats above honest for the dense bf16
  adversary at the same steps, and the backdoor never appears. Some residual steering may
  remain (s = 1 ends at −7.78 vs honest −9.49), within the size of honest checkpoint-to-checkpoint
  fluctuations.
* Measured detection is 1.0 even for s = 1 *because of the padding*: replaying ≥ 64 rows of a
  128-row GEMM covers half of it. That number does not transfer to large models.

**Unpadded replay (formula, not measured):** with batch-invariant kernels each row is replayed
independently with probability ρ, so p_step = 1 − (1 − ρ)^(perturbed rows), 11 dW GEMMs:

| ρ_rows | s = 1 (11 rows) | s = 4 (44) | s = 16 (176) | all rows (2049) |
|---|---|---|---|---|
| 0.001 | 0.011 | 0.043 | 0.161 | 0.871 |
| 0.01 | 0.105 | 0.357 | 0.829 | 1 − 1.1e−9 |
| 0.05 | 0.431 | 0.895 | 0.9999 | 1.000 |

## E. Whole-run detection with random step auditing

If the auditor checks a random fraction q of steps and the adversary must cheat on c steps,
detection = 1 − (1 − q·p)^c (`phase5_analysis.json → whole_run`):

| strategy (p per audited step) | q = 0.001, c = 100 | q = 0.001, c = 10⁴ | q = 0.01, c = 100 | q = 0.01, c = 10⁴ |
|---|---|---|---|---|
| float-only verifier vs sub-tolerance steering (p = 0, measured) | 0 | 0 | 0 | 0 |
| hybrid vs dense sub-tolerance steering (p = 1.0, measured) | 0.095 | 1.000 | 0.634 | 1.000 |
| hybrid (unpadded, ρ = 0.01) vs s = 4 rows/GEMM (p = 0.357, formula) | 0.035 | 0.972 | 0.301 | 1.000 |
| hybrid (unpadded, ρ = 0.01) vs s = 1 row/GEMM (p = 0.105, formula) | 0.010 | 0.649 | 0.099 | 1.000 |

The backdoor in Phase 4 needed ~500–800 consecutive cheating steps. Under the float-only
verifier no amount of auditing finds it. Under the hybrid it is found with probability
≥ 0.95 at q = 0.01 already for c = 300 dense steps (1 − 0.99³⁰⁰ = 0.951; formula), and the
sparse alternative did not implant anything in 400 steps here.

## Takeaways

* Freivalds + bitwise sampled-row replay turns "within tolerance" from a safe harbour into a
  probabilistic trap: the adversary's budget per step becomes (few rows) × (τ per row).
* Its price is exact reproducibility at row granularity (batch-invariant kernels, pinned
  threads), plus a replay cost of ρ (or the padding) of GEMM FLOPs.
* What remains open: a sparse adversary over many more steps, or one that targets the
  most "sensitive" rows; adaptive choice of ρ for small GEMMs; and the non-GEMM ops, which
  are still only coordinate-sampled with a recompute tolerance (could be made bitwise too —
  in Phase 2 recomputed ops were bit-identical).
