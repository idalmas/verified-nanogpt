# REPORT — Freivalds-based verification of LLM training steps (overnight, CPU only)

Branch `freivalds-overnight` (pushed to `claude/modest-newton-csbrh2`; see LOG.md for why).
Library `freivalds/`, experiments `experiments/`, tests `tests/` (**41 passed**), raw outputs
and per-phase write-ups in `results/`, journal `LOG.md`, ideas and post-mortem `IDEAS.md`.

## 1. Summary

Freivalds' check works as advertised when arithmetic is exact. Over the integers or F_p
(p = 2⁶¹−1), a single changed accumulator entry is caught with probability 1 − 2⁻⁶¹. In a
training loop with int8 weight-gradient GEMMs, the smallest possible tamper (±1 per dW per
step) was rejected on 400/400 steps. With floats, the verifier needs a tolerance, and that
tolerance is the attack surface. Using the rigorous worst-case bound required for a recipe
that stores weight gradients in bf16, an adversary perturbed every weight-gradient GEMM inside
the tolerance and implanted a trigger→output backdoor in a tiny GPT. It reached 100% greedy
success in both 800-step runs while **all 1600 steps passed Freivalds on every GEMM plus
coordinate sampling of every non-linear op**, and clean loss matched honest training. With the
tighter fp32 bounds, the backdoor did not appear at this scale (a weak, consistent nudge only).
But the fp32 worst-case budget grows about linearly with tokens per step, and a follow-up
(Phase 6) **measured** it up to 2²⁰ tokens: it passes the budget that implanted the backdoor at
~65k tokens per step, and at ~1M tokens it lets the adversary change each weight-gradient entry by
0.36–0.85× the gradient's own RMS while every check still passes. Three of my "obviously fine" constructions
failed when tested:
- Batching float checks into one random linear combination hides 100×-RMS tampers in gradient
  GEMMs.
- The textbook γₙ bound produced honest false alarms once attention outputs became subnormal
  (FTZ/DAZ).
- A sublinear verifier with plain Merkle commitments catches a lying sketch only at the
  row-opening rate, and Fiat–Shamir makes sparse lies grindable.

The cheapest fix I found for the float tolerance gap is a hybrid: Freivalds for large errors
plus bit-exact replay of sampled GEMM rows. It caught the dense sub-tolerance adversary on
every step. It depends on batch-invariant kernels, which this CPU's MKL provides only for
≥32-row replays at a fixed thread count. On cost, Freivalds plus 1% sampling cuts verifier
FLOPs 39–130× vs replay (growing with width), but CPU wall time only 0.6–2.4×, because the
checks are memory-bound and the verifier must read every activation.

## 2. Phases

### Phase 0 — ideas
`IDEAS.md`: 16 ideas, ranked, plus 2 added during the night, and a post-mortem table.

### Phase 1 — exact vs float Freivalds ([results/phase1_detection.md](results/phase1_detection.md))
Built `freivalds/core.py` with exact ℤ checks, exact F_p checks (limb-split float64 BLAS, so
every partial sum is exact), and float checks with a rigorous γₙ tolerance, plus probabilistic
and empirical tolerance modes. Raw: `phase1_raw.json`, `phase1_extra_raw.json`, plots
`phase1_detection.png`, `phase1_repetitions.png`.

| result | value |
|---|---|
| honest fp32 residual vs rigorous bound | 0 failures in 4096 checks per size; bound is 596× (n=64) to 124 744× (n=4096) loose |
| bf16 inputs, bf16 output | 0 failures; bound 56×–442× loose (output rounding dominates) |
| float detection threshold, single-entry tamper, n = 512 (worst-case bound) | ‖E‖/‖C‖ ≤ 3.2e−4 never detected (≈16% of a typical entry), 1e−3 always |
| same, empirical 10× calibration | 3.2e−7 never, 1e−6 always |
| exact ℤ / F_p | every non-zero integer tamper detected at every size (100/100 trials per point) |
| false-accept vs repetitions (ℤ, R = 2, single entry) | 0.499, 0.248, 0.125, 0.063, 0.033 for k = 1..5 (bound 2⁻ᵏ) |
| F_p blind spot | tampers ≡ 0 mod p always accepted (p = 7, 31, any k); bound holds for the rest |
| float repetition | tamper at 0.9·τ: detection 0.000 for k = 1, 4, 16; at 1.1·τ: 1.000 |
| cost, n = 4096 (wall) | fp32 GEMM 0.249 s; float Freivalds + τ 0.145 s; int Freivalds 0.40 s; F_p 0.50 s (exact int GEMM via fp64: 0.98 s) |

Tests: `tests/test_core.py`, 10 tests, pass.

### Phase 2 — verifying a whole step ([results/phase2_step_verification.md](results/phase2_step_verification.md))
Built `freivalds/tinygpt.py`: a hand-written forward/backward/AdamW for a char-level GPT. It
records all 43 GEMMs (forward, dX, dW, attention, embeddings) and 85 non-GEMM ops, and its
gradients match autograd to 3e−8. Built `freivalds/stepverify.py`: per-GEMM float Freivalds, a
float combined check, an exact mod-p combined check, and row/element sampling for ops. Raw:
`phase2_raw.json`, `phase2_stdout_part{1,2}.txt`, `phase2_cost_stdout.txt`.

| result | value |
|---|---|
| honest 300-step training, verified every 25 steps | loss 4.22 → 2.43, 13/13 verified |
| mod-p combined check, false accept (p = 31 / 127) | 0.093 / 0.023 (bound 3/p = 0.097 / 0.024) |
| float combined check | blind up to 100× RMS tampers in 3 of 4 GEMMs tested; per-GEMM check catches them from 0.001–0.03× RMS |
| op sampling, single tampered coordinate, ρ = 0.001 / 0.01 / 0.1 | detection 0.000–0.003 / 0.003–0.013 / 0.093–0.110 (theory ≈ ρ) |
| dense ×(1+1e−3) tamper of an op | 1.000 at every ρ; ×(1+1e−7) (below rtol): 0.000 |
| dense ×(1+1e−3) tamper of weight gradient h1.fc2.dW | detected only 5% (inside tolerance) |
| verifier vs replay FLOPs (ρ = 0.01) | 39× fewer (d = 128), 74× (256), 130× (512) |
| verifier vs replay wall time (ρ = 0.01, rigorous fp32 verifier) | 1.35× (d = 128), 0.80× (256), 2.42× (512); at ρ = 0.1: 0.43–0.94× |

Tests: `tests/test_stepverify.py`, 21 tests, pass. They cover backward vs autograd, honest
passes, tampers in each GEMM type and each op type, the layout-cache regression, int8-dW
exact checks, and the underflow regression.

### Phase 3 — commit → Fiat–Shamir → sketches ([results/phase3_protocol.md](results/phase3_protocol.md))
Built `freivalds/protocol.py`:
- the prover commits each step's 187 tensors with SHA-256 Merkle trees, one leaf per row;
- challenges come from SHAKE-256 of the commitment root;
- per GEMM the prover sends float64 sketches (Br, Cr, A(Br), and their absolute-value
  versions), plus Merkle openings of sampled rows;
- the verifier checks openings and the Freivalds relation without full tensors.

There is also an exact F_p version for int GEMMs. Raw: `phase3_raw.json`, `phase3_stdout.txt`.

| result (d = 128, 92 MB of step tensors) | rate 0.01 | 0.05 | 0.2 |
|---|---|---|---|
| proof size | 7.0 MB | 22.3 MB | 65.2 MB |
| fraction of data bytes the verifier sees | 2.9% | 12.3% | 38.1% |
| verifier time | 120 ms | 524 ms | 1938 ms |
| prover commit (hashing) | 399 ms (step: 42 ms) | 415 ms | 395 ms |

| attack | outcome |
|---|---|
| tamper after commitment | caught 5/5 (Merkle) |
| choose C after seeing r (perturbation ⟂ r), with Fiat–Shamir | caught for 3/3 GEMMs |
| same without Fiat–Shamir | passes for 3/3, even at the size of the tensor's largest entry |
| lying sketch hiding one wrong row | caught w.p. 0.013 / 0.051 / 0.190 at row rate 0.01 / 0.05 / 0.2 |
| grinding to dodge openings, rate 0.05 | 1.1 / 1.8 / 221 re-commits for 1 / 10 / 100 lied rows |

Tests: `tests/test_protocol.py`, 7 tests, pass.

### Phase 4 — steering inside the tolerance ([results/phase4_adversary.md](results/phase4_adversary.md))
The adversary adds ‖E_i‖₁ = 0.9·τ_i per row to every weight-gradient GEMM, in the direction of
a backdoor gradient (trigger `$&$` → `QZQZ`). With ±1 test vectors this passes for every r.
Raw: `phase4_raw.json`, `phase4_stdout_*.txt`, plot `phase4_backdoor.png`.

| run (1024 tokens/step unless noted) | steps passed | final greedy backdoor | mean log p(target), 2nd half | clean loss |
|---|---|---|---|---|
| honest (400 / 800 steps) | 400/400, 800/800 | 0.000, 0.000 | −9.28, −9.54 | 2.338, 2.098 |
| unbounded poisoning | **0/400** | 1.000 | −0.00 | 2.517 |
| **bf16-output rigorous tolerance, 800 steps, 2 runs** | **800/800, 800/800** | **1.000, 1.000** | −0.25, −1.12 | 2.095, 2.101 |
| bf16-output, 400 steps (fixed verifier) | 400/400 | 0.000 (log p −1.96 at step 400) | −3.36 | 2.347 |
| fp32 worst-case (400 / 800 steps / 4096 tokens) | all passed | 0.000 | −8.67 / −7.68 / −7.49 (honest: −9.28 / −9.54 / −8.54) | ≈ honest |
| probabilistic bound / empirical 10× / Adam rtol 1e−5 | all passed | 0.000 | −9.95 / −9.93 / −9.57 | ≈ honest |
| **int8 dW + exact F_p, ±1 tamper per dW per step** | **0/400** | 0.000 | −7.54 | 2.349 |

Adversary budget per entry relative to the gradient: bf16-output ≈ 1.5–2.3e−2 at every batch
size; fp32 worst-case 5.9e−5 (256 tokens) → 4.6e−3 (16 384 tokens), growing ~linearly.

### Phase 5 — hybrid verifier and whole-run detection ([results/phase5_hybrid.md](results/phase5_hybrid.md))
Built `freivalds/hybrid.py`: Freivalds plus bitwise replay of sampled GEMM rows, padded to ≥64
rows. Tests: `tests/test_hybrid.py`, 3 tests, pass. Raw: `phase5_raw.json`,
`phase5_stdout.txt`, `phase5_analysis.json`.

| result | value |
|---|---|
| MKL row replay bit-identical | never for 1 row; always for ≥32 rows; for 2–16 rows, only on 2 of 6 shapes; 1 vs 4 threads differ on 4 of 6 shapes |
| honest hybrid checks | 0/60 failures; replay cost 16.5% of GEMM MACs (padding) |
| Phase-4 dense sub-tolerance adversaries (worst, bf16out) | Freivalds detection 0.000, replay 1.000 (90 checks each) |
| sparse best response, 1 or 4 rows per dW GEMM | steering mostly gone (log p within 1.7 nats of honest at every checkpoint, no backdoor in 400 / 351 steps); measured detection 1.0 because of padding |
| unpadded replay, ρ = 0.01 (formula) | p_step = 0.105 (1 row/GEMM), 0.357 (4), 0.829 (16) |
| whole run, audit q = 1% of steps (formula) | float-only: 0 for any number of cheating steps; hybrid vs dense cheating over 300 steps: 0.951 |

### Phase 6 (follow-up) — measured budget vs tokens per step ([results/phase6_budget_scaling.md](results/phase6_budget_scaling.md))
Streamed up to 2²⁰ tokens through the model in 1024-token microbatches, accumulating Xᵀ·dY and
|X|ᵀ·|dY| for every weight GEMM. Raw: `phase6_raw.json`, `phase6_stdout.txt`, plot
`phase6_budget.png`. The numbers are the adversary's per-entry budget divided by the gradient's
RMS; init / after 300 steps.

| tokens per step | fp32 worst-case | bf16-output | probabilistic |
|---|---|---|---|
| 1 024 | 3.3e−4 / 5.8e−4 | 2.2e−2 / 3.8e−2 (implanted the backdoor in Phase 4) | 6.3e−5 / 1.1e−4 |
| 65 536 | 2.1e−2 / 4.8e−2 | 4.2e−2 / 9.6e−2 | 4.9e−4 / 1.1e−3 |
| 1 048 576 | **3.6e−1 / 8.5e−1** | 3.8e−1 / 9.0e−1 | 2.0e−3 / 4.6e−3 |

## 3. Verdicts

**(a) Float vs exact-integer Freivalds.** Exact wins decisively for verification. Exact checks
have zero tolerance, a provable failure probability (1/p per check, confirmed where measurable),
are cheap (F_p check 0.5 s vs 0.98 s exact product at n = 4096 on CPU), and left the adversary
no room at all in Phase 4. Float checks with a rigorous tolerance are sound for honest provers
only if the bound models the actual kernel; my first "rigorous" bound missed FTZ underflow.
And they leave a deterministic, never-detected band that, at bf16-output precision, was enough
to implant a backdoor while every step verified. Tighter tolerances (probabilistic, empirical)
shrink the band ~3×–1000× but lose the honest-always-passes guarantee and are hardware-specific.

**(b) Sketch-verifying a step vs replay.** In FLOPs: 39×/74×/130× cheaper at d = 128/256/512
with 1% op sampling. The ratio grows with the smallest GEMM dimension, so with width, and is
capped by sequence length for attention. In CPU wall time: 0.6–2.4× at ρ = 0.01, and worse than
replay at ρ = 0.1. The verifier is memory-bound and must read every activation, which for the
protocol means the prover commits (hashes) all of them: ~10× the step time in my Python
prototype. Bitwise row replay (needed to close the tolerance gap) adds ρ of GEMM FLOPs (16.5% here
with MKL's padding). Freivalds turns "recompute" into "re-read", which is a real but bounded win.

**(c) Can tolerance-based checks be gamed?** Yes. With a rigorous tolerance for bf16-output
GEMMs, a backdoor was implanted to 100% with all 1600 steps verified, in 2 of 2 runs.
With fp32 rigorous, probabilistic or empirical tolerances at 1024–4096 tokens/step, it was not
implanted in 400–800 steps (weak consistent nudge under fp32 worst-case). But the fp32 budget
grows ∝ tokens per step; measured (Phase 6), it exceeds the implanting bf16 budget from ~65k
tokens and reaches 0.36–0.85× the gradient at 1M tokens. At LLM batch sizes the rigorous fp32
tolerance is therefore *more* gameable than the bf16 one that was gamed here.
Exact integer checks and the hybrid Freivalds + bitwise-replay verifier both stopped the
attack (100% per-step detection).

## 4. What failed or was skipped

- **Float batched check**: works as code, useless as a verifier (documented, not "fixed").
- **First rigorous bound**: honest false alarms from subnormals. Fixed with an underflow term,
  and the affected run was rerun.
- **First Adam-tolerance adversary**: my bug (it used an absolute tolerance the verifier computes
  differently). Fixed and rerun.
- **Two experiment bugs caught by asserting honest-passes**: a layout-cache key collision, and a
  4-D indexing bug.
- **Lost time to my own tooling.** Two background jobs hit the 30-min / 2-h limits: the main
  Phase-4 job before its int runs, which I reran, and the Phase-5 job during the s = 4 run, so
  s = 16 never ran. `pkill -f` patterns killed my own queued shell chains once.
- **Dropped runs**: bf16-output at 4096 tokens/step (~10 s/step under contention); the budget
  there is nearly batch-independent.
- **Not done**: multi-step aggregation, gradient-accumulation linearity check, activation-free
  verification, direction tests, a separate Adam-linear check, polynomial commitments.
- **Unresolved**: training runs with identical seeds diverge across processes after ~100 steps
  (matmul and SDPA backward are bit-stable in-process). Single-checkpoint backdoor numbers are
  therefore noisy; headline claims rest on 2 repeats.

## 5. Caveats

- Tiny models (d = 128–512, 1–2 layers, 1024–4096 tokens per step), CPU fp32 (MKL), one
  dataset, one backdoor, one adversary family (weight-gradient GEMMs only; also perturbing
  forward/dX GEMMs would give the adversary more room). Two repeats for the main Phase-4 result.
- "bf16" runs train in fp32 and only use the bf16-output tolerance. Real bf16 kernels, GPUs, and
  tensor-core accumulation orders were not tested.
- Tampering in Phases 1–3 is synthetic (single entries, rank-1, dense noise, orthogonal-to-r).
- The worst-case bounds are a-priori theory. The empirical/probabilistic tolerances held on the
  cases I ran but carry no guarantee; batch-invariance of MKL row replay is an observed property
  of one library build.
- FLOP counts for non-GEMM ops use an assumed per-element cost model (stated in Phase 2).
  Wall times are noisy (fp64 checks varied up to ~2× between runs).
- The Phase-6 budget measurement uses the tiny model and character data; the backdoor itself
  was not rerun at 65k–1M tokens per step (CPU too slow), so "would succeed there" rests on the
  Phase-4 result at a smaller budget. It also assumes dW is one GEMM over all tokens.
- The protocol is a prototype: no link between committed tensors and real data/weights,
  per-row Merkle paths without multiproofs, Python hashing, sketches that leak a random
  projection of every GEMM.

## 6. Next steps (ranked)

1. **Exact-arithmetic training recipes** (int8/fixed-point GEMMs with exact int32 accumulation,
   deterministic requantization, and integer/LUT definitions of softmax, norms and Adam). Then
   every check is exact. Needs GPU int8 tensor-core kernels to check accuracy and speed at a
   meaningful scale. Phase 4 suggests that int8 dW alone already removes the GEMM channel.
2. **Hybrid Freivalds + bitwise row replay on GPU** with batch-invariant kernels
   (fixed reduction order independent of batch size), so replay needs no padding. Measure the
   replay cost at ρ ≈ 1% on real model sizes, and test the sparse adversary over thousands of
   steps. Needs a GPU.
3. **Close the sparse-lie gap in the protocol**: polynomial / linearly-homomorphic commitments,
   so Cr is opened as one evaluation (zk-SNARK-style), or adaptive opening rates for small
   tensors. Also chain commitments across steps and to a pre-registered dataset.
4. **Bitwise checks for non-GEMM ops** instead of rtol, since recomputation was bit-identical on
   one machine. This removes the op-tolerance channel; cross-hardware determinism of elementwise
   kernels needs testing on GPU.
5. **Realistic-scale attack**: rerun the Phase-4 adversary at 10⁵–10⁶ tokens per step on a
   ~100M-parameter model (budget measured in Phase 6, attack not yet run), and analyse recipes
   that split the reduction (gradient accumulation, all-reduce, split-K). Needs a GPU.
6. Fused GPU Freivalds kernels (read bf16 once, fp32 accumulate) to measure real
   verifier/replay wall-time ratios.
7. Step-sampling economics with realistic cheating strategies, the auditor's storage cost,
   and confidentiality leakage of the sketches.


---

## Part 2 (follow-up session): creative alternatives that remove the tolerance

Asked to invent and test new ideas (e.g. "take the mod of things") that cost compute instead of
changing the model, I built and tested three more. Write-ups:
[phase7_certified.md](results/phase7_certified.md), [phase8_misc.md](results/phase8_misc.md),
[phase9_data_order.md](results/phase9_data_order.md). Tests: 47 passing.

### Phase 7 — rounding certificates (the main new result)
Each matmul is defined as "exact product of the bf16 (block-float) inputs, rounded once to
fp32". The lab publishes the integer rounding residual R, and the verifier checks
A·B ≡ C + R (mod 2⁶¹−1) plus an exact range check on R.

| result | value |
|---|---|
| exact vs rational arithmetic | 0 / 480 mismatches |
| forged outputs ±1–3 ulp with matching forged residual | 0 / 2500 accepted (true output 500/500) |
| adversaries vs certified weight gradients (naive / forged residual / 1-ulp forgery) | caught 40/40, 40/40, 40/40 steps |
| float training with all 43 matmuls certified, 400 steps | held-out loss 2.417 vs 2.394 fp32; 8/8 checkpoints fully verified |
| check cost vs fp32 matmul (fused numba) | 17.6× at n=256 → 1.65× at n=4096 (O(n²) vs O(n³)); small-r variant 1.26× at 4096 |
| complete exact verification of a step (all certificates + all ops bitwise), d=128 | 318 ms vs 28 ms fp32 replay, 477 ms certified replay |

This gives the "no room at all" property of integer training while keeping float training,
and it doesn't depend on batch-invariant kernels. The costs: a slow prover in my prototype,
and a check that only beats sgemm at large matmuls on CPU. Along the way, the first certified
training run crashed on subnormal outputs. The spec now defines underflow, the same lesson as
Phase 4.

### Phase 8 — random-prime Freivalds; cross-library ops
- A modulus chosen by Fiat–Shamir after commitment blocks the multiple-of-p attack: a
  k·(2³¹−1) tamper was accepted 2000/2000 by the fixed modulus, 0/2000 by the random prime,
  and 0/2000 by a grinding attacker.
- numpy vs torch disagree bit-wise on every non-matmul op tested, even AdamW. torch's CPU sqrt
  is not correctly rounded (6740/2²⁰ entries wrong) but is self-consistent across tensor sizes.
  A verifiable recipe must pin op implementations, not just formulas.

### Phase 9 — the data channel
With every computation exact and honest:
- poisoning 1 of 16 sequences implants the backdoor to 100% in 400 steps;
- greedy batch selection from clean data shifts the target log-prob (one seed −9.94 → −5.60,
  the other −9.01 → −8.52) without implanting it.

Verification must also bind the dataset and the batch order (commit the data, derive the order
from public randomness).

### Phase 10 — which matmuls need certificates ([results/phase10_budget_by_type.md](results/phase10_budget_by_type.md))
Under the fp32 worst-case tolerance, forward and input-gradient budgets grow ∝ width but stay
~10⁻³ at d = 1024, about 20× below the level that implanted the backdoor. Weight-gradient
budgets grow ∝ tokens per step and reach 0.4–0.9× the gradient at 1M tokens (Phase 6).
Certifying just the weight-gradient matmuls closes the dominant channel; full certification is
the safe choice at frontier widths.

### Phase 11 — the real nanoGPT ([results/phase11_nanogpt.md](results/phase11_nanogpt.md))
`freivalds/nanogpt_hook.py` verifies all 75 matmuls of a training step of the repo's own
`model.py` through `step.training_step` (real AdamW, grad clipping, weight tying), without
modifying them.
- Float and certified honest steps verify, and 300-step certified training matches fp32
  (held-out loss 2.407 vs 2.396; 12/12 checkpoints verified).
- A sub-tolerance weight-gradient perturbation and a 1-ulp forgery pass the float check but are
  caught by the certificates, pinpointing the tampered layer.
- Non-matmul ops on the real model (LayerNorm, GELU, softmax, embeddings, clipping, AdamW) are
  not yet covered.

### Hardening found by fuzzing
A bit-flip fuzzer (4000 random single-bit flips of C or R) found one accepted forgery class:
0 → −0, a free covert bit per zero output. The spec now requires +0, and the "bit-exact" op
check now compares bit patterns (torch.equal treats −0 = +0). Re-fuzzed: every flip rejected
(test `test_bitflip_fuzz_and_negative_zero`).

### Updated verdict
- Exact verification is achievable without integer training: certified correctly-rounded
  matmuls plus pinned, bit-exact non-matmul ops leave an adversary zero computational room.
  The price is compute for the lab (exact accumulation) and an O(n²) verifier.
- Once computation is exact, the remaining attack surface is **data**: what is in the batch
  and which batch is used. That needs commitments to the dataset and to the order.
