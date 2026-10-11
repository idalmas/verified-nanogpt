# Phase 4 — steering training inside the tolerance (backdoor experiment)

Code: `experiments/phase4_adversary.py` (adversaries, backdoor, evaluation),
`experiments/phase4_plot.py`. Raw: `results/phase4_raw.json` (all runs, budget scaling,
empirical-κ calibration). Logs: `results/phase4_stdout_main.txt` (first 7 runs + budget
scaling), `phase4_stdout_int.txt`, `phase4_stdout_v2.txt`, `phase4_stdout_B64.txt`,
`phase4_stdout_800.txt`. Plot: `results/phase4_backdoor.png`.

## Setup

* Model/data: the Phase-2 tiny GPT (d = 128, 2 layers, 4 heads, context 64), tiny-Shakespeare
  characters, AdamW lr 3e−3, batch 16 × 64 = 1024 tokens (64 × 64 = 4096 for "_B64"),
  400 or 800 steps, same seeds and data order for every run.
* Backdoor: trigger `$&$` (the corpus contains 1 `$` and 3 `&`) must be followed by `QZQZ`.
  Metrics on 200 held-out prompts ending in the trigger: **greedy success** (all 4 generated
  chars correct) and **teacher-forced mean log p(target char)**; clean loss on 6 batches.
* Verifier (every step): Phase-2 `verify_step` = float Freivalds on all 43 GEMMs with the
  tolerance setting under test (now including the underflow term, see below) + coordinate
  sampling of all 85 ops at rho = 0.01 (+ exact F_p checks of int GEMMs in the int runs).
* Adversary: each step it computes the gradient of the backdoor loss on its own private
  trigger batch, G = ∇_W L_bd, and adds to every **weight-gradient GEMM output** dW = Xᵀ·dY a
  perturbation E with ‖E_i‖₁ = f·τ_i per row, E_i ∝ G_i (L1-normalised). With Rademacher r,
  the verifier's tolerance τ_i = κ·Σ_j(|A||B|)_ij does not depend on r, so |E_i·r| ≤ ‖E_i‖₁ < τ_i
  for **every** r: the perturbation passes no matter what randomness the verifier draws.
  f = 0.9 (0.5 for the empirical tolerance, where honest residuals are closer to τ).
  Everything downstream (AdamW, the next steps) is computed honestly, so all other checks pass.

Tolerance settings (κ):

| setting | κ (N = 1024 inner dim) | status |
|---|---|---|
| `tol_worst` | γ_N = 6.1e−5 | rigorous for fp32 accumulation + fp32 output |
| `tol_bf16out` | γ_N + u_bf16(1+γ_N) ≈ 3.9e−3 | rigorous bound a verifier must accept if the recipe stores dW (C) in bf16 |
| `tol_prob` | 6·√N·u = 1.1e−5 | Higham–Mary probabilistic bound (not a guarantee) |
| `tol_emp` | 10 × max honest \|res\|/M = 1.97e−6 | calibrated on 10 honest steps (max ratio 1.97e−7) |
| `adam_tol` | — | perturb AdamW's output θ' by 0.9 × the op-recompute rtol (1e−5·\|θ'\|) |
| `int_exact` | 0 | dW GEMMs in int8 with exact int32 accumulation, exact F_p Freivalds; adversary adds ±1 to one int32 entry of each dW (the smallest possible change) |

## Results

| run | tokens/step | steps passed | greedy backdoor success (final) | mean log p(target), 2nd half of run | clean loss (final) |
|---|---|---|---|---|---|
| honest | 1024 | 400/400 | 0.000 | −9.28 | 2.338 |
| honest_800 | 1024 | 800/800 | 0.000 | −9.54 | 2.098 |
| honest_B64 | 4096 | 400/400 | 0.000 | −8.54 | 2.129 |
| int_honest (int8 dW) | 1024 | 400/400 | 0.000 | −9.95 | 2.350 |
| poison_unbounded (no stealth; ∇L_bd at ‖dW‖ scale) | 1024 | **0/400** | 1.000 | −0.00 | 2.517 |
| tol_worst | 1024 | 400/400 | 0.000 | −8.67 | 2.362 |
| tol_worst_B64 | 4096 | 400/400 | 0.000 | −7.49 | 2.095 |
| tol_worst_800 | 1024 | 800/800 | 0.000 | −7.68 | 2.100 |
| tol_prob | 1024 | 400/400 | 0.000 | −9.95 | 2.333 |
| tol_emp | 1024 | 400/400 | 0.000 | −9.93 | 2.328 |
| adam_tol_v2 | 1024 | 400/400 | 0.000 | −9.57 | 2.328 |
| **tol_bf16out_v2** | 1024 | **400/400** | 0.000 (log p −1.96 at step 400) | **−3.36** | 2.347 |
| **tol_bf16out_800a** | 1024 | **800/800** | **1.000** (0.92 at step 501, 0.73 at 651, 1.0 at 701–800) | **−0.25** | 2.095 |
| **tol_bf16out_800b** | 1024 | **800/800** | **1.000** (0.68 at 601, 1.0 at 651, 0.0 at 751, 1.0 at 800) | **−1.12** | 2.101 |
| int_exact (±1 per dW per step) | 1024 | **0/400** | 0.000 | −7.54 | 2.349 |

Superseded runs, kept in the raw json for the record: `tol_bf16out` (v1) passed 380/400 — all 20
failures were the *honest* GEMM h1.att_y.dB (see "Underflow" below), and it reached 0.92 greedy
success at step 400; `adam_tol` (v1) failed 400/400 because my adversary used the verifier's
absolute tolerance term computed over the whole tensor, while the verifier computes it over
the sampled entries (an adversary bug, fixed in v2).

### What this shows

1. **Tolerance-based checks can be gamed — with the tolerance a bf16 recipe needs.** Under
   the rigorous bf16-output bound, the adversary implanted the backdoor to 100% greedy success
   at the end of both 800-step runs (with non-monotone checkpoints in between, e.g. run b
   reads 0.0 at step 751) while **every one of 1600 steps passed** Freivalds on all GEMMs and
   coordinate sampling of all ops. Clean loss is indistinguishable from honest training
   (2.095 / 2.101 vs 2.098). The unconstrained poisoner reaches the same backdoor faster but
   fails every step, i.e. the verifier does stop naïve cheating.
2. **With fp32-level tolerances at this scale, steering did not implant the backdoor** in
   400–800 steps (or with 4096-token batches): probabilistic, empirical and Adam-rtol channels
   stayed in the honest runs' log-prob band. The fp32 worst-case channel shows a **weak but
   consistent** effect: in the 800-step pair (same data order) tol_worst's log p(target) is
   above honest at all 15 checkpoints from step 101 on (by 0.2–4.0 nats; 2nd-half mean −7.68
   vs −9.54), and also above at 400 steps (−8.67 vs −9.28) and at 4096 tokens (−7.49 vs −8.54).
   Greedy success stayed 0. Given the cross-process nondeterminism noted below, I report this
   as suggestive (consistent sign), not as a measured effect size.
3. **Exact integer checks leave no room.** The smallest possible perturbation (±1 in one int32
   accumulator entry of each dW) was rejected on 400/400 steps, by exactly the tampered GEMMs.
   There is no sub-threshold band to hide in. (The int8-dW recipe trains normally: clean loss
   2.350 vs 2.338 fp32.) The remaining channels in the int setting are the non-GEMM ops, which
   are only coordinate-sampled (Phase 2), and the quantization ops themselves.

### Why the bf16 budget is enough and the fp32 one (here) is not — budget vs batch size

Per-entry adversary budget relative to the gradient, (τ_i / k) / rms(dW_i), median over rows
and the 8 transformer dW GEMMs of one step (`budget_scaling` in the raw json):

| tokens per step N | fp32 worst-case | bf16-output | probabilistic |
|---|---|---|---|
| 256 | 5.9e−5 | 1.5e−2 | 2.2e−5 |
| 1 024 | 2.6e−4 | 1.7e−2 | 4.9e−5 |
| 4 096 | 1.1e−3 | 1.9e−2 | 1.1e−4 |
| 16 384 | 4.6e−3 | 2.3e−2 | 2.2e−4 |

* The bf16 budget is ~2% of the gradient per entry, independent of N, because the output
  rounding term u_bf16·|A||B| dominates; |A||B| ≫ |AB| because the weight gradient is a sum
  over tokens with cancelling signs. 2% of the gradient, pushed coherently every step through
  Adam's normalisation, implanted the backdoor in ~500–700 steps.
* The fp32 worst-case budget grows ~linearly in N (×79 for ×64 tokens). **Extrapolating this fit
  (not measured)** to N ≈ 10⁶ tokens per step, typical of LLM pre-training, gives a budget of
  ~0.3× the gradient per entry — an order of magnitude more than the bf16 budget that sufficed
  here. A rigorous a-priori fp32 bound at frontier batch sizes would therefore leave *more*
  room than the bf16 one tested here. (The bound needs N·u < 1, true up to N = 2²⁴.)

### Underflow: a "rigorous" bound that wasn't

The first tol_bf16out run had 20 failing steps, all on the honest GEMM `h1.att_y.dB` (dV = attᵀ·dY),
which the adversary never touches. Reproduction: as the backdoor makes attention peaked, 943
softmax outputs became subnormal (down to 1.4e−45); MKL's sgemm flushed them, the error equalled
the full product (3.7e−46), 2.6×10⁵ times the γ_n bound (which assumes no underflow). Fix: add
η(‖A_i‖₁‖r‖₁ + Σ_k(|B||r|)_k + 2n‖r‖₁), η = 2⁻¹²⁶, to τ (core, step verifier, protocol);
regression test `test_underflow_gemm_honest_passes` uses the saved GEMM. All runs labelled
v2 / B64 / 800 use the fixed verifier. Lesson: honest-passes-always needs the *full* floating-point
model of the actual kernel (FTZ/DAZ, subnormals), and an adversary can trigger these regimes
on purpose.

### Caveats

* One tiny model, one backdoor, one adversary family (dW GEMMs only; perturbing forward and
  dX GEMMs too would give more budget). Few repeats: 2 runs for the headline bf16 result.
* Training is not bit-reproducible across processes here: tol_bf16out v1 and v2 (identical
  seeds) diverge after step ~100 even though matmul and SDPA backward are bit-stable
  in-process; I did not find the source. Greedy success at a single checkpoint is therefore
  noisy (v1 0.92 vs v2 0.00 at step 400); the 800-step result is robust in both repeats.
* The "bf16" runs train in fp32 and only use the bf16-output tolerance; a true bf16 recipe would
  also have bf16 honest noise, which does not change the adversary's budget.
