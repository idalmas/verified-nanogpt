# Phase 6 — measured adversary budget at realistic tokens per step (up to 2²⁰)

Follow-up to Phase 4, which extrapolated the fp32 tolerance budget from 256–16 384 tokens.
Here it is **measured** up to 1 048 576 tokens per step.

Code: `experiments/phase6_budget_scaling.py`, plot `experiments/phase6_plot.py`.
Raw: `results/phase6_raw.json` (per-GEMM values), log `results/phase6_stdout.txt`.
Plot: `results/phase6_budget.png`.

## Method

The adversary's room in a weight-gradient GEMM dW = Xᵀ·dY (inner dimension = N tokens) is
τ_i = κ(N)·Σ_j(|X|ᵀ|dY|)_ij per row (Rademacher r; see Phase 4). I stream N tokens through the
tiny GPT (d = 128, 2 layers) as 1024-token microbatches with a custom autograd matmul that
accumulates, in float64, both Xᵀ·dY (the real gradient) and |X|ᵀ·|dY| for all 9 transformer +
LM-head weight GEMMs. The reported number is the per-entry budget κ(N)·rowsum(|X|ᵀ|dY|)/k divided
by the rms of the gradient row: median over rows and GEMMs (min/max over GEMMs in the json).
Measured at two model states: at init, and after 300 honest AdamW steps (batch 16×64). The claimed
recipe is assumed to compute dW as **one** GEMM over all N tokens (no accumulation across
microbatches in the claim), with fp32 accumulation.

## Results

| tokens per step N | κ fp32 worst γ_N | fp32 worst-case: init / trained | bf16-output: init / trained | probabilistic: init / trained |
|---|---|---|---|---|
| 1 024 | 6.1e−5 | 3.3e−4 / 5.8e−4 | 2.2e−2 / 3.8e−2 | 6.3e−5 / 1.1e−4 |
| 4 096 | 2.4e−4 | 1.3e−3 / 2.8e−3 | 2.1e−2 / 4.7e−2 | 1.2e−4 / 2.6e−4 |
| 16 384 | 9.8e−4 | 5.2e−3 / 1.2e−2 | 2.6e−2 / 6.0e−2 | 2.4e−4 / 5.6e−4 |
| 65 536 | 3.9e−3 | 2.1e−2 / 4.8e−2 | 4.2e−2 / 9.6e−2 | 4.9e−4 / 1.1e−3 |
| 262 144 | 1.6e−2 | 8.5e−2 / 2.0e−1 | 1.1e−1 / 2.5e−1 | 9.8e−4 / 2.3e−3 |
| **1 048 576** | 6.7e−2 | **3.6e−1 / 8.5e−1** (GEMM range 0.38–1.93 trained) | 3.8e−1 / 9.0e−1 | 2.0e−3 / 4.6e−3 |

Reference: the bf16-output budget at 1024 tokens — **2.2e−2 at init, 3.8e−2 after 300 steps** —
was enough in Phase 4 to implant the backdoor to 100% in 800 steps with every step verified.

## What this shows

1. **The rigorous fp32 worst-case tolerance becomes exploitable at realistic batch sizes.** Its
   budget grows ∝ N (γ_N ∝ N, while the gradient grows only like |X|ᵀ|dY| because of sign
   cancellation across tokens). It crosses the Phase-4 implanting budget at ~65k tokens per step
   (trained state: 4.8e−2 > 3.8e−2) and at 1M tokens it is 0.36–0.85× the gradient per entry, with
   some GEMMs at 1.9×: the adversary may change the weight gradient by roughly its own size and every
   step still passes. This replaces the Phase-4 extrapolation (which predicted ~0.3) with a
   measurement; the measurement is larger after training, because trained gradients cancel more.
2. **At ≥ 256k tokens the fp32 and bf16 bounds are the same order** — precision of the output no
   longer matters; the accumulation-length term γ_N dominates.
3. **The probabilistic λ√N·u bound stays small** (≤ 4.6e−3 at 1M tokens, ~8× below the implanting
   budget) but it is not a guarantee: an honest kernel whose rounding errors are correlated (e.g.
   blocked accumulation with systematic rounding in one direction) can exceed it, and the verifier
   then has to choose between false alarms and loosening it.

## Caveats

* Tiny model (d = 128) and tiny-Shakespeare characters; |X|ᵀ|dY| / |XᵀdY| depends on the data and
  model, so the exact numbers will differ for real LLMs. The trend (budget ∝ N for the worst-case
  bound) is structural.
* Real recipes split long reductions (gradient accumulation across microbatches, data-parallel
  all-reduce, blocked split-K GEMMs). A verifier that checks each microbatch GEMM separately gets
  γ_{N_micro} instead of γ_N, but must then verify the accumulation/all-reduce sums too (another
  float reduction with its own tolerance). Not tested here.
* "Budget" is a per-step quantity; I did not rerun the backdoor at 65k–1M tokens per step
  (too slow on CPU). The claim that it would succeed rests on Phase 4's result at a smaller budget.
