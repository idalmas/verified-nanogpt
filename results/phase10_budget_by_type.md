# Phase 10 — which matmuls need exact certification?

Code: `experiments/phase10_budget_by_type.py`. Raw: `results/phase10_raw.json`, log
`results/phase10_stdout.txt`.

Per-entry adversary budget under the rigorous fp32 worst-case tolerance, divided by the rms of
the output row: median over GEMMs of each type (max in parentheses). 2-layer model, batch
4×64 = 256 tokens, after 50 honest steps.

| width d | forward (inner d or 4d) | input-gradient (inner d or 4d) | weight-gradient (inner = 256 tokens here) | attention (inner d_head / T) |
|---|---|---|---|---|
| 128 | 4.1e−5 (9.8e−5) | 4.5e−5 (1.6e−4) | 9.3e−5 (1.5e−4) | 7.4e−6 |
| 256 | 8.9e−5 (2.7e−4) | 1.4e−4 (3.8e−4) | 9.5e−5 (1.3e−4) | 8.4e−6 |
| 512 | 1.3e−4 (2.9e−4) | 3.7e−4 (1.2e−3) | 5.0e−5 (8.8e−5) | 5.6e−6 |
| 1024 | 3.8e−4 (8.1e−4) | 1.0e−3 (3.2e−3) | 9.7e−5 (1.4e−4) | 8.4e−6 |

Reference points: the budget that implanted the backdoor in Phase 4 was ~2–4e−2. Weight-gradient
GEMMs reach 3.6e−1–8.5e−1 at 1M tokens per step (Phase 6).

* Weight-gradient matmuls are where the room is, because their inner dimension is the token
  count. **Certifying only the dW GEMMs** (Phase 7 certificates) removes the dominant channel,
  at roughly a third of the matmul FLOPs.
* Forward and input-gradient budgets grow with width, roughly ∝ d (×9 and ×22 from d = 128 to
  1024). At d = 1024 they are ~10⁻³, about 20× below the implanting level, but frontier widths
  (d ≈ 8–16k, inner 4d) would bring them toward it. That is an extrapolation, not measured, and
  I did not test whether perturbing activations or input-gradients steers as effectively as
  perturbing dW. A fully certified recipe sidesteps the question.
* Attention matmuls (inner dimension d_head or the sequence length) have negligible budgets
  at these sizes.
