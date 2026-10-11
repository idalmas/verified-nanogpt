# Phase 2 — sketch-verifying a whole training step

Code: `freivalds/tinygpt.py` (hand-written forward/backward/AdamW with every tensor recorded),
`freivalds/stepverify.py` (verifier). Experiment: `experiments/phase2_step.py`.
Raw: `results/phase2_raw.json`; logs `results/phase2_stdout_part1.txt` (honest training +
first batched-mod-p run), `results/phase2_stdout_part2.txt` (batched, sampling, tamper
locations), `results/phase2_cost_stdout.txt` (cost, quiet machine).
Tests: `tests/test_stepverify.py` (backward == autograd; honest step passes (separate,
combined, fp32-verifier); GEMM tampers in forward / dX / dW / attention / embedding caught;
op tampers caught at rho = 1; layout tamper; sub-tolerance GEMM tamper documented as missed;
mod-p combined honest/tampered; layout-cache regression; int8-dW exact check).

## The step

Char-level GPT, non-parametric LayerNorm (as OLMo-1), no biases, untied embeddings, AdamW,
no grad clipping. Default d = 128, 2 layers, 4 heads, batch 16 × 64 tokens. Per step the
trace holds **43 GEMMs** (6 forward + 12 backward per layer — including QKᵀ, PV and their
backward as batched GEMMs — plus embeddings as one-hot GEMMs and the LM head) and
**85 non-GEMM ops** (LN fwd/bwd, softmax fwd/bwd, GELU fwd/bwd, CE grad, residual adds,
mask/scale, AdamW per parameter, transposes/head split as layout ops). The hand-written
gradients match torch autograd to max |Δ| ≈ 3e−8 (test). Honest 300-step training run:
loss 4.22 → 2.43, verification (rho = 0.01) passed at all 13 checkpoints.

Verifier: float Freivalds per GEMM (Rademacher r, rigorous γ_n tolerance; float64
verifier or a rigorous fp32-verifier mode whose tolerance also covers the verifier's
rounding) + random sampling of a fraction rho of elements (elementwise ops) or rows (row-wise
ops) recomputed and compared at rtol 1e−5 + 1e−6·max. On the same machine the recomputed op
outputs were **bit-identical** to the prover's in all 65 sampled outputs checked, so an exact
comparison would also have worked here.

## Batched (one random linear combination) vs separate Freivalds

**Exact, mod p** (synthetic int8 GEMMs with step-like shapes, single-entry tamper not ≡ 0
mod p, 20 000 trials):

| p | combined false-accept | bound 3/p | separate false-accept | bound 1/p |
|---|---|---|---|---|
| 31 | 0.0933 | 0.0968 | 0.0349 | 0.0323 |
| 127 | 0.0234 | 0.0236 | 0.0085 | 0.0079 |

(The separate check is 2σ above 1/p at p = 31 and 1σ above at p = 127 — binomial standard
errors 0.0013 and 0.0006 — consistent with sampling noise; the exact acceptance probability
for a single-entry tamper is 1/p.) The combined check's soundness follows
Schwartz–Zippel for a degree-3 polynomial (α, s, r) and is ≈3× weaker per repetition, which is
irrelevant at p = 2⁶¹ − 1. With p = 2⁶¹ − 1 on all 43 GEMM shapes of the real step (operands
quantized to int8): honest passed both ways; a +1 tamper in a random GEMM was caught 50/50.
It saves communication (one field element per step) but **not** verifier work (0.19 s vs 0.16 s).

**Float** — single-entry tamper of size rel × rms(C) in one GEMM, 30 trials per point:

| GEMM (rms of C) | separate: first rel detected ≥ 50% | combined: first rel detected ≥ 50% |
|---|---|---|
| h0.fc1 forward (0.23) | 0.032 | never (up to 100×) |
| h1.att_s (QKᵀ) (0.30) | 0.001 | 100 |
| lm_head.dW (0.011) | 0.01 | never (up to 100×) |
| h0.qkv.dX (6.6e−6) | 0.032 (0.2 at 0.003) | never (up to 100×) |

**The float combined check is useless**: its tolerance is the sum of all 43 GEMMs'
tolerances, dominated by the largest-magnitude GEMMs, so a tamper 100× larger than a
gradient tensor's typical entry disappears into the budget. Batching float checks across
tensors of different scales must not be done; exact batching is fine.

## Coordinate sampling for non-linear ops (300 verifier seeds per cell)

| op (units) | single large tamper, rho = 0.001 / 0.01 / 0.1 | theory 1−(1−1/U)^⌈ρU⌉ | dense ×(1+1e−3) | single ×(1+1e−7) (below rtol) |
|---|---|---|---|---|
| h0.gelu (524 288 elems) | 0.000 / 0.013 / 0.100 | 0.001 / 0.010 / 0.095 | 1.000 at every rho | 0.000 at every rho |
| h1.softmax (4096 rows) | 0.000 / 0.003 / 0.100 | 0.001 / 0.010 / 0.095 | 1.000 | 0.000 |
| h0.ln2.bwd (1024 rows) | 0.000 / 0.010 / 0.110 | 0.002 / 0.011 / 0.096 | 1.000 | 0.000 |
| ce_grad (1024 rows) | 0.000 / 0.010 / 0.110 | 0.002 / 0.011 / 0.096 | 1.000 | 0.000 |
| adam.h0.w1 (65 536 elems) | 0.000 / 0.007 / 0.093 | 0.001 / 0.010 / 0.095 | 1.000 | 0.000 |

Sampling behaves exactly as the binomial model: dense tampering is always caught, a single
corrupted coordinate is caught with probability ≈ rho, and a change below the recompute
tolerance is never caught.

## Tamper injected into different parts of the step (full verify, rho = 0.01, 20 seeds)

"single" = one entry += 10×(max|·|); "dense" = whole tensor ×(1 + 1e−3).

| location | single | dense ×(1+1e−3) |
|---|---|---|
| GEMMs: h0.fc1, h1.att_s, emb_tok.dW | 1.00 | 1.00 |
| h0.qkv.dX | 1.00 | 0.90 |
| lm_head.dW | 1.00 | 0.95 |
| **h1.fc2.dW** | 1.00 | **0.05** |
| non-linear ops: ln1, softmax, gelu, gelu.bwd, softmax.bwd, ce_grad, res1, adam | 0.00 | 1.00 |
| lnf.bwd, h0.merge (layout) | 0.05 | 1.00 |

Only the tampered record's check ever fired (the adversary recomputes everything downstream
honestly, so downstream checks pass). Two clean failure modes: (1) single-coordinate tampering
of non-linear ops is caught only ≈ rho of the time; (2) a **0.1% dense rescaling of a weight
gradient (h1.fc2.dW) passes Freivalds 95% of the time** because it is within the rigorous
tolerance — this is the channel Phase 4 exploits.
