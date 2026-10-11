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

## Cost: sketch-verify vs replay (`results/phase2_cost_stdout.txt`, quiet machine, 4 threads)

FLOPs: replay = 2·b·m·n·k per GEMM + an assumed per-element cost for each op (LN 8–14,
GELU 12–20, softmax 4–5, AdamW 16, add 1; `OP_FLOPS_PER_ELEM`). Verifier = per GEMM
2(mk + nk + mn) for Cr, Br, A(Br) plus 2(nk + mn) for the tolerance |A|(|B||r|) (the
underflow term's extra row sums, ~mn per GEMM, are not counted) + rho × op cost. Wall: median
of 3 verifications (after one warm-up that builds the layout index maps) vs median of 5 replays
of the step (`training_step`, record off). Across the three cost runs made tonight
(`phase2_cost_stdout.txt` is the last, on a quiet machine), the fp64 GEMM-check times varied
by up to ~2× for the same configuration, so treat single wall-time ratios as ±50%.

| d | batch×T | rho | verifier precision | replay FLOP | verifier FLOP | FLOP ratio | replay ms | verify ms (GEMM + ops) | wall ratio |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 16×64 | 0.01 | fp64 | 2.799e+09 | 7.226e+07 | 38.7× | 44.8 | 48.4 (27.4 + 20.9) | 0.92× |
| 128 | 16×64 | 0.01 | fp32 | 2.799e+09 | 7.226e+07 | 38.7× | 44.8 | 33.1 (13.9 + 19.3) | 1.35× |
| 128 | 16×64 | 0.1 | fp32 | 2.799e+09 | 7.790e+07 | 35.9× | 44.8 | 66.9 (14.3 + 52.6) | 0.67× |
| 256 | 16×64 | 0.01 | fp64 | 1.043e+10 | 1.402e+08 | 74.4× | 66.0 | 103.6 (73.3 + 30.3) | 0.64× |
| 256 | 16×64 | 0.01 | fp32 | 1.043e+10 | 1.402e+08 | 74.4× | 66.0 | 82.1 (28.5 + 53.6) | 0.80× |
| 256 | 16×64 | 0.1 | fp32 | 1.043e+10 | 1.520e+08 | 68.6× | 66.0 | 154.4 (25.8 + 128.6) | 0.43× |
| 512 | 16×64 | 0.01 | fp64 | 4.025e+10 | 3.107e+08 | 129.5× | 210.9 | 224.7 (179.2 + 45.5) | 0.94× |
| 512 | 16×64 | 0.01 | fp32 | 4.025e+10 | 3.107e+08 | 129.5× | 210.9 | 87.0 (40.4 + 46.7) | 2.42× |
| 512 | 16×64 | 0.1 | fp32 | 4.025e+10 | 3.388e+08 | 118.8× | 210.9 | 223.9 (35.2 + 188.7) | 0.94× |
| 512 | 8×128 | 0.01 | fp64 | 4.120e+10 | 3.325e+08 | 123.9× | 172.1 | 109.7 (67.0 + 42.7) | 1.57× |
| 512 | 8×128 | 0.01 | fp32 | 4.120e+10 | 3.325e+08 | 123.9× | 172.1 | 82.0 (40.7 + 41.3) | 2.10× |
| 512 | 8×128 | 0.1 | fp32 | 4.120e+10 | 3.618e+08 | 113.9× | 172.1 | 247.3 (36.5 + 210.8) | 0.70× |

* FLOP savings grow linearly with width: 39× (d = 128), 74× (d = 256), 130× (d = 512), because
  each GEMM's cost/size ratio is ~min(m, n, k). The step's GEMMs are mostly 1024-row token
  matrices, so the ratio is bounded by d, and by T for attention GEMMs.
* Wall time on CPU barely follows: 0.6–2.4× at rho = 0.01 (best with the rigorous fp32
  verifier at d = 512); at rho = 0.1 the verifier is *slower* than replay (0.4–0.9×).
  Reasons: matrix–vector work is memory-bound (the verifier must read every A, B, C once),
  float64 checks double the bytes, and the op checker does random gathers plus per-op Python
  overhead (85 ops). The FLOP ratio is the right proxy only when the verifier's bytes moved,
  not FLOPs, are cheap — on a GPU, a fused Freivalds kernel reading bf16 operands would run at
  memory bandwidth; GEMM replay at tensor-core peak. The wall-time ratio there would be set by
  the GEMM's arithmetic intensity relative to the hardware's FLOP:byte ratio — not measured
  here (no GPU).
* An important hidden cost: the verifier needs **all activations** of the step (A, B, C of
  every GEMM, ~92 MB for d = 128 here) — see Phase 3 for how commitments + openings reduce
  what must be shipped.
