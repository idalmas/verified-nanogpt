# Phase 11 — verification on the repo's real nanoGPT (`model.py`, `step.training_step`)

Code: `freivalds/nanogpt_hook.py` (no changes to `model.py`, `step.py` or `train.py`).
Experiment: `experiments/phase11_nanogpt.py`. Raw: `results/phase11_raw.json`, log
`results/phase11_stdout.txt`. Tests: `tests/test_nanogpt_hook.py` (3 tests).

## What it does

During a training step, `recording(model, rec)`:
- routes every `nn.Linear` through an autograd function that records the three matmuls
  (forward Y = X·Wᵀ + b, input-gradient dY·W, weight-gradient dYᵀ·X);
- runs `model.py`'s own explicit-attention code path, with QKᵀ and PV (and their backward)
  recorded.

Each matmul is either plain fp32, checked with float Freivalds, or a certified matmul (Phase 7),
checked exactly. The step itself is the repo's `step.training_step`: the real
`configure_optimizers` AdamW (weight decay 0.1), gradient clipping at 1.0, and weight tying
between `wte` and `lm_head`. Config: 4 layers, 4 heads, d = 128, context 64, biases on,
batch 12.

## Results

| check | result |
|---|---|
| hooked vs original model (flash attention) | loss identical to 6 digits; max gradient difference 8.9e−8 |
| matmuls recorded per step | 75 (4 layers × (4 Linears × 3 + 2 attention matmuls × 3) + lm_head × 3), 1.98e9 MACs |
| honest step, float Freivalds on all 75 | verified |
| honest step, certified, all 75 certificates | verified |
| sub-tolerance perturbation of one weight gradient (bf16-output-sized), float mode | **not detected** (as in Phase 4) |
| same perturbation, certified mode (with a forged consistent certificate) | **detected** — exactly `transformer.h.1.mlp.c_fc.dW` |
| 1-ulp forgery of one weight-gradient entry: float / certified | not detected / **detected** |
| 300 training steps, fp32 vs certified (same seeds), verified every 50 steps | held-out loss 2.396 vs **2.407**; 6/6 and 6/6 checkpoints verified |
| wall time for 300 steps (CPU) | 11 s fp32, 233 s certified (prototype prover) |

## Bug found on the way

The first run reported honest failures. The recorder stored *views* of tensors that the step
later mutates in place: AdamW updates the weights, and gradient clipping rescales `p.grad`,
which autograd aliases to the recorded dW output. Recording clones fixed it. A real verifier
needs the same care: commit to tensors *at the moment they are used*.

## Coverage and what is still missing

All matmuls of the step are verified. Not covered yet on the real model:
- token/position embeddings (a gather, and a scatter-add in the backward);
- LayerNorm with weight and bias;
- GELU, softmax, cross-entropy;
- bias gradients;
- the global gradient-norm clip;
- the AdamW update.

In the hand-written tiny GPT these were checked by bit-exact recomputation (Phases 2 and 7).
Here they would need the same recording, or the repo's existing full replay (`verify.py`)
for that part. With certified matmuls, the non-matmul part is all that a replay would still
have to redo.
