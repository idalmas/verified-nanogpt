# Phase 12 — complete verification of a real nanoGPT step by matmul-oracle replay

Code: `freivalds/oracle_replay.py` (+ `freivalds/nanogpt_hook.py`). Experiment:
`experiments/phase12_oracle_replay.py`. Raw: `results/phase12_raw.json`, log
`results/phase12_stdout.txt`. Tests: `tests/test_oracle_replay.py` (3 tests).

## Idea

The verifier replays the repo's own `step.training_step` on the pre-step state and batch, but
never multiplies matrices. At each matmul it:
1. checks that its own recomputed inputs are **bit-identical** to the prover's committed
   inputs, which verifies every non-matmul op upstream;
2. checks the committed output with its certificate (exact) or float Freivalds;
3. continues with the committed output.

At the end its weights and AdamW state must be bit-identical to the prover's claimed post-step
state, which covers gradient clipping and the optimizer. The result is complete coverage of
the step with zero matmul FLOPs for the verifier. It needs the same op implementations on both
sides (Phase 8).

## Results (model.py: 4 layers, d = 128, biases, weight tying; AdamW wd 0.1; clip 1.0; after 20 warm-up steps)

| attack | float matmuls (fp32 Freivalds) | certified matmuls |
|---|---|---|
| honest step | accepted | accepted |
| AdamW output: +1e−6 on one weight | **rejected** (final weights differ) | **rejected** |
| LayerNorm output: +1 ulp in one entry | **rejected** (inputs of h.1.mlp.c_fc differ) | **rejected** |
| GELU backward scaled by (1+1e−3) | **rejected** (inputs of h.2.mlp.c_fc.dX/dW differ) | **rejected** |
| weight-gradient matmul perturbed inside the fp32 tolerance | accepted | **rejected** (certificate) |
| 1-ulp forgery of a forward matmul output | accepted | **rejected** (certificate) |
| claimed post-step weights off by 1 ulp | **rejected** | **rejected** |

Cost (median of 3, CPU):

| mode | replay of the step (same recipe) | plain fp32 step | complete oracle verification |
|---|---|---|---|
| float | 30 ms | 24 ms | 114 ms |
| certified | 837 ms | 38 ms | 335 ms |

## What this shows

* **End-to-end, complete, exact verification of the real nanoGPT training step works.**
  Certified matmuls plus oracle replay caught every attack, from one ulp in a LayerNorm output
  to one ulp in a matmul output. Each failure message names the place the computation first
  diverged.
* With float matmuls, everything except the matmul tolerance band is still caught exactly.
  The band (Phase 4/6) is the only gap, and certificates close it.
* Cost: the verifier does no matmul FLOPs and is 2.5× faster than replaying the certified step.
  At this tiny size it is still slower than a plain fp32 step: Python overhead, plus O(n²)
  checks that only beat O(n³) matmuls at large n (Phase 7). It is also memory-bound: the
  verifier must read every committed activation.
