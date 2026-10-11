# Phase 13 (capstone) — an audited end-to-end training run of the real nanoGPT

Code: `experiments/phase13_audited_run.py` (uses `freivalds/data_binding.py`,
`freivalds/nanogpt_hook.py`, `freivalds/oracle_replay.py`, `freivalds/certified*.py`).
Raw: `results/phase13_raw.json` (per-audit details), log `results/phase13_stdout.txt`.

## Protocol

- **Lab.** Commits to the dataset (Merkle root over 4096-token chunks), then trains 120 steps of
  `model.py` (2 layers, d = 128, biases, weight tying) with `step.training_step` (AdamW wd 0.1,
  clip 1.0). The batch of step t is derived from (dataset root, t, hash of the pre-step weights).
  After every step the lab publishes c_t = H(weights_t, AdamW state_t).
- **Auditor**, after training:
  1. derive up to 8 step indices from the final commitment (Fiat–Shamir);
  2. for each, obtain the pre-state, the step's matmul records and the post-state;
  3. check both states against the hash chain;
  4. check the batch is the prescribed one (with Merkle openings into the dataset);
  5. run the complete matmul-oracle replay (Phase 12).
- **Cheating labs** misbehave only during steps 30–89 (half the run):
  - `cheat_dW`: every weight-gradient matmul perturbed inside the fp32 tolerance;
  - `cheat_batch`: trains on the highest-loss of 4 random batches instead of the prescribed one;
  - `cheat_adam`: +1e−6 on one weight after each AdamW update.

## Results

| matmuls | lab | audited steps (in cheat window) | flagged | verdict | first failure |
|---|---|---|---|---|---|
| float | honest | 8 (5) | 0 | ACCEPT | — |
| float | cheat_dW | 7 (2) | 0 | **ACCEPT (missed)** | — |
| float | cheat_batch | 7 (4) | 4 | REJECT | batch is not the prescribed one |
| float | cheat_adam | 7 (4) | 4 | REJECT | weights differ after replay |
| certified | honest | 8 (5) | 0 | ACCEPT | — |
| certified | cheat_dW | 8 (7) | 7 | REJECT | `lm_head.dW`: certificate rejected |
| certified | cheat_batch | 8 (1) | 1 | REJECT | batch is not the prescribed one |
| certified | cheat_adam | 8 (4) | 4 | REJECT | weights differ after replay |

(Fewer than 8 distinct steps were audited when the hash-derived indices collided.)

Every audited step inside a cheating window was flagged, and no step outside one was. The only
miss is the one the whole project predicts: steering inside the float tolerance. Certified
matmuls close it. Time: 120 training steps took 3–5 s (float) / 34–37 s (certified prototype);
auditing 8 steps took 1 s / 3–5 s.

## Bug found on the way

The first audit rejected the honest lab. `torch.optim.Optimizer.load_state_dict` keeps
references to the tensors it is given, so replaying a challenged step mutated the lab's
*stored* copy of that step's pre-state. Fix: always load deep copies (noted in
`oracle_replay.py`). A real system should hash and store committed states in immutable form.

## What is still not covered

- Content of the committed dataset: poisoning (Phase 9) is a data-audit problem.
- The verifier must run the same software stack (Phase 8).
- The lab must keep (or be able to regenerate) the state and activations of any step it may be
  asked to open.
- A step outside the audited sample is only covered probabilistically: 1 − (1 − q)^c for c
  cheating steps at audit rate q (Phase 5).
