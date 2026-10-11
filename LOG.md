# LOG — Freivalds overnight

All times UTC.

- **07:38** Start. Repo is NanoGPT fork with bit-deterministic training. Torch not installed;
  `pip install torch` (2.14.1, CUDA wheel but CPU used). numpy 2.5.3. 4 CPU cores, 15 GB RAM.
  Ran `data/shakespeare_char/prepare.py` (vocab 65, 1.0M train tokens).
- **07:41** Decision: branch. Task says work on `freivalds-overnight` and never push; the session
  harness says develop on `claude/modest-newton-csbrh2` and push there because the container is
  ephemeral. I commit on local branch `freivalds-overnight` and, at the very end, push that
  history to `claude/modest-newton-csbrh2` only (never main), so the night's work is not lost
  when the container is reclaimed.
- **07:41** Phase 0: wrote IDEAS.md (16 ideas, ranked).
- **07:51** Phase 1 done: core.py (int, F_p via exact float64 limbs, float with rigorous γ_n tolerance), tests pass (10). Experiments: honest slack 600×–125000× (fp32), detection thresholds, reps, timing. Found and documented F_p blind spot for multiples of p (small p); optimized matvec_mod limb layout (2.19s→0.50s at n=4096). Wall-time saving of Freivalds on CPU only 0.6–3× because matvecs are memory-bound — recorded honestly.
- **08:06** Phase 2 built: freivalds/tinygpt.py (hand-written fwd/bwd/AdamW, every GEMM and op recorded; matches autograd to ~3e-8), freivalds/stepverify.py (float Freivalds per GEMM, combined check, mod-p combined, coordinate/row sampling). Bugs found by asserting honest-passes in experiments: (1) layout-map cache key collided for equal token counts (B·T) → added output shape to key + regression test; (2) indexing bug for 4-D attention tensors in the experiment. Decision: sample with replacement (randperm too slow); layout ops sampled via cached index map; added rigorous fp32-verifier mode (tolerance also bounds verifier rounding).
- Key Phase-2 finding: float *combined* (random linear combination) check is blind to tampers up to 100× RMS in gradient GEMMs because the summed tolerance is dominated by forward GEMMs with larger magnitudes. Exact mod-p combined check matches 3/p bound.
- **08:06** Phase 3 done: protocol.py (Merkle per-row, SHAKE-256 Fiat–Shamir, sketches v,w,z,q,μ + openings). Decision: elementwise ops sampled per *row* in the protocol (per-element sampling revealed 73% of data at rho=0.01 since openings are rows). Attacks: post-commit caught 5/5, choose-after-r caught with FS / passes without, lying sketch caught at ≈row_rate, grinding cost ≈(1-rate)^-t.
- **08:06** Phase 4 launched (10 runs × 400 steps). Adversary perturbs dW GEMM outputs with ‖E_i‖₁ = 0.9 τ_i (τ_i independent of r for Rademacher r, so passes for every r).
- **08:26** Phase 4 runs: tol_bf16out implanted backdoor (92%) but 20/400 steps failed on h1.att_y.dB — an *honest* GEMM. Hypothesis: subnormal softmax outputs + FTZ/DAZ break the pure γ_n bound. Added an underflow term η=2^-126 (core, stepverify, protocol). Reproduction running. Phase 5 hybrid verifier (Freivalds + bitwise row replay) written; MKL rows bitwise-reproducible only for ≥32-row gathers and fixed thread count.
