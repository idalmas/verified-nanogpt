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
