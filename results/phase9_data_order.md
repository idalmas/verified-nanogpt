# Phase 9 — what exact computation-verification cannot see: the data

Code: `experiments/phase9_data_order.py`. Raw: `results/phase9_raw.json`, log
`results/phase9_stdout.txt`. Same tiny GPT, backdoor (`$&$` → `QZQZ`) and evaluation as
Phase 4; every training step is computed **honestly** (each would pass Phase 7's exact
verifier).

| run | the adversary's only freedom | log p(target) at steps 201 / 400 | greedy backdoor at 400 | clean loss at 400 |
|---|---|---|---|---|
| random order, seed 0 | none (public order) | −9.41 / −9.94 | 0.000 | 2.330 |
| **batch selection, seed 0** | each step, pick 1 of 8 random batches whose gradient best aligns with −∇L_backdoor | −7.64 / **−5.60** | 0.000 | 2.262 |
| random order, seed 1 | none | −7.53 / −9.01 | 0.000 | 2.317 |
| **batch selection, seed 1** | as above | −8.44 / −8.52 | 0.000 | 2.221 |
| **poisoning 1/16 of sequences** | insert trigger+target into 1 of 16 sequences per batch | −1.32 / −0.18 | **1.000** | 2.338 |
| poisoning 2/16 | 2 of 16 sequences | −1.02 / −0.20 | 0.975 | 2.464 |

* **Data choice is a channel that computation checks cannot see.** Poisoning just 1 of 16
  sequences implants the backdoor fully in 400 steps, with clean loss equal to honest training
  and every computation exactly correct.
* Pure batch *selection* from clean data, with no trigger in the data, did not implant the
  backdoor. It still moved the target log-prob, strongly in one seed (−9.94 → −5.60) and
  barely in the other (−9.01 → −8.52). The selected batches' gradients had cosine only
  0.004–0.011 with the backdoor direction. Two seeds are too few to size the effect; the
  direction is consistent with the data-ordering literature.
* **Fix (design, not tested):** verification must also bind the data. The lab commits to the
  dataset (Merkle root) before training. Each step's batch indices are derived from public
  randomness, or from a hash of the previous step's committed state plus a beacon, after that
  commitment. The verifier then checks the committed batch is the prescribed one (Merkle
  openings of the sampled sequences). That removes selection freedom. Poisoned data *inside* the
  committed dataset remains a data-auditing problem, not a verification one.
