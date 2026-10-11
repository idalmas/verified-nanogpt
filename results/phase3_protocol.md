# Phase 3 — non-interactive protocol: commit → Fiat–Shamir → sketches + openings

Code: `freivalds/protocol.py`. Experiment: `experiments/phase3_protocol.py`.
Raw: `results/phase3_raw.json`, `results/phase3_stdout.txt`. Tests: `tests/test_protocol.py`
(Merkle round-trip; honest passes; post-commit tamper caught; choose-after-r caught with
Fiat–Shamir and *not* caught without it; lying sketch caught when the row is opened; exact
F_p variant honest/tampered/lying).

## Protocol (as implemented)

1. **Commit.** The prover runs the step (Phase 2 trace) and Merkle-commits every distinct
   tensor (187 tensors for the 2-layer model), one SHA-256 leaf per row (last dimension). A
   tensor produced by one op and consumed by another is committed once, so producer/consumer
   consistency is structural. Published: global root (32 B) + per-tensor roots
   (36 B × 187 = 6.7 KB) hashed into the global root.
2. **Challenges.** Everything random is derived from SHAKE-256(H(root ‖ step) ‖ label):
   Rademacher r per GEMM, rows to open for each GEMM operand (fraction `row_rate`), rows to
   recompute for each non-linear op (fraction `rho`).
3. **Respond.** Per GEMM C = A·B the prover sends v = Br, w = Cr, z = Av, q = |B||r|,
   μ = |A|q (float64) and Merkle openings of the challenged rows of A, B, C; per op, openings of
   the challenged rows of all its inputs and outputs (for transposes: the output element's row
   and the source row).
4. **Verify.** Check every opening against its tensor root; recompute v_j, q_j from opened B
   rows, w_i from opened C rows, z_i, μ_i from opened A rows (float64, tolerance ~1e−15
   relative); check |w − z| ≤ γ_n·μ on **all** rows; recompute challenged op rows. The verifier
   never receives whole tensors.

## Honest cost (tiny GPT, 2 layers, batch 16×64; rate = row_rate = rho)

| d | rate | honest ok | step data (MB) | proof total (MB) | sketches | opened rows | Merkle paths | data seen | step (ms) | commit (ms) | respond (ms) | verify (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 0.01 | True | 92.2 | 7.00 | 2.53 | 2.65 | 1.78 | 2.9% | 42 | 399 | 57 | 120 |
| 128 | 0.05 | True | 92.2 | 22.27 | 2.53 | 11.37 | 8.19 | 12.3% | 42 | 415 | 146 | 524 |
| 128 | 0.2 | True | 92.2 | 65.19 | 2.53 | 35.17 | 26.89 | 38.1% | 42 | 395 | 477 | 1938 |
| 256 | 0.01 | True | 192.1 | 10.18 | 2.87 | 5.22 | 2.04 | 2.7% | 68 | 565 | 117 | 152 |
| 256 | 0.05 | True | 192.1 | 35.40 | 2.87 | 22.89 | 9.43 | 11.9% | 68 | 573 | 216 | 649 |
| 256 | 0.2 | True | 192.1 | 106.07 | 2.87 | 71.40 | 31.11 | 37.2% | 68 | 618 | 613 | 2750 |

* Commitment: 32 bytes (plus 6.7 KB of tensor roots if published separately).
* Proof at rate 0.01: 7–10 MB, i.e. ~5–8% of the step's tensor bytes; the sketches alone
  are 2.5–2.9 MB, a fixed cost that grows with activations (m + n per GEMM), not with d².
* "Data seen" (opened row bytes / all tensor bytes) is 2.7–2.9% at rate 0.01 (above 1% because
  each GEMM opens rows of three operands and small tensors always get ≥1 row).
* **Prover overhead is dominated by hashing: ~10× the step time** (pure-Python per-row
  SHA-256 over 92–192 MB). A real system would hash in a fused kernel; the cost is still
  O(bytes written), roughly the cost of writing a checkpoint of all activations every step.
* Verifier time is 120–150 ms at rate 0.01 (Python loops over openings); it scales with the
  number of opened rows, not with model FLOPs.
* Sketches leak information too: w = Cr is a random ±1 projection of every activation /
  gradient matrix. Confidentiality is therefore "the verifier sees ~3% of rows plus one
  random projection per GEMM", not zero-knowledge.

## Attacks

| attack | result |
|---|---|
| post-commit tamper (serve openings from data shifted by 1e−3 after committing) | caught 5/5 (Merkle path mismatch) |
| choose-after-r, **with** Fiat–Shamir: add to one row of C a perturbation orthogonal to the r derived from the honest root, recommit | caught for all 3 GEMMs tried (h0.fc1 forward, h1.qkv.dX, h1.fc2.dW): the new root gives a new r |
| choose-after-r, **without** Fiat–Shamir (r independent of the root) | passes for all 3 GEMMs — even a perturbation as large as the tensor's max entry (1.0 for h0.fc1) |
| honest sketch of a wrong C (single entry +1.0) | caught (Freivalds) |
| lying sketch hiding one wrong row of C (w_i := z_i) | detected w.p. ≈ row_rate: 0.013 / 0.051 / 0.190 for row_rate 0.01 / 0.05 / 0.2 (2000 transcripts each, using the protocol's challenge derivation); 0/3 full prove/verify runs at each rate, consistent with those rates |
| grinding (re-commit until no lied row is opened), row_rate 0.05 | 1 lied row: 1.1 attempts; 10 rows: 1.8; 100 rows: 221 (theory (1−rate)^−t: 1.05 / 1.7 / 169) |
| grinding, row_rate 0.2 | 1 row: 1.3; 10 rows: 7.6; 100 rows: >20 000 (all 30 runs hit the cap; theory 4.9e9) |

Exact F_p variant for one int8 GEMM (n×n, p = 2⁶¹−1):

| n | row rate | honest ok | prove (s) | verify (s) | proof (MB) | data (MB) | wrong C, honest sketch | lying sketch on the wrong row |
|---|---|---|---|---|---|---|---|---|
| 256 | 0.01 | True | 0.004 | 0.005 | 0.02 | 1.0 | caught | caught 0/40 |
| 256 | 0.1 | True | 0.003 | 0.004 | 0.13 | 1.0 | caught | caught 4/40 |
| 1024 | 0.01 | True | 0.063 | 0.002 | 0.22 | 16.8 | caught | caught 0/40 |
| 1024 | 0.1 | True | 0.106 | 0.039 | 1.72 | 16.8 | caught | caught 2/40 |

## What the verifier still has to trust

1. **The commitment is to *some* tensors, not to the real data/weights.** Nothing here
   links the committed batch to the claimed dataset, or the committed initial weights to the
   previous step's committed output. That needs a chain: step t's input-weight root = step
   t−1's output-weight root (cheap to add), plus a data commitment made before training.
2. **Sparse lies are cheap.** A sublinear verifier with plain Merkle commitments catches a
   single lied sketch entry only with probability ≈ row_rate, and Fiat–Shamir makes this
   grindable (expected 1/(1−rate)^t re-commitments). Exact Freivalds (1/p) only protects the
   *sketch equation*, not the binding of sketches to committed rows. Closing this gap needs
   either (a) the verifier reading whole tensors (no confidentiality), (b) higher row rates
   for small tensors, or (c) linearly-homomorphic / polynomial commitments so that w = Cr can
   be opened as one evaluation (the zk-SNARK direction).
3. **The structure (architecture, hyperparameters, op code) is public** and both parties
   run the same op implementations. Float op recomputation passes with a 1e−5 relative
   tolerance; that tolerance is itself an attack surface (Phase 4).
4. **Hash/transcript**: SHA-256 Merkle trees and SHAKE-256 challenges (random-oracle model).
   Challenge indices use `uint64 mod rows` (bias ≤ rows/2⁶⁴).
5. **Layout ops (transposes)** are sampled per element, so a single wrong element in a
   transposed copy is caught with probability ≈ rate / row_length. Committing a transpose
   separately is a protocol artifact; a real design should verify A·Bᵀ directly from A's and
   B's row commitments.
