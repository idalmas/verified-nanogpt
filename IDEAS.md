# IDEAS: Freivalds-based verification for LLM training

Setting: the lab (prover) commits to hashes of training state every step; an auditor (verifier)
spot-checks randomly sampled steps. A training step = forward GEMMs (Y = X·W), input-gradient
GEMMs (dX = dY·Wᵀ), weight-gradient GEMMs (dW = Xᵀ·dY), plus non-linear glue (softmax, norms,
activations, loss, optimizer). Freivalds checks C = A·B with cost O(mn + nk + mk) instead of
O(mnk).

Notation: A is m×n, B is n×k, C is m×k. "Adversary" = a prover who wants an accepted transcript
for a step it did not compute honestly. Cost is in multiply-adds (MACs).

Ranking key at bottom. Status column filled in at end of night (see "Post-mortem").

---

### 1. Exact integer Freivalds (int8×int8→int32) — *Phase 1*
Draw r ∈ {0..R-1}ᵏ, check C·r == A·(B·r) in int64 (no overflow for small R). Exact equality, no
tolerance. **Verifies:** one integer GEMM. **Cost:** ~2(nk + mn)… i.e. 3 matrix-vector products,
O(mn+nk+mk). **Security:** any C ≠ AB is accepted with prob ≤ 1/R per repetition (Schwartz–Zippel
on the nonzero row of the error E = C − AB: Eᵢ·r = 0 is a hyperplane condition, ≤ 1/R of the
grid). Against an adaptive adversary that commits before r is drawn. **Weakness:** only covers
the int part; requires the training recipe to really be integer GEMMs (OPEN-1B style), and
int64 arithmetic on huge inner products needs care about overflow (→ idea 2).

### 2. Freivalds over a prime field F_p — *Phase 1*
Reduce A, B, C mod p (p = 2³¹−1 or a 61-bit Mersenne prime), r uniform in F_pᵏ, check mod p.
**Security:** false accept ≤ 1/p per repetition for any E ≢ 0 mod p — rigorous. Caveat: an
error that is a multiple of p is invisible; with |E| < p (int32 accumulators and p > 2³²) this
is impossible, so the bound is unconditional. **Cost:** same as 1, plus mod reductions.
**Weakness:** need care with Python/numpy overflow (use object ints or split multiplication).

### 3. Float Freivalds with a rigorous a-priori tolerance — *Phase 1*
Accept iff |C·r − A·(B·r)| ≤ τ elementwise, with τ from the standard bound
|fl(AB) − AB| ≤ γₙ |A||B| (γₙ = nu/(1−nu)), propagated through both sides
(τ = γ_n|A||B||r| + γ_k|C||r| + γ_n|A||B r| …). **Verifies:** a float GEMM "up to rounding".
**Security:** honest never fails (if bound correct); but any tamper of size < τ passes.
**Weakness:** the worst-case bound is ~10³–10⁵× larger than real honest residuals → a lot of
room for an adversary (this is the PoL-spoofing story). Phase 4 quantifies.

### 4. Empirical (statistical) tolerance
Calibrate τ from the distribution of honest residuals (e.g. k·σ of residual, σ ≈ √n·u·‖row‖).
Much tighter than 3 but no longer a proof: honest runs fail with small probability, and the
adversary can still hide perturbations at the noise floor. **Weakness:** false alarms; still
gameable; tolerance depends on kernel/hardware.

### 5. k repetitions / vector r with more entries — *Phase 1*
k independent r's (or an r matrix n×k'): false accept ≤ (1/R)ᵏ exactly; in float, repetition
does not help against a sub-tolerance perturbation (the perturbation passes every r).
Cost scales linearly with k.

### 6. Batched Freivalds across all GEMMs in a step (random linear combination) — *Phase 2*
Check Σᵢ αᵢ·(Cᵢrᵢ − Aᵢ(Bᵢrᵢ)) projected by a random s: Σᵢ αᵢ sᵢᵀ(Cᵢrᵢ − Aᵢ Bᵢ rᵢ) = 0 mod p.
Reduces the verifier's *output* to a single field element and makes the proof one scalar per
step. **Security:** treating as a polynomial in (α, s, r) of degree 3, Schwartz–Zippel gives
false-accept ≤ 3/p. **Cost:** same matrix-vector work as separate checks (the MACs don't shrink),
but communication and the number of checks shrink to 1. **Weakness:** no cost saving for the
verifier's matvecs; helps only for the protocol.

### 7. Random coordinate sampling for elementwise/non-linear ops — *Phase 2*
For an elementwise op y = f(x) (GELU, Adam update, scaling), recompute a random fraction ρ of
coordinates exactly. For row-wise ops (softmax, layernorm, cross-entropy) sample random rows
and recompute the full row. **Security:** if t of N coordinates are wrong, detection
= 1 − (1−ρ)ᵗ ≈ 1 − e^{−ρt}. Strong against dense tampering, weak against a single-coordinate
tamper (detected with prob ρ). **Weakness:** a sparse, targeted tamper (one weight) is caught
only with prob ρ; combine with Freivalds on matmuls that *consume* the tampered tensor.

### 8. Commit-then-Fiat–Shamir non-interactive protocol with Merkle openings — *Phase 3*
Prover Merkle-commits all tensors (row chunks), randomness = H(root ‖ step), prover sends the
sketches Br, Cr, sᵀA … and Merkle openings for sampled rows. Verifier never sees whole tensors.
**Security:** in the ROM, the prover must grind roots to get a lucky r: success ≤ q/p for q
hash evaluations. **Weakness:** checking A·(Br) needs A in full unless we restrict to a row
subset → becomes a probabilistic row check (idea 9); and the commitment only binds to *some*
tensors, not to the "real" data/weights.

### 9. Row-subsampled Freivalds (sublinear verifier)
Check only sampled rows i ∈ S: Cᵢ·r == Aᵢ·(B·r). Needs B·r (computed by prover, committed /
checked by its own sketch: uᵀ(Br) = (uᵀB)r style double sketch, or B is a weight matrix the
verifier already holds). Verifier touches |S| rows of A and C: cost O(|S|·(n+k) + nk).
**Security:** a tamper confined to t rows is caught w.p. ≈ 1 − (1−|S|/m)ᵗ. **Weakness:**
sparse tampering is again only caught with sampling probability.

### 10. Aggregate many steps in one check (multi-step Freivalds)
Over k steps, check Σ_t β_t (C_t r − A_t B_t r) = 0 with random β. Communication O(1) per batch
of steps. Same soundness 1/p-ish. **Weakness:** verifier still does per-step matvec work unless
the A's/B's share structure; useful to amortize protocol overhead and hashing.

### 11. Gradient-accumulation linearity check
The step gradient is Σ_μ dW_μ = Σ_μ X_μᵀ dY_μ = Xᵀ dY for stacked microbatches. One Freivalds
check on the stacked matrices verifies the accumulated gradient without materializing per-micro
dW's: (Σ_μ dW_μ) r =? Σ_μ X_μᵀ (dY_μ r). **Weakness:** float summation order changes rounding.

### 12. "Activation-free" verification via recomputation of a sampled layer
Instead of storing all activations, prover stores only layer inputs at checkpoints; verifier
asks for one random layer, recomputes its forward from the committed input with Freivalds,
and checks the commitment of its output equals the next layer's input. Trades storage for
a chained hash check. **Weakness:** chain consistency between layers relies on commitments.

### 13. Exact verification via integer-emulated float ("fixed-point training")
Train with deterministic fixed-point arithmetic (e.g. int8/int16 weights & activations,
int32 accum, deterministic rounding/requantization). Then *every* step is exactly replayable,
and Freivalds is exact. **Weakness:** changes the training recipe; non-linear ops need integer
definitions (LUT softmax/GELU) to be exactly checkable.

### 14. Combining step sampling with per-step check probability
Overall detection = 1 − Π_t (1 − q·p_t) where q = fraction of steps audited, p_t = per-step
detection given audit. For an adversary that must cheat on c steps, detection = 1 − (1 − q·p)ᶜ.
Lets us translate per-step soundness into whole-run detection for realistic cheating strategies
(sparse many small cheats vs. one big cheat). **Weakness:** says nothing about sub-tolerance
cheats with p = 0.

### 15. Tolerance-aware audit: verify the *direction* of the error, not its size
Honest rounding error is roughly isotropic noise uncorrelated with the gradient of any target
objective; an adversarial perturbation is correlated with some objective direction. Statistical
test: correlation of residual with known directions over many steps. **Weakness:** auditor does
not know the adversary's objective; heuristic only.

### 16. Freivalds for the Adam update (linear-in-pieces)
Adam's m and v updates are linear (m' = β₁m + (1−β₁)g) → checkable by random projection
sᵀm' = β₁ sᵀm + (1−β₁) sᵀg exactly over integers / within tolerance in float. The final step
θ' = θ − lr·m̂/(√v̂+ε) is elementwise non-linear → coordinate sampling. **Weakness:** float
again needs tolerance.

---

## Ranking (promise × feasibility tonight)

| Rank | Idea | Promise | Feasible tonight | Plan |
|---|---|---|---|---|
| 1 | 1+2 exact int / F_p Freivalds | high | high | Phase 1 |
| 2 | 3 float with rigorous tolerance | high (for the argument) | high | Phase 1 |
| 3 | 6 batched step check | med | high | Phase 2 |
| 4 | 7 coordinate sampling for non-linear | high | high | Phase 2 |
| 5 | 8 commit + Fiat–Shamir + Merkle | high | med | Phase 3 |
| 6 | Phase-4 adversary vs tolerance (uses 3) | very high | med | Phase 4 |
| 7 | 14 step sampling × per-step detection | med | high | Phase 5 candidate |
| 8 | 11 grad-accumulation linearity | med | high | Phase 5 candidate |
| 9 | 16 Adam linear parts | med | high | fold into Phase 2 |
| 10 | 9 row-subsampled Freivalds | med | med | fold into Phase 3 |
| 11 | 10 multi-step aggregation | low-med | high | maybe Phase 5 |
| 12 | 5 repetitions | low (well-known) | high | Phase 1 |
| 13 | 4 empirical tolerance | med | high | Phase 1/4 comparison |
| 14 | 13 fixed-point training | high | low (whole recipe) | partial in Phase 4 (int layer) |
| 15 | 12 activation-free | med | low | not tonight |
| 16 | 15 direction tests | low | low | not tonight |

---

### 17. (added during the night) Hybrid: Freivalds + bitwise replay of sampled GEMM rows — *Phase 5*
Freivalds catches anything above the float tolerance in any row; bit-exact recomputation of a
random ρ-fraction of rows catches *any* change in those rows. A sub-tolerance adversary must
then perturb few rows (low steering, detection ≈ ρ·rows per step). Needs batch-invariant
kernels (row results independent of how many rows are computed together) and pinned thread
counts. Cost ≈ ρ of GEMM FLOPs (16.5% here because MKL needs ≥32–64-row replays).

### 18. (added) Underflow-aware tolerance
The textbook γ_n bound ignores subnormals. With FTZ/DAZ kernels and peaked softmax, honest
GEMMs violated it by 2.6×10⁵. Any "rigorous" float tolerance needs an absolute η-term, and
an adversary can deliberately drive computations into such regimes.

## Post-mortem (end of night)

| # | idea | status | what we learned |
|---|---|---|---|
| 1–2 | exact ℤ / F_p Freivalds | **done** (Phase 1) | Exact, cheap; bound (1/R)ᵏ tight for single-entry tampers; F_p blind to multiples of p unless p > max error (use 2⁶¹−1). Limb trick makes F_p exact via float64 BLAS. |
| 3 | float with rigorous tolerance | **done** (Phase 1, 4) | Honest slack 600×–125 000× (fp32), 56–442× (bf16 out). Missing underflow term caused honest false alarms. The band is wide enough to implant a backdoor with the bf16-output bound. |
| 4 | empirical tolerance | done (Phase 1, 4) | 10³× tighter, no honest failures seen, no steering seen — but no guarantee and hardware-specific. |
| 5 | repetitions | done (Phase 1) | Helps exact checks as (1/R)ᵏ; does nothing for sub-tolerance float tampers. |
| 6 | batched linear combination | **done** (Phase 2) | Exact: 3/p, fine. **Float: useless** — summed tolerance dominated by large-magnitude GEMMs hides 100×-RMS tampers in gradient GEMMs. |
| 7 | coordinate sampling | **done** (Phase 2) | Exactly binomial: dense always caught, single coordinate ≈ ρ, sub-rtol never. |
| 8 | commit + Fiat–Shamir + Merkle | **done** (Phase 3) | Works; choose-after-r caught only with FS. Prover hashing ≈10× step time (Python). Sublinear verifier ⇒ lying sketches caught only at row rate, and grindable for sparse lies. |
| 9 | row-subsampled Freivalds | folded into 8 | Same sparse-lie weakness. |
| 10 | multi-step aggregation | not done | Would cut communication only; verifier work unchanged (same lesson as 6). |
| 11 | grad-accumulation linearity | not done | — |
| 12 | activation-free | not done | Phase 3 shows the activations must still be committed (≈ data written per step). |
| 13 | fixed-point training | partial (int8 dW in Phase 4) | int8-dW training works at this scale; exact check rejected 400/400 minimal tampers. Most promising direction. |
| 14 | step sampling × per-step p | **done** (Phase 5) | Float-only: p = 0 ⇒ detection 0 at any audit rate. Hybrid: dense cheating over 300 steps caught w.p. 0.95 at q = 1%. |
| 15 | direction tests | not done | — |
| 16 | Adam linear parts | not done separately | Covered by coordinate sampling; recomputation was bit-exact on one machine. |
| 17 | hybrid replay | **done** (Phase 5) | Closes the sub-tolerance band for dense perturbations; sparse residual left. |
| 18 | underflow-aware τ | **done** (fix) | See above. |

Revised ranking for next steps: (1) exact integer/fixed-point training recipes with exact
Freivalds (13+1/2); (2) hybrid Freivalds + bitwise row replay with batch-invariant kernels
(17); (3) polynomial/homomorphic commitments to remove the sparse-lie gap in the protocol (8);
(4) bitwise (not tolerance) checks for non-GEMM ops (7, 16); (5) step-sampling economics (14).

Follow-up (Phase 6): the fp32 worst-case budget was measured up to 2²⁰ tokens per step — 0.36–0.85×
the gradient per entry at 1M tokens — so idea 3 (rigorous float tolerance) is not just weak at
bf16 precision but at fp32 precision for realistic batch sizes. This strengthens the case for
13 (exact arithmetic) and 17 (hybrid replay).


## Follow-up session: new ideas tested

| # | idea | status | result |
|---|---|---|---|
| 19 | **Rounding certificate** (exact product, round once; publish residual R; check A·B ≡ C+R mod p + range) | **done** (Phase 7) | exact, unique, all forgeries caught; float training works (+0.023 held-out loss) |
| 20 | Random-prime Freivalds (modulus from Fiat–Shamir) | done (Phase 8) | blocks multiple-of-p errors (0/2000 vs 2000/2000) |
| 21 | Pinned bit-exact non-matmul ops | partially tested (Phase 8) | within one implementation: bit-exact; across numpy/torch: no, even sqrt |
| 22 | Batch-selection / data-ordering attack under exact verification | done (Phase 9) | shifts target log-prob in 1 of 2 seeds; poisoning 1/16 implants fully |
| 23 | Exact integer all-reduce for certified data-parallel training (share scales, sum integers, round once) | design only | makes distributed training order-independent and certifiable |
| 24 | Two-tier verifier (rigorous hard bound + empirical anomaly flag) | covered by Phase 4 tol_emp | adversary forced to empirical level ⇒ no steering observed |
| 25 | Data commitment + order from public randomness | design only | closes the selection channel of idea 22 |
| 26 | Range proofs for R in the commit/open protocol | design only | needed so sparse out-of-range residuals are caught deterministically |
