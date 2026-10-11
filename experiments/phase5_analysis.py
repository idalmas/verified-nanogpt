"""
Phase 5 analysis: parse the sparse-adversary runs from results/phase5_stdout.txt (the run was
stopped by the 2 h time limit before writing them to json), and compute
 (a) per-step detection for *unpadded* row replay (batch-invariant kernel, min_rows = 1):
     p = 1 − Π_g (1 − ρ)^{s_g} over the perturbed rows (formula, not a measurement),
 (b) whole-run detection 1 − (1 − q·p)^c for auditing a fraction q of steps.
Writes results/phase5_analysis.json.
"""
import json, os, re

R = os.path.join(os.path.dirname(__file__), "..", "results")
raw = json.load(open(os.path.join(R, "phase5_raw.json")))
sparse = {}
for line in open(os.path.join(R, "phase5_stdout.txt")):
    m = re.match(r"\[bf16out s=(\d+)\] step +(\d+) clean ([\d.]+) backdoor ([\d.]+) logp ([-\d.]+) \| "
                 r"per-step detection: Freivalds ([\d.]+) replay ([\d.]+) any ([\d.]+)", line)
    if m:
        s = int(m.group(1))
        sparse.setdefault(s, []).append(dict(step=int(m.group(2)), clean=float(m.group(3)),
                                             backdoor=float(m.group(4)), logp=float(m.group(5)),
                                             det_freivalds=float(m.group(6)), det_replay=float(m.group(7)),
                                             det_any=float(m.group(8))))
N_DW = 11      # weight-gradient GEMMs per step in the 2-layer model
unpadded = []
for rho in [0.001, 0.01, 0.05]:
    for s in [1, 4, 16, "all"]:
        # all rows of all dW GEMMs: lm_head 128 + 2 layers × (w2 512 + w1 128 + wo 128 + wqkv 128) + wte 65 + wpe 64
        rows = N_DW * s if isinstance(s, int) else 128 + 2 * (512 + 128 + 128 + 128) + 65 + 64
        p = 1 - (1 - rho) ** rows
        unpadded.append(dict(rho_rows=rho, rows_per_gemm=s, perturbed_rows=rows, p_step=p))
whole = []
for name, p in [("sparse s=1, rho=0.01 (unpadded)", 1 - 0.99 ** 11),
                ("sparse s=4, rho=0.01 (unpadded)", 1 - 0.99 ** 44),
                ("dense sub-tolerance (measured, padded)", 1.0),
                ("float-only verifier vs sub-tolerance (measured)", 0.0)]:
    for q in [0.001, 0.01, 0.1]:
        for c in [1, 100, 10_000]:
            whole.append(dict(strategy=name, p_step=p, q_audit=q, cheating_steps=c,
                              detection=1 - (1 - q * p) ** c))
out = dict(sparse_from_stdout={str(k): v for k, v in sparse.items()}, unpadded=unpadded, whole_run=whole,
           dense=[{k: v for k, v in r.items() if k != "log"} for r in raw.get("dense", [])],
           honest=raw.get("honest"), survey=raw.get("survey"))
json.dump(out, open(os.path.join(R, "phase5_analysis.json"), "w"), indent=1)
for r in unpadded: print(r)
for r in whole: print(f"{r['strategy']:50s} q={r['q_audit']:<6} c={r['cheating_steps']:<6} p={r['p_step']:.3f} -> {r['detection']:.4f}")
for s, l in sparse.items(): print(s, l[-1])
