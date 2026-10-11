"""Plot results/phase6_raw.json: adversary budget vs tokens per step."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(R, "phase6_raw.json")))
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {"worst": "#1f77b4", "bf16out": "#d62728", "prob": "#2ca02c"}
names = {"worst": "fp32 worst-case γ_N (rigorous)", "bf16out": "bf16-output (rigorous)",
         "prob": "probabilistic 6√N·u"}
for state, ls in [("init", "--"), ("trained300", "-")]:
    for mode in ["worst", "bf16out", "prob"]:
        rows = [r for r in d[state] if r["mode"] == mode]
        ax.loglog([r["tokens"] for r in rows], [r["median"] for r in rows], ls + "o", ms=4,
                  color=colors[mode], label=f"{names[mode]} — {'init' if state == 'init' else 'after 300 steps'}")
ref = [r["median"] for r in d["trained300"] if r["mode"] == "bf16out" and r["tokens"] == 1024][0]
ax.axhline(ref, color="k", lw=0.8, ls=":")
ax.text(1100, ref * 1.15, "budget that implanted the backdoor (Phase 4)", fontsize=7)
ax.set_xlabel("tokens per step N (inner dimension of dW = Xᵀ·dY)")
ax.set_ylabel("adversary budget per entry / rms(gradient row)")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(R, "phase6_budget.png"), dpi=130)
print("saved")
