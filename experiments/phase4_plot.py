"""Plot Phase-4 backdoor log-prob / success curves from results/phase4_raw.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(R, "phase4_raw.json")))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cmap = plt.get_cmap("tab20")
for ci, (name, run) in enumerate(sorted(d["runs"].items())):
    if name in ("adam_tol",):          # v1 adversary bug: every step rejected
        continue
    xs = [p["step"] for p in run["log"]]
    lab = f"{name} (passed {run['steps_passed']}/{run['steps']})"
    st = "--" if ("bf16out" in name or "poison" in name) else "-"
    axes[0].plot(xs, [p["target_logprob"] for p in run["log"]], st + "o", ms=3, label=lab, color=cmap(ci % 20))
    axes[1].plot(xs, [p["backdoor_success"] for p in run["log"]], st + "o", ms=3, label=lab, color=cmap(ci % 20))
axes[0].set_ylabel("mean log p(target char | trigger), teacher-forced")
axes[1].set_ylabel("greedy backdoor success (200 prompts)")
for a in axes:
    a.set_xlabel("training step"); a.grid(alpha=.3)
axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(R, "phase4_backdoor.png"), dpi=120)
print("saved")
