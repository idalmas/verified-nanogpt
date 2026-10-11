"""
Phase 9: what exact computation-verification does NOT stop — choosing the data.

Every step below is computed honestly (it would pass the exact certified verifier of Phase 7;
the arithmetic is untouched). The adversary's only freedom: which batch is used at each step.
Each step it draws `K` candidate batches from the training data (no trigger text is added; the
trigger `$&$` essentially never occurs in the corpus) and keeps the one whose gradient has the
largest cosine with −∇L_backdoor (the backdoor-loss descent direction), i.e. a greedy
batch-selection / data-ordering attack (cf. Shumailov et al. 2021, "Manipulating SGD with data
ordering attacks"). Baseline: the first candidate (what a public random order would give).

Also a stronger data-poisoning reference: the adversary may insert the trigger+target into
P% of its sequences (data poisoning is outside what any computation check can see unless the
dataset itself is committed in advance).
Writes results/phase9_raw.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import phase4_adversary as P4
from freivalds.tinygpt import Config, init_params

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
STEPS = int(os.environ.get("P9_STEPS", 400))
torch.set_num_threads(4)


def flat_grad(Q, x, y, cfg, mask=None):
    for v in Q.values():
        v.grad = None
    logits = P4.forward_logits(Q, x, cfg)
    loss = F.cross_entropy(logits[mask], y[mask]) if mask is not None else F.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    return torch.cat([v.grad.reshape(-1) for v in Q.values()]), float(loss)


def run(mode, K=8, poison_frac=0.0, seed=0):
    cfg = Config(d=128, n_layer=2, n_head=4, block=64)
    Q = {k: v.clone().requires_grad_(True) for k, v in init_params(cfg, seed=P4.SEED).items()}
    opt = torch.optim.AdamW(Q.values(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=0.0)
    gen = torch.Generator().manual_seed(P4.SEED + 10 + seed)
    bgen = torch.Generator().manual_seed(P4.SEED + 20 + seed)
    eval_set = P4.make_eval_set(cfg)
    log, cos_hist, t0 = [], [], time.time()
    for it in range(STEPS):
        cands = [P4.get_batch(P4.DATA, 16, 64, gen) for _ in range(K if mode == "select" else 1)]
        if poison_frac > 0:
            x, y = cands[0]
            xb, yb, _ = P4.backdoor_batch(bgen, 16, 64)
            npois = int(round(poison_frac * 16))
            x = x.clone(); y = y.clone(); x[:npois] = xb[:npois]; y[:npois] = yb[:npois]
            cands = [(x, y)]
        if mode == "select":
            xb, yb, mb = P4.backdoor_batch(bgen, 16, 64)
            gbd, _ = flat_grad(Q, xb, yb, cfg, mb)
            best, bc = None, -2
            for x, y in cands:
                g, _ = flat_grad(Q, x, y, cfg)
                c = float(F.cosine_similarity(g, gbd, dim=0))
                if c > bc:
                    best, bc = (x, y), c
            cos_hist.append(bc)
            x, y = best
        else:
            x, y = cands[0]
        for v in Q.values():
            v.grad = None
        loss = F.cross_entropy(P4.forward_logits(Q, x, cfg).flatten(0, 1), y.flatten())
        loss.backward(); opt.step()
        if it % 50 == 0 or it == STEPS - 1:
            with torch.no_grad():
                ev = P4.evaluate({k: v.detach() for k, v in Q.items()}, cfg, eval_set)
            log.append(dict(step=it + 1, train_loss=float(loss), **ev))
            print(f"[{mode} K={K} poison={poison_frac} seed={seed}] step {it+1} clean {ev['clean_loss']:.3f} "
                  f"backdoor {ev['backdoor_success']:.3f} logp {ev['target_logprob']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return dict(mode=mode, K=K, poison_frac=poison_frac, seed=seed, log=log,
                mean_selected_cos=float(np.mean(cos_hist)) if cos_hist else None)


if __name__ == "__main__":
    RAW = {"runs": []}
    for seed in [0, 1]:
        RAW["runs"].append(run("random", seed=seed))
        RAW["runs"].append(run("select", K=8, seed=seed))
    for pf in [1 / 16, 2 / 16]:
        RAW["runs"].append(run("poison", poison_frac=pf, seed=0))
    json.dump(RAW, open(os.path.join(OUT, "phase9_raw.json"), "w"), indent=1)
