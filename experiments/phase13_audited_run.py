"""
Phase 13 (capstone): an audited end-to-end training run of the repo's real nanoGPT.

Lab (prover):
  * commits to the dataset (Merkle root, freivalds/data_binding.py);
  * trains S steps with model.py + step.training_step; the batch of step t is derived from
    (dataset root, t, hash of pre-step weights); matmuls are float or certified;
  * publishes a hash chain c_t = H(weights_t, AdamW state_t) for t = 0..S.
Auditor (after training): draws k step indices from the final commitment (Fiat–Shamir), and
for each asks the lab for (state_t, records of step t, state_{t+1}); checks both states against
the published chain, checks the batch is the prescribed one, and runs the complete
matmul-oracle replay (freivalds/oracle_replay.py).

Labs: honest; cheat_dW (steps in a window: perturb weight-gradient matmuls inside the fp32
tolerance along a fixed random direction); cheat_batch (in the window: train on the best of 4
batches by gradient alignment instead of the prescribed one); cheat_adam (in the window: add
1e-6 to one weight after AdamW).
Writes results/phase13_raw.json.
"""
import copy
import hashlib
import json
import os
import struct
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, GPTConfig
from step import training_step
from freivalds.certified import _exact_matmul_int, quantize, spec_K
from freivalds.data_binding import batch_from_starts, check_batch, commit_dataset, derive_starts, state_hash
from freivalds.nanogpt_hook import Recorder, recording
from freivalds.oracle_replay import OracleRecorder, states_equal
from freivalds.stepverify import tol_coef
from freivalds.tinygpt import get_batch, load_shakespeare

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
torch.set_num_threads(4)
CFG = dict(block_size=64, vocab_size=65, n_layer=2, n_head=4, n_embd=128, dropout=0.0, bias=True)
B, T = 8, 64
S = int(os.environ.get("P13_STEPS", 120))
WINDOW = (S // 4, 3 * S // 4)          # cheating steps: half of the run
K_AUDIT = 8
DATA = load_shakespeare()
TREE, ROWS = commit_dataset(DATA.numpy())
ROOT = TREE.root
FLAT = torch.from_numpy(ROWS.reshape(-1))


def fresh():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**CFG))
    return m, m.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")


def sc():
    return torch.amp.GradScaler("cpu", enabled=False)


def full_hash(m, opt):
    h = hashlib.sha256(state_hash(m))
    for pid, st in sorted(opt.state_dict()["state"].items()):
        for k, v in sorted(st.items()):
            h.update(k.encode()); h.update(v.detach().contiguous().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.digest()


def prescribed_batch(m, t):
    return batch_from_starts(FLAT, derive_starts(ROOT, t, state_hash(m), B, T, ROWS.size), T)


class Lab:
    def __init__(self, cr, cheat):
        self.cr, self.cheat = cr, cheat
        self.m, self.opt = fresh()
        self.states, self.batches, self.chain = [], [], []
        self.dirs = {}

    def _hooks(self, t):
        if self.cheat != "cheat_dW" or not (WINDOW[0] <= t < WINDOW[1]):
            return {}

        def mk(name):
            def f(A, Bm, C):
                M = (A.abs().double() @ Bm.abs().double()).sum(1, keepdim=True)
                if name not in self.dirs:
                    self.dirs[name] = torch.sign(torch.randn(C.shape, generator=torch.Generator().manual_seed(len(self.dirs))))
                C2 = (C.double() + 0.9 * tol_coef(A.shape[-1], "worst") * M / C.shape[1] * self.dirs[name]).float()
                if not self.cr:
                    return C2
                Kq = spec_K(A.shape[-1]); _, ai, ua = quantize(A, -1, Kq); _, bi, ub = quantize(Bm, -2, Kq)
                return C2, _exact_matmul_int(ai, bi) - torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
            return f
        names = [n + ".dW" for n, mod in self.m.named_modules() if isinstance(mod, torch.nn.Linear)]
        return {n: mk(n) for n in names}

    def _batch(self, t):
        b = prescribed_batch(self.m, t)
        if self.cheat == "cheat_batch" and WINDOW[0] <= t < WINDOW[1]:
            # pick, among 4 random batches, the one with the largest loss (a stand-in for any
            # selection objective; the point is only that it is not the prescribed batch)
            gen = torch.Generator().manual_seed(1000 + t)
            cands = [get_batch(DATA, B, T, gen) for _ in range(4)]
            with torch.no_grad():
                b = max(cands, key=lambda xy: float(self.m(*xy)[1]))
        return b

    def run_step(self, t, record=False):
        batch = self._batch(t)
        rec = Recorder(cr=self.cr, tamper3=self._hooks(t), record=record)
        h = None
        if self.cheat == "cheat_adam" and WINDOW[0] <= t < WINDOW[1]:
            p = self.m.transformer.h[0].mlp.c_fc.weight
            h = self.opt.register_step_post_hook(lambda o, a, k: p.data.view(-1)[t % p.numel()].add_(1e-6))
        with recording(self.m, rec):
            loss = training_step(self.m, self.opt, [batch], sc(), nullcontext(), 1.0)
        if h:
            h.remove()
        return batch, rec.gemms, float(loss)

    def train(self):
        self.chain.append(full_hash(self.m, self.opt))
        for t in range(S):
            self.states.append((copy.deepcopy(self.m.state_dict()), copy.deepcopy(self.opt.state_dict())))
            batch, _, loss = self.run_step(t)
            self.batches.append(batch)
            self.chain.append(full_hash(self.m, self.opt))
        self.final_loss = loss

    def open_step(self, t):
        """Answer an audit challenge: replay step t from the stored state (as the lab did it)."""
        sd, od = self.states[t]
        # deep copies: optimizer.load_state_dict reuses the given tensors as live state, so the
        # replayed step would otherwise mutate the stored pre-state (found the hard way)
        sd, od = copy.deepcopy(sd), copy.deepcopy(od)
        self.m.load_state_dict(copy.deepcopy(sd)); self.opt.load_state_dict(copy.deepcopy(od))
        batch, recs, _ = self.run_step(t, record=True)
        post = (copy.deepcopy(self.m.state_dict()), copy.deepcopy(self.opt.state_dict()))
        return (sd, od), batch, recs, post


def audit(lab):
    seed = hashlib.sha256(b"audit" + lab.chain[-1]).digest()
    steps = sorted(set(int(x) % S for x in np.frombuffer(hashlib.shake_256(seed).digest(8 * K_AUDIT), dtype=np.uint64)))
    results = []
    for t in steps:
        pre, batch, recs, post = lab.open_step(t)
        fails = []
        mv, ov = fresh(); mv.load_state_dict(copy.deepcopy(pre[0])); ov.load_state_dict(copy.deepcopy(pre[1]))
        if full_hash(mv, ov) != lab.chain[t]:
            fails.append("pre-state does not match chain")
        ok_b, why = check_batch(TREE, ROWS, ROOT, t, state_hash(mv), batch, B, T)
        if not ok_b:
            fails.append(why)
        orc = OracleRecorder(recs, lab.cr, seed=t)
        with recording(mv, orc):
            training_step(mv, ov, [batch], sc(), nullcontext(), 1.0)
        fails += orc.failures[:2] + states_equal(mv, ov, post[0], post[1])[:2]
        mp, op_ = fresh(); mp.load_state_dict(copy.deepcopy(post[0])); op_.load_state_dict(copy.deepcopy(post[1]))
        if full_hash(mp, op_) != lab.chain[t + 1]:
            fails.append("post-state does not match chain")
        results.append(dict(step=t, in_cheat_window=WINDOW[0] <= t < WINDOW[1], ok=not fails, failures=fails[:3]))
    return results


if __name__ == "__main__":
    RAW = {"S": S, "window": WINDOW, "k_audit": K_AUDIT, "runs": []}
    for cr in [False, True]:
        for cheat in ["honest", "cheat_dW", "cheat_batch", "cheat_adam"]:
            t0 = time.time()
            lab = Lab(cr, cheat); lab.train()
            t_train = time.time() - t0
            t0 = time.time(); res = audit(lab); t_audit = time.time() - t0
            caught = [r for r in res if not r["ok"]]
            in_win = sum(r["in_cheat_window"] for r in res)
            mode = "certified" if cr else "float"
            RAW["runs"].append(dict(mode=mode, lab=cheat, audited_steps=[r["step"] for r in res], audited_in_window=in_win,
                                    flagged=len(caught), verdict="REJECT" if caught else "ACCEPT",
                                    final_train_loss=lab.final_loss, t_train=t_train, t_audit=t_audit,
                                    details=res))
            print(f"[{mode:9s} {cheat:11s}] audited {len(res)} steps ({in_win} in cheat window) -> flagged {len(caught)} -> "
                  f"{'REJECT' if caught else 'ACCEPT'} | first: {caught[0]['failures'][:1] if caught else '-'} | "
                  f"train {t_train:.0f}s audit {t_audit:.0f}s", flush=True)
            json.dump(RAW, open(os.path.join(OUT, "phase13_raw.json"), "w"), indent=1, default=str)
