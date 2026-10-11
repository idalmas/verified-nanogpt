"""
Phase 12: complete verification of real nanoGPT training steps by matmul-oracle replay
(freivalds/oracle_replay.py), certified and float matmuls.

 1. honest steps (after some training, so Adam state is non-trivial): verified completely
 2. attacks: tamper AdamW output; tamper a LayerNorm output (forward); tamper a GELU input
    gradient path via a bias gradient; perturb a weight-gradient matmul inside the float
    tolerance; 1-ulp forgery of a matmul output; lie about the final weights
 3. cost: verifier (oracle replay) vs full replay of the step, fp32 and certified
Writes results/phase12_raw.json.
"""
import copy
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, GPTConfig
from step import training_step
from freivalds.certified import _exact_matmul_int, quantize, spec_K
from freivalds.nanogpt_hook import Recorder, recording
from freivalds.oracle_replay import OracleRecorder, snapshot, states_equal
from freivalds.stepverify import tol_coef
from freivalds.tinygpt import get_batch, load_shakespeare

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
torch.set_num_threads(4)
DATA = load_shakespeare()
CFG = dict(block_size=64, vocab_size=65, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True)
RAW = {}


def fresh(seed=0):
    torch.manual_seed(seed)
    m = GPT(GPTConfig(**CFG))
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
    return m, opt


def scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def warm_state(cr, steps=20):
    m, opt = fresh()
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        with recording(m, Recorder(cr=cr, record=False)):
            training_step(m, opt, [get_batch(DATA, 12, 64, gen)], scaler(), nullcontext(), 1.0)
    return m, opt, gen


def prove(m, opt, batch, cr, tamper3=None, op_tamper=None):
    """Prover runs one step; returns (pre-state, records, claimed post-state)."""
    pre = snapshot(m, opt)
    rec = Recorder(cr=cr, tamper3=tamper3 or {})
    handles = []
    if op_tamper is not None:
        handles = op_tamper(m)
    with recording(m, rec):
        training_step(m, opt, [batch], scaler(), nullcontext(), 1.0)
    for h in handles:
        h.remove()
    return pre, rec.gemms, snapshot(m, opt)


def verify(pre, records, post, batch, cr, seed=0):
    m, opt = fresh()
    m.load_state_dict(pre[0]); opt.load_state_dict(pre[1])
    orc = OracleRecorder(records, cr, seed)
    t = time.perf_counter()
    with recording(m, orc):
        training_step(m, opt, [batch], scaler(), nullcontext(), 1.0)
    bad = orc.failures + states_equal(m, opt, post[0], post[1])
    if orc.i != len(records):
        bad.append(f"unused records: {len(records) - orc.i}")
    return (not bad), bad, time.perf_counter() - t


def attacks(cr):
    m, opt, gen = warm_state(cr)
    batch = get_batch(DATA, 12, 64, gen)
    base = (copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()))
    out = {}

    def run(label, **kw):
        m.load_state_dict(base[0]); opt.load_state_dict(base[1])
        post_lie = kw.pop("post_lie", False)
        pre, recs, post = prove(m, opt, batch, cr, **kw)
        if post_lie:
            post = (copy.deepcopy(post[0]), post[1])
            k = "transformer.h.0.mlp.c_fc.weight"
            post[0][k] = post[0][k].clone(); post[0][k].view(-1)[0] = torch.nextafter(post[0][k].view(-1)[0], torch.tensor(1e30))
        ok, bad, tv = verify(pre, recs, post, batch, cr)
        out[label] = dict(accepted=ok, first_failures=bad[:3], n_failures=len(bad))
        print(f"[{'certified' if cr else 'float'}] {label:44s} accepted={ok}  {bad[:2]}", flush=True)

    run("honest")

    # tamper the AdamW update (non-matmul): add 1e-6 to one parameter entry after the step
    def adam_tamper(model):
        p = model.transformer.h[0].mlp.c_fc.weight
        return [opt.register_step_post_hook(lambda o, a, k: p.data.view(-1)[5].add_(1e-6))]
    run("AdamW output +1e-6 in one weight", op_tamper=adam_tamper)

    # tamper a LayerNorm forward output (non-matmul) by 1 ulp in one entry
    def ln_tamper(model):
        def hook(mod, inp, outp):
            d = torch.zeros_like(outp)
            x = outp.detach().view(-1)[3]
            d.view(-1)[3] = torch.nextafter(x, torch.tensor(1e30)) - x     # +1 ulp, out of place
            return outp + d
        return [model.transformer.h[1].ln_2.register_forward_hook(hook)]
    run("LayerNorm output 1 ulp (one entry)", op_tamper=ln_tamper)

    # tamper the GELU backward path: scale the gradient flowing into one MLP by (1+1e-3)
    def gelu_tamper(model):
        return [model.transformer.h[2].mlp.gelu.register_full_backward_hook(lambda mod, gi, go: (gi[0] * (1 + 1e-3),))]
    run("GELU backward ×(1+1e-3)", op_tamper=gelu_tamper)

    # perturb a weight-gradient matmul inside the bf16-output float tolerance
    def subtol(A, B, C):
        M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)
        C2 = (C.double() + 0.9 * tol_coef(A.shape[-1], "worst") * M / C.shape[1]).float()   # inside the verifier's fp32 bound
        if not cr:
            return C2
        K = spec_K(A.shape[-1]); _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
        return C2, _exact_matmul_int(ai, bi) - torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
    run("dW matmul perturbed inside float tolerance", tamper3={"transformer.h.1.mlp.c_fc.dW": subtol})

    def ulp(A, B, C):
        C2 = C.clone(); C2.view(-1)[0] = torch.nextafter(C2.view(-1)[0], torch.tensor(1e30))
        if not cr:
            return C2
        K = spec_K(A.shape[-1]); _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
        return C2, _exact_matmul_int(ai, bi) - torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
    run("1-ulp forgery of a forward matmul output", tamper3={"transformer.h.0.attn.c_attn": ulp})
    run("claimed post-step weights off by 1 ulp", post_lie=True)
    return out


def cost(cr, reps=3):
    m, opt, gen = warm_state(cr, steps=5)
    batch = get_batch(DATA, 12, 64, gen)
    base = (copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()))
    t_full = []
    for _ in range(reps):
        m.load_state_dict(base[0]); opt.load_state_dict(base[1])
        t = time.perf_counter()
        with recording(m, Recorder(cr=cr, record=False)):
            training_step(m, opt, [batch], scaler(), nullcontext(), 1.0)
        t_full.append(time.perf_counter() - t)
    m.load_state_dict(base[0]); opt.load_state_dict(base[1])
    pre, recs, post = prove(m, opt, batch, cr)
    tv = [verify(pre, recs, post, batch, cr, seed=s)[2] for s in range(reps)]
    m2, opt2 = fresh(); m2.load_state_dict(base[0]); opt2.load_state_dict(base[1])
    t_plain = []
    for _ in range(reps):
        m2.load_state_dict(base[0]); opt2.load_state_dict(base[1])
        t = time.perf_counter(); training_step(m2, opt2, [batch], scaler(), nullcontext(), 1.0); t_plain.append(time.perf_counter() - t)
    row = dict(mode="certified" if cr else "float", t_replay_same_recipe=float(np.median(t_full)),
               t_plain_fp32_step=float(np.median(t_plain)), t_oracle_verify=float(np.median(tv)))
    print(f"[cost {row['mode']}] replay of the same recipe {row['t_replay_same_recipe']*1e3:.0f} ms | plain fp32 step "
          f"{row['t_plain_fp32_step']*1e3:.0f} ms | complete oracle verification {row['t_oracle_verify']*1e3:.0f} ms", flush=True)
    return row


if __name__ == "__main__":
    RAW["attacks"] = {"float": attacks(False), "certified": attacks(True)}
    RAW["cost"] = [cost(False), cost(True)]
    json.dump(RAW, open(os.path.join(OUT, "phase12_raw.json"), "w"), indent=1)
