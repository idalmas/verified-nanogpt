"""
Phase 11: verification on the repo's real nanoGPT (model.py + step.training_step), unmodified.

 1. equivalence: hooked model vs original (flash) model — loss and gradients
 2. honest steps through step.training_step (repo AdamW via configure_optimizers, grad clip 1.0):
    float mode → float Freivalds on every recorded GEMM; certified mode → exact certificates
 3. tampering a weight-gradient GEMM: Phase-4-sized sub-tolerance perturbation (bf16-output
    budget) and a 1-ulp forgery, in float and certified modes
 4. training: 300 steps fp32 vs certified, same seeds, verify every 50 steps
 5. coverage: fraction of the step's matmul work that is verified
Writes results/phase11_raw.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, GPTConfig
from step import training_step
from freivalds.certified import _exact_matmul_int, quantize, spec_K
from freivalds.nanogpt_hook import Recorder, gemm_macs, recording
from freivalds.stepverify import tol_coef, verify_step
from freivalds.tinygpt import get_batch, load_shakespeare

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
torch.set_num_threads(4)
DATA = load_shakespeare()
RAW = {}
CFG = dict(block_size=64, vocab_size=65, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True)
TRAIN = dict(lr=1e-3, wd=0.1, betas=(0.9, 0.99), clip=1.0, batch=12)


def make(seed=0):
    torch.manual_seed(seed)
    m = GPT(GPTConfig(**CFG))
    opt = m.configure_optimizers(TRAIN["wd"], TRAIN["lr"], TRAIN["betas"], "cpu")
    return m, opt


def scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def step(m, opt, batch, rec):
    from contextlib import nullcontext
    with recording(m, rec):
        loss = training_step(m, opt, [batch], scaler(), nullcontext(), TRAIN["clip"])
    return float(loss)


def exp_equivalence():
    m, _ = make()
    x, y = get_batch(DATA, 8, 64, torch.Generator().manual_seed(1))
    m.zero_grad(); _, l0 = m(x, y); l0.backward(); g0 = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(); rec = Recorder()
    with recording(m, rec):
        _, l1 = m(x, y); l1.backward()
    g1 = {n: p.grad.clone() for n, p in m.named_parameters()}
    d = max(float((g0[n] - g1[n]).abs().max()) for n in g0)
    RAW["equivalence"] = dict(loss_orig=float(l0), loss_hooked=float(l1), max_grad_diff=d, n_gemms=len(rec.gemms))
    print(f"[equiv] loss {float(l0):.6f} vs {float(l1):.6f}; max |grad diff| {d:.2e}; {len(rec.gemms)} GEMMs recorded per step")


def coverage(m, rec, tokens):
    n_params = sum(p.numel() for p in m.parameters())
    v = sum(1 for _ in [0])
    macs = gemm_macs(rec.gemms)
    # all matmul MACs of a step = recorded GEMMs; embedding lookups/scatter, norms, softmax etc. are not GEMMs
    return dict(recorded_gemm_macs=macs, n_params=n_params, tokens=tokens,
                macs_per_token_per_param=macs / (tokens * n_params))


def exp_honest_and_tamper():
    res = {}
    gen = torch.Generator().manual_seed(2)
    batch = get_batch(DATA, TRAIN["batch"], 64, gen)
    for cr in [False, True]:
        m, opt = make()
        rec = Recorder(cr=cr)
        step(m, opt, batch, rec)
        ok = verify_step(rec.as_trace(), rho=0.0)["ok"]
        cov = coverage(m, rec, TRAIN["batch"] * 64)
        res[("certified" if cr else "float") + "_honest"] = dict(ok=ok, n_gemms=len(rec.gemms), **cov)
        print(f"[honest] {'certified' if cr else 'float':9s}: {len(rec.gemms)} GEMMs, verified={ok}, "
              f"recorded GEMM MACs {cov['recorded_gemm_macs']:.3e} ({cov['macs_per_token_per_param']:.2f} per token·param)")
    target = "transformer.h.1.mlp.c_fc.dW"
    for cr in [False, True]:
        for kind in ["subtol_bf16out", "one_ulp"]:
            def tam(A, B, C, kind=kind, cr=cr):
                if kind == "subtol_bf16out":
                    M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)
                    E = 0.9 * tol_coef(A.shape[-1], "bf16out") * M / C.shape[1]
                    C2 = (C.double() + E * torch.sign(torch.randn(C.shape, generator=torch.Generator().manual_seed(0)))).float()
                else:
                    C2 = C.clone(); C2.view(-1)[0] = torch.nextafter(C2.view(-1)[0], torch.tensor(1e30))
                if not cr:
                    return C2
                K = spec_K(A.shape[-1])
                _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
                E_ = _exact_matmul_int(ai, bi)
                Cs2 = torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
                return C2, E_ - Cs2          # forged consistent certificate
            m, opt = make()
            rec = Recorder(cr=cr, tamper3={target: tam})
            step(m, opt, batch, rec)
            r = verify_step(rec.as_trace(), rho=0.0, mode="bf16out" if not cr else "worst")
            key = f"{'certified' if cr else 'float(bf16out tol)'}_{kind}"
            res[key] = dict(detected=not r["ok"], failing=r["failing"])
            print(f"[tamper] {key:32s}: detected={not r['ok']} {r['failing']}")
    RAW["honest_and_tamper"] = res


def exp_training(steps=300, every=50):
    rows = {}
    for cr in [False, True]:
        m, opt = make(seed=3)
        gen = torch.Generator().manual_seed(4)
        vgen = torch.Generator().manual_seed(5)
        val = [get_batch(DATA, 16, 64, vgen) for _ in range(4)]
        log, ver, t0 = [], [], time.time()
        for it in range(steps):
            rec = Recorder(cr=cr, record=(it % every == 0))
            loss = step(m, opt, get_batch(DATA, TRAIN["batch"], 64, gen), rec)
            if it % every == 0:
                ver.append(verify_step(rec.as_trace(), rho=0.0)["ok"])
            if it % every == 0 or it == steps - 1:
                with torch.no_grad():
                    vl = float(np.mean([float(m(xv, yv)[1]) for xv, yv in val]))
                log.append(dict(step=it + 1, train_loss=loss, val_loss=vl))
                print(f"[train {'certified' if cr else 'fp32'}] step {it+1} loss {loss:.4f} heldout {vl:.4f} "
                      f"verified so far {sum(ver)}/{len(ver)} ({time.time()-t0:.0f}s)", flush=True)
        rows["certified" if cr else "fp32"] = dict(log=log, verified=ver, seconds=time.time() - t0)
    RAW["training"] = rows


if __name__ == "__main__":
    exp_equivalence()
    exp_honest_and_tamper()
    exp_training()
    json.dump(RAW, open(os.path.join(OUT, "phase11_raw.json"), "w"), indent=1, default=str)
