"""Verification hooks on the repo's real model.py / step.training_step."""
from contextlib import nullcontext

import torch

from freivalds.nanogpt_hook import Recorder, recording
from freivalds.stepverify import verify_step
from freivalds.tinygpt import get_batch, load_shakespeare
from model import GPT, GPTConfig
from step import training_step

CFG = dict(block_size=32, vocab_size=65, n_layer=1, n_head=2, n_embd=32, dropout=0.0, bias=True)


def _setup():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**CFG))
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")
    batch = get_batch(load_shakespeare(), 4, 32, torch.Generator().manual_seed(0))
    return m, opt, batch


def test_hooked_model_matches_original():
    m, _, (x, y) = _setup()
    m.zero_grad(); _, l0 = m(x, y); l0.backward(); g0 = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    with recording(m, Recorder()):
        _, l1 = m(x, y); l1.backward()
    assert abs(float(l0) - float(l1)) < 1e-5
    for n, p in m.named_parameters():
        assert torch.allclose(g0[n], p.grad, atol=1e-6), n


def _run(cr, tamper=None):
    m, opt, batch = _setup()
    rec = Recorder(cr=cr, tamper3=tamper or {})
    with recording(m, rec):
        training_step(m, opt, [batch], torch.amp.GradScaler("cpu", enabled=False), nullcontext(), 1.0)
    return rec


def test_real_training_step_verifies_float_and_certified():
    for cr in [False, True]:
        rec = _run(cr)
        assert len(rec.gemms) == 3 * 5 + 6 * 1   # 5 Linears (incl. lm_head) × 3 GEMMs + 2 attention matmuls × 3
        assert verify_step(rec.as_trace(), rho=0.0)["ok"]


def test_certified_catches_ulp_tamper_on_real_model():
    def ulp(A, B, C):
        C = C.clone(); C.view(-1)[0] = torch.nextafter(C.view(-1)[0], torch.tensor(1e30)); return C
    rec = _run(True, {"transformer.h.0.mlp.c_fc.dW": ulp})
    assert verify_step(rec.as_trace(), rho=0.0)["failing"] == ["transformer.h.0.mlp.c_fc.dW"]
