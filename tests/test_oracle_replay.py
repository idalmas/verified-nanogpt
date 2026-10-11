"""Complete verification of a real nanoGPT step by matmul-oracle replay."""
import copy
from contextlib import nullcontext

import torch

from freivalds.nanogpt_hook import Recorder, recording
from freivalds.oracle_replay import OracleRecorder, snapshot, states_equal
from freivalds.tinygpt import get_batch, load_shakespeare
from model import GPT, GPTConfig
from step import training_step

CFG = dict(block_size=32, vocab_size=65, n_layer=1, n_head=2, n_embd=32, dropout=0.0, bias=True)


def _fresh():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**CFG))
    return m, m.configure_optimizers(0.1, 1e-3, (0.9, 0.99), "cpu")


def _sc():
    return torch.amp.GradScaler("cpu", enabled=False)


def _prove_verify(cr, tamper3=None, adam_tamper=False):
    m, opt = _fresh()
    batch = get_batch(load_shakespeare(), 4, 32, torch.Generator().manual_seed(0))
    pre = snapshot(m, opt)
    rec = Recorder(cr=cr, tamper3=tamper3 or {})
    h = opt.register_step_post_hook(lambda o, a, k: m.lm_head.weight.data.view(-1)[0].add_(1e-6)) if adam_tamper else None
    with recording(m, rec):
        training_step(m, opt, [batch], _sc(), nullcontext(), 1.0)
    if h:
        h.remove()
    post = snapshot(m, opt)
    mv, ov = _fresh(); mv.load_state_dict(pre[0]); ov.load_state_dict(pre[1])
    orc = OracleRecorder(rec.gemms, cr)
    with recording(mv, orc):
        training_step(mv, ov, [batch], _sc(), nullcontext(), 1.0)
    return orc.failures + states_equal(mv, ov, post[0], post[1])


def test_honest_complete_verification():
    assert _prove_verify(False) == []
    assert _prove_verify(True) == []


def test_adam_tamper_caught():
    assert _prove_verify(True, adam_tamper=True)


def test_certified_catches_matmul_forgery_float_does_not():
    def ulp(A, B, C):
        C = C.clone(); C.view(-1)[0] = torch.nextafter(C.view(-1)[0], torch.tensor(1e30)); return C
    assert _prove_verify(False, {"transformer.h.0.attn.c_attn": ulp}) == []        # float: 1 ulp inside tolerance
    assert any("certificate" in f for f in _prove_verify(True, {"transformer.h.0.attn.c_attn": ulp}))
