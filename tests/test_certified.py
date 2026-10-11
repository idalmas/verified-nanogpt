from fractions import Fraction

import numpy as np
import pytest
import torch

from freivalds.certified import _exact_matmul_int, certificate_check, cr_matmul, quantize
from freivalds.certified_fast import certificate_check_fast
from freivalds.stepverify import verify_step
from freivalds.tinygpt import Config, Trace, get_batch, init_opt, init_params, load_shakespeare, training_step


def _f32_rne(x):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from phase7_certified import f32_rne
    return f32_rne(x)


def test_cr_matmul_is_exact_product_rounded_once():
    torch.manual_seed(0)
    A = torch.randn(3, 25) * torch.exp(torch.randn(3, 1) * 4); B = torch.randn(25, 3)
    C, R, meta = cr_matmul(A, B)
    Aq, _, _ = quantize(A, -1, meta["K"]); Bq, _, _ = quantize(B, -2, meta["K"])
    for i in range(3):
        for j in range(3):
            ex = sum(Fraction(float(Aq[i, l])) * Fraction(float(Bq[l, j])) for l in range(25))
            assert _f32_rne(ex) == C[i, j].item()


@pytest.mark.parametrize("fast", [False, True])
def test_certificate_honest_and_forgeries(fast):
    rng = np.random.default_rng(0)
    A = torch.randn(32, 200); B = torch.randn(200, 24)
    C, R, meta = cr_matmul(A, B); K = meta["K"]
    chk = (lambda C_, R_: certificate_check_fast(A.numpy(), B.numpy(), C_.numpy(), R_.numpy(), K, rng)[0]) if fast \
        else (lambda C_, R_: certificate_check(A, B, C_, R_, K, rng)[0])
    assert chk(C, R)
    _, ai, ua = quantize(A, -1, K); _, bi, ub = quantize(B, -2, K)
    E = _exact_matmul_int(ai, bi)
    for idx in [0, 77, 500]:
        for direction in [1e30, -1e30]:
            C2 = C.clone(); C2.view(-1)[idx] = torch.nextafter(C2.view(-1)[idx], torch.tensor(direction))
            assert not chk(C2, R)                                   # honest R, 1-ulp change
            Cs2 = torch.ldexp(C2.double(), (-(ua + ub)).int()).round().long()
            assert not chk(C2, E - Cs2)                             # forged matching residual
    R2 = R.clone(); R2.view(-1)[3] += 1
    assert not chk(C, R2)


def test_certified_training_step_verifies_and_rejects_dw_tamper():
    cfg = Config(d=64, n_layer=1, n_head=2, block=32, cr=True)
    P = init_params(cfg, seed=0)
    batch = get_batch(load_shakespeare(), 4, 32, torch.Generator().manual_seed(0))
    *_, tr = training_step(P, init_opt(P), *batch, cfg)
    assert verify_step(tr, rho=0.1, exact_ops=True)["ok"]

    def ulp(A, B, C):
        C = C.clone(); C.view(-1)[0] = torch.nextafter(C.view(-1)[0], torch.tensor(1e30)); return C
    *_, tr = training_step(P, init_opt(P), *batch, cfg, tr=Trace(tamper3={"h0.fc1.dW": ulp}))
    assert verify_step(tr, rho=0.0, exact_ops=True)["failing"] == ["h0.fc1.dW"]
