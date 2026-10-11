import numpy as np
import pytest
import torch

from freivalds.core import int_matmul_exact
from freivalds.stepverify import (check_gemms_combined, modp_combined_check, modp_separate_check,
                                  verify_step)
from freivalds.tinygpt import (Config, get_batch, init_opt, init_params, load_shakespeare,
                               reference_grads, training_step)

CFG = Config(d=64, n_layer=2, n_head=2, block=32)


@pytest.fixture(scope="module")
def batch():
    data = load_shakespeare()
    return get_batch(data, 4, 32, torch.Generator().manual_seed(0))


def _step(batch, tamper=None):
    from freivalds.tinygpt import Trace
    P = init_params(CFG, seed=1)
    return training_step(P, init_opt(P), *batch, CFG, tr=Trace(tamper=tamper or {}))


def test_handwritten_backward_matches_autograd(batch):
    P = init_params(CFG, seed=1)
    _, _, loss, tr = training_step(P, init_opt(P), *batch, CFG)
    G = {o.name[5:]: o.inputs["g"] for o in tr.ops if o.name.startswith("adam.")}
    R, rl = reference_grads(P, *batch, CFG)
    assert abs(loss - rl) < 1e-5
    for k in G:
        assert torch.allclose(G[k], R[k], atol=1e-6, rtol=1e-4), k


def test_honest_step_passes(batch):
    *_, tr = _step(batch)
    assert verify_step(tr, rho=1.0)["ok"]
    assert verify_step(tr, rho=0.05, combined=True)["ok"]


def _bump(C, d=1.0):
    C = C.clone(); C.view(-1)[7] += d; return C


@pytest.mark.parametrize("name", ["h0.fc1", "h1.att_s", "h0.qkv.dX", "h1.fc2.dW", "emb_tok.dW"])
def test_gemm_tamper_caught(batch, name):
    *_, tr = _step(batch, {name: _bump})
    res = verify_step(tr, rho=0.0)
    assert not res["ok"] and name in res["failing"]


@pytest.mark.parametrize("name", ["h0.gelu", "h1.softmax", "h0.ln2.bwd", "adam.h0.w1", "ce_grad", "h1.res2"])
def test_op_tamper_caught_with_full_sampling(batch, name):
    *_, tr = _step(batch, {name: _bump})
    res = verify_step(tr, rho=1.0)
    assert not res["ok"] and name in res["failing"]


def test_layout_tamper_caught(batch):
    *_, tr = _step(batch, {"h0.merge": _bump})
    assert verify_step(tr, rho=0.0)["ok"]          # layout ops are sampled: rho=0 checks nothing
    res = verify_step(tr, rho=1.0)
    assert "h0.merge" in res["failing"]


def test_float_small_gemm_tamper_is_missed(batch):
    """Documented limitation: a tamper far below the rigorous tolerance passes."""
    *_, tr = _step(batch, {"h0.fc1": lambda C: _bump(C, 1e-6)})
    assert verify_step(tr, rho=0.0)["ok"]


def test_modp_combined_honest_and_tampered():
    rng = np.random.default_rng(0)
    gem = []
    for (m, n, k) in [(8, 16, 4), (16, 8, 8), (4, 4, 32)]:
        A = rng.integers(-128, 128, (m, n)); B = rng.integers(-128, 128, (n, k))
        gem.append((A, B, int_matmul_exact(A, B)))
    assert modp_combined_check(gem, rng)
    assert modp_separate_check(gem, rng) == []
    A, B, C = gem[1]; C = C.copy(); C[3, 3] += 1; gem[1] = (A, B, C)
    assert not modp_combined_check(gem, rng)
    assert modp_separate_check(gem, rng) == [1]


def test_fp32_verifier_mode(batch):
    *_, tr = _step(batch)
    assert verify_step(tr, rho=0.1, vprec="fp32")["ok"]
    *_, tr = _step(batch, {"h0.fc2.dW": _bump})
    assert "h0.fc2.dW" in verify_step(tr, rho=0.0, vprec="fp32")["failing"]


def test_layout_cache_distinguishes_same_token_count():
    """Regression: B=16,T=32 and B=8,T=64 share input shapes for head split but differ in layout."""
    data = load_shakespeare()
    for B, T in [(16, 32), (8, 64)]:
        cfg = Config(d=64, n_layer=1, n_head=2, block=T)
        P = init_params(cfg, seed=0)
        *_, tr = training_step(P, init_opt(P), *get_batch(data, B, T, torch.Generator().manual_seed(0)), cfg)
        assert verify_step(tr, rho=1.0)["ok"]


def test_int_dw_step_exact_check(batch):
    from freivalds.tinygpt import Trace
    cfg = Config(d=64, n_layer=2, n_head=2, block=32, int_dw=True)
    P = init_params(cfg, seed=1)
    *_, tr = training_step(P, init_opt(P), *batch, cfg)
    assert len(tr.int_gemms) == 2 + 4 * cfg.n_layer + 1
    assert verify_step(tr, rho=0.1)["ok"]

    def plus1(A, B, C):
        C = C.clone(); C[0, 0] += 1; return C
    *_, tr = training_step(P, init_opt(P), *batch, cfg, tr=Trace(tamper3={"h1.fc1.dW": plus1}))
    assert verify_step(tr, rho=0.0)["failing"] == ["h1.fc1.dW"]


def test_tolerance_modes_ordering():
    from freivalds.stepverify import tol_coef
    n = 1024
    assert tol_coef(n, "prob") < tol_coef(n, "worst") < tol_coef(n, "bf16out")


def test_underflow_gemm_honest_passes():
    """Regression (Phase 4): an honest attᵀ·dY GEMM with 943 subnormal attention entries,
    which MKL flushes to zero, failed the pure γ_n bound by 2.6e5×. The underflow term fixes it."""
    import os
    from freivalds.tinygpt import GemmRec
    from freivalds.stepverify import check_gemm
    import numpy as np
    raw = np.load(os.path.join(os.path.dirname(__file__), "data_underflow_gemm.npz"))
    d = {k: torch.from_numpy(raw[k]) for k in ("A", "B", "C")}
    g = GemmRec("att_y.dB", d["A"], d["B"], d["C"])
    gen = torch.Generator().manual_seed(0)
    assert all(check_gemm(g, gen) for _ in range(20))
    assert all(check_gemm(g, gen, vprec="fp32") for _ in range(20))
    C = d["C"].clone(); C.view(-1)[5] += 1e-3
    assert not check_gemm(GemmRec("t", d["A"], d["B"], C), gen)
