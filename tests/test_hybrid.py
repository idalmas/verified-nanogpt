import torch

from freivalds.hybrid import replay_rows_check, verify_step_hybrid
from freivalds.stepverify import tol_coef, verify_step
from freivalds.tinygpt import (Config, Trace, get_batch, init_opt, init_params, load_shakespeare,
                               training_step)

CFG = Config(d=64, n_layer=1, n_head=2, block=32)


def _batch():
    return get_batch(load_shakespeare(), 8, 32, torch.Generator().manual_seed(0))


def test_honest_step_passes_hybrid():
    P = init_params(CFG, seed=0)
    *_, tr = training_step(P, init_opt(P), *_batch(), CFG)
    for s in range(3):
        assert verify_step_hybrid(tr, seed=s)["ok"]


def test_sub_tolerance_perturbation_missed_by_freivalds_caught_by_replay():
    """Every row of h0.fc1.dW perturbed by 0.5·τ_i (L1): passes Freivalds for any r,
    but bit-exact replay of any row exposes it."""
    def tam(A, B, C):
        M = (A.abs().double() @ B.abs().double()).sum(1, keepdim=True)
        E = 0.5 * tol_coef(A.shape[-1]) * M / C.shape[1]
        return (C.double() + E).float()
    P = init_params(CFG, seed=0)
    *_, tr = training_step(P, init_opt(P), *_batch(), CFG, tr=Trace(tamper3={"h0.fc1.dW": tam}))
    assert verify_step(tr, rho=0.01)["ok"]
    res = verify_step_hybrid(tr, seed=0)
    assert not res["ok"] and res["replay_failing"] == ["h0.fc1.dW"]


def test_replay_rows_batched_slices():
    A = torch.randn(4, 3, 16, 8); B = torch.randn(4, 3, 8, 16)
    from freivalds.tinygpt import GemmRec
    g = GemmRec("bmm", A, B, A @ B)
    assert replay_rows_check(g, 1.0, torch.Generator().manual_seed(0))[0]
    C = (A @ B).clone(); C[2, 1, 3, 4] += 1e-6
    assert not replay_rows_check(GemmRec("bmm", A, B, C), 1.0, torch.Generator().manual_seed(0))[0]
