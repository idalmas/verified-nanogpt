import numpy as np
import pytest
import torch

from freivalds.core import int_matmul_exact
from freivalds.protocol import (MerkleTree, Transcript, build_structure, challenges_for_gemm,
                                commit, prove, prove_int_gemm, verify, verify_int_gemm,
                                verify_path, _rows)
from freivalds.tinygpt import (Config, Trace, get_batch, init_opt, init_params,
                               load_shakespeare, training_step)

CFG = Config(d=64, n_layer=1, n_head=2, block=32)
RATE = 0.05


@pytest.fixture(scope="module")
def setup():
    P = init_params(CFG, seed=3)
    batch = get_batch(load_shakespeare(), 4, 32, torch.Generator().manual_seed(1))
    return P, batch


def run(setup, tamper=None):
    P, batch = setup
    return training_step(P, init_opt(P), *batch, CFG, tr=Trace(tamper=tamper or {}))[3]


def test_merkle_roundtrip():
    rows = np.arange(40, dtype=np.float32).reshape(10, 4)
    t = MerkleTree(7, rows)
    for i in range(10):
        assert verify_path(7, i, rows[i].tobytes(), t.path(i), t.root)
    bad = rows[3].copy(); bad[0] += 1
    assert not verify_path(7, 3, bad.tobytes(), t.path(3), t.root)
    assert not verify_path(7, 4, rows[3].tobytes(), t.path(3), t.root)


def test_honest_protocol_passes(setup):
    tr = run(setup)
    S, troots, root, proof, _ = prove(tr, RATE, RATE)
    ok, fails, st = verify(S, troots, root, proof, RATE, RATE)
    assert ok, fails
    assert st["frac_bytes_seen"] < 0.5


def test_post_commit_tamper_caught(setup):
    """Prover commits honestly, then serves openings from modified data."""
    tr = run(setup)

    def post(tensors):
        out = dict(tensors)
        for k, v in out.items():
            if v.dim() == 2 and v.shape[0] > 100:
                out[k] = v + 1e-3
        return out
    S, troots, root, proof, _ = prove(tr, RATE, RATE, adversary={"post_commit": post})
    ok, fails, _ = verify(S, troots, root, proof, RATE, RATE)
    assert not ok and any("Merkle" in f for f in fails)


def _orth_tamper(setup, name, fs):
    """Choose-after-r attack: add to one row of GEMM `name` a big perturbation orthogonal to
    the r derived from the honest commitment, then recommit."""
    tr0 = run(setup)
    S0, troots0, root0, _, _ = prove(tr0, RATE, RATE, fs=fs)
    g = next(x for x in tr0.gemms if x.name == name)
    r, *_ = challenges_for_gemm(Transcript(root0, 0, fs), name, tuple(g.A.shape), tuple(g.B.shape), RATE)
    rb = torch.from_numpy(r[0, :, 0])
    u = torch.randn(rb.numel(), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    e = (u - (u @ rb) / (rb @ rb) * rb) * 0.5      # e ⟂ r exactly (in float64)

    def tam(C):
        C = C.clone(); C[3] += e.float(); return C
    tr1 = run(setup, {name: tam})
    S1, troots1, root1, proof1, _ = prove(tr1, RATE, RATE, fs=fs)
    return verify(S1, troots1, root1, proof1, RATE, RATE, fs=fs), root0 != root1


def test_choose_after_r_caught_with_fiat_shamir(setup):
    (ok, fails, _), changed = _orth_tamper(setup, "h0.fc1", fs=True)
    assert changed and not ok and "h0.fc1" in fails


def test_choose_after_r_passes_without_fiat_shamir(setup):
    """Documented attack: if r does not depend on the commitment, an orthogonal tamper of
    size 0.5 per entry (far above any rounding) passes the Freivalds check."""
    (ok, fails, _), _ = _orth_tamper(setup, "h0.fc1", fs=False)
    assert "h0.fc1" not in fails


def test_lying_sketch_caught_when_row_opened(setup):
    def tam(C):
        C = C.clone(); C[5] += 1.0; return C
    tr = run(setup, {"h0.fc2": tam})

    def lie(name, sk):
        if name == "h0.fc2":
            sk = dict(sk); sk["w"] = sk["w"].copy(); sk["w"][0, 5] = sk["z"][0, 5]
        return sk
    # honest sketch of a wrong C: caught by Freivalds even with tiny row rate
    S, tro, root, proof, _ = prove(tr, 0.001, RATE)
    assert "h0.fc2" in verify(S, tro, root, proof, 0.001, RATE)[1]
    # lying sketch with every row opened: caught
    S, tro, root, proof, _ = prove(tr, 1.0, RATE, adversary={"sketch": lie})
    assert "h0.fc2" in verify(S, tro, root, proof, 1.0, RATE)[1]


def test_int_protocol():
    rng = np.random.default_rng(0)
    A = rng.integers(-128, 128, (64, 48)); B = rng.integers(-128, 128, (48, 32))
    C = int_matmul_exact(A, B)
    shapes = (A.shape, B.shape)
    assert verify_int_gemm(shapes, *prove_int_gemm(A, B, C))
    C2 = C.copy(); C2[9, 4] += 1
    assert not verify_int_gemm(shapes, *prove_int_gemm(A, B, C2))                     # honest sketch
    assert not verify_int_gemm(shapes, *prove_int_gemm(A, B, C2, row_rate=1.0, lie_row=9), row_rate=1.0)
