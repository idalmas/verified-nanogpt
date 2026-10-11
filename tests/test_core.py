import numpy as np
import pytest

from freivalds.core import (freivalds_int, freivalds_modp, freivalds_float, int_matmul_exact,
                            matvec_mod, mersenne, U_FP32, U_BF16)


def _int8(rng, *shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int64)


@pytest.mark.parametrize("q", [7, 31, 61])
def test_matvec_mod_matches_python_ints(q):
    rng = np.random.default_rng(0)
    p = mersenne(q)
    M = rng.integers(-2**40, 2**40, size=(7, 33), dtype=np.int64)
    v = rng.integers(0, p, size=33, dtype=np.int64)
    got = matvec_mod(M, v, q)
    want = [sum(int(M[i, j]) * int(v[j]) for j in range(33)) % p for i in range(7)]
    assert got.tolist() == want


def test_int_honest_passes_tamper_fails():
    rng = np.random.default_rng(1)
    A, B = _int8(rng, 64, 96), _int8(rng, 96, 48)
    C = int_matmul_exact(A, B)
    assert freivalds_int(A, B, C, rng=rng)
    C2 = C.copy(); C2[3, 5] += 1
    assert not freivalds_int(A, B, C2, k=3, rng=rng)


def test_modp_honest_passes_tamper_fails():
    rng = np.random.default_rng(2)
    A, B = _int8(rng, 64, 96), _int8(rng, 96, 48)
    C = int_matmul_exact(A, B)
    assert freivalds_modp(A, B, C, rng=rng)
    C2 = C.copy(); C2[10, 0] -= 1
    assert not freivalds_modp(A, B, C2, rng=rng)


def test_modp_blind_spot_multiple_of_p():
    """A tamper by exactly p is invisible mod p: why q=61 (p > any int32 error) is needed."""
    rng = np.random.default_rng(3)
    A, B = _int8(rng, 16, 16), _int8(rng, 16, 16)
    C = int_matmul_exact(A, B)
    C2 = C.copy(); C2[0, 0] += mersenne(31)
    assert freivalds_modp(A, B, C2, q=31, rng=rng)        # blind spot: accepted
    assert not freivalds_modp(A, B, C2, q=61, rng=rng)    # p=2^61-1 > |error|: caught


def test_int_false_accept_rate_matches_bound():
    """Single-entry tamper with R=4: accepted iff r_j = 0, prob exactly 1/4."""
    rng = np.random.default_rng(4)
    A, B = _int8(rng, 8, 8), _int8(rng, 8, 8)
    C = int_matmul_exact(A, B); C[2, 2] += 7
    acc = np.mean([freivalds_int(A, B, C, R=4, rng=rng) for _ in range(4000)])
    assert abs(acc - 0.25) < 0.03


def test_float_honest_never_fails_worst_case():
    rng = np.random.default_rng(5)
    for n in (16, 256, 1024):
        A = rng.standard_normal((64, n)).astype(np.float32)
        B = rng.standard_normal((n, 64)).astype(np.float32)
        C = A @ B
        assert freivalds_float(A, B, C, k=4, rng=rng)


def test_float_bf16_out_rounding():
    import torch
    rng = np.random.default_rng(6)
    A = torch.randn(64, 512).bfloat16(); B = torch.randn(512, 32).bfloat16()
    C = (A.float() @ B.float()).bfloat16().float().numpy()
    assert freivalds_float(A.float().numpy(), B.float().numpy(), C, rng=rng, out_round_u=U_BF16)
    # without accounting for output rounding, the fp32 bound is too tight
    assert not freivalds_float(A.float().numpy(), B.float().numpy(), C, rng=rng)


def test_float_large_tamper_caught_small_tamper_missed():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((64, 512)).astype(np.float32)
    B = rng.standard_normal((512, 64)).astype(np.float32)
    C = A @ B
    big = C.copy(); big[1, 1] += 1.0
    assert not freivalds_float(A, B, big, rng=rng)
    # a tamper of 1 ulp-scale is below tolerance: documented, cannot be caught in float
    small = C.copy(); small[1, 1] = np.nextafter(small[1, 1], np.float32(np.inf))
    assert freivalds_float(A, B, small, rng=rng)
