"""
Fused numba implementation of freivalds.certified.certificate_check for 2-D fp32 GEMMs.

Same spec and same decisions as the torch version (tests compare them); one pass over B, one
over A, one over (C, R), all parallel over rows. Arithmetic mod p = 2^61 − 1 uses 64-bit limb
products (no 128-bit integers needed).
"""
import numba as nb
import numpy as np

P61 = (1 << 61) - 1
MASK29 = (1 << 29) - 1
MASK32 = (1 << 32) - 1


@nb.njit(cache=True, inline="always")
def _red(x):
    """x mod p for x < 2^64 (uint64)."""
    x = (x & np.uint64(P61)) + (x >> np.uint64(61))
    x = (x & np.uint64(P61)) + (x >> np.uint64(61))
    if x >= np.uint64(P61):
        x -= np.uint64(P61)
    return x


@nb.njit(cache=True, inline="always")
def _mulmod(a, b):
    """a·b mod p for a, b < p (uint64), via 32-bit halves: 2^64 ≡ 8, 2^61 ≡ 1."""
    ah = a >> np.uint64(32); al = a & np.uint64(MASK32)
    bh = b >> np.uint64(32); bl = b & np.uint64(MASK32)
    hh = ah * bh                                   # < 2^58  → coefficient of 2^64 ≡ 8
    mid = ah * bl + al * bh                        # < 2^62  → coefficient of 2^32
    ll = al * bl                                   # < 2^64
    mid_hi = mid >> np.uint64(29)                  # mid·2^32 = mid_hi·2^61 + (mid & MASK29)·2^32
    mid_lo = (mid & np.uint64(MASK29)) << np.uint64(32)
    s = _red(hh * np.uint64(8)) + _red(mid_hi) + _red(mid_lo) + _red(ll)
    return _red(s)


@nb.njit(cache=True, inline="always")
def _signed_mod(x):
    """int64 → [0, p)."""
    if x >= 0:
        return _red(np.uint64(x))
    return np.uint64(P61) - _red(np.uint64(-x)) if _red(np.uint64(-x)) != 0 else np.uint64(0)


@nb.njit(cache=True, inline="always")
def _bf16(u):
    """fp32 bits (uint32 as int64) → bf16-rounded fp32 bits (RNE)."""
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) << 16


@nb.njit(cache=True, parallel=True)
def _quant_cols(Bbits, K):
    n, k = Bbits.shape
    Bi = np.empty((n, k), np.int64)
    ub = np.empty(k, np.int64)
    ok = True
    for j in nb.prange(k):
        emax = -1000
        for l in range(n):
            u = _bf16(Bbits[l, j] & 0xFFFFFFFF)
            e = (u >> 23) & 0xFF
            if (u & 0x7FFFFFFF) != 0 and e - 127 > emax:
                emax = e - 127
        if emax == -1000:
            ub[j] = 0
            for l in range(n):
                Bi[l, j] = 0
            continue
        if emax - K < -126:
            ok = False
        ub[j] = emax - K - 7
        for l in range(n):
            u = _bf16(Bbits[l, j] & 0xFFFFFFFF)
            e = ((u >> 23) & 0xFF) - 127
            if (u & 0x7FFFFFFF) == 0 or e < emax - K:
                Bi[l, j] = 0
            else:
                m = 128 + ((u >> 16) & 0x7F)
                v = m << (e - emax + K)
                Bi[l, j] = -v if (u >> 31) & 1 else v
    return Bi, ub, ok


@nb.njit(cache=True, parallel=True)
def _check(Abits, Bi, ub, Cbits, R, K, r):
    m, n = Abits.shape
    k = Bi.shape[1]
    # w = B_int r mod p
    w = np.empty(n, np.uint64)
    for l in nb.prange(n):
        s = np.uint64(0)
        for j in range(k):
            s = _red(s + _mulmod(_signed_mod(Bi[l, j]), r[j]))
        w[l] = s
    bad = np.zeros(m, np.int64)        # 0 ok, 1 range, 2 grid, 3 freivalds, 4 spec
    for i in nb.prange(m):
        # quantize row i of A and dot with w
        emax = -1000
        for l in range(n):
            u = _bf16(Abits[i, l] & 0xFFFFFFFF)
            e = (u >> 23) & 0xFF
            if (u & 0x7FFFFFFF) != 0 and e - 127 > emax:
                emax = e - 127
        ua = 0 if emax == -1000 else emax - K - 7
        if emax != -1000 and emax - K < -126:
            bad[i] = 4
            continue
        z = np.uint64(0)
        if emax != -1000:
            for l in range(n):
                u = _bf16(Abits[i, l] & 0xFFFFFFFF)
                e = ((u >> 23) & 0xFF) - 127
                if (u & 0x7FFFFFFF) != 0 and e >= emax - K:
                    mm = 128 + ((u >> 16) & 0x7F)
                    v = mm << (e - emax + K)
                    if (u >> 31) & 1:
                        v = -v
                    z = _red(z + _mulmod(_signed_mod(v), w[l]))
        # decode C row, range-check R, accumulate (C_s + R)·r
        y = np.uint64(0)
        for j in range(k):
            cb = Cbits[i, j] & 0xFFFFFFFF
            eC = (cb >> 23) & 0xFF
            frac = cb & 0x7FFFFF
            if eC == 255 or (eC == 0 and frac != 0):
                bad[i] = 4
                break
            rr = R[i, j]
            if rr >= (1 << 59) or rr <= -(1 << 59):
                bad[i] = 1
                break
            t = -126 - (ua + ub[j])
            if eC == 0:
                # C = 0 allowed only if the exact value (= R) is below 2^-126
                ar = rr if rr >= 0 else -rr
                if (t <= 0 and ar != 0) or (0 < t < 62 and ar >= (1 << t)):
                    bad[i] = 1
                    break
                y = _red(y + _mulmod(_signed_mod(rr), r[j]))
                continue
            mant = frac | (1 << 23)
            sh = eC - 150 - (ua + ub[j])
            if sh > 59 - 24:
                bad[i] = 4
                break
            if sh < 0:
                if (mant & ((1 << (-sh)) - 1)) != 0:
                    bad[i] = 2
                    break
                cs = mant >> (-sh)
                tb = 0
            else:
                cs = mant << sh
                tb = 0 if sh == 0 else (1 << sh)
            if (cb >> 31) & 1:
                cs = -cs
            if mant == (1 << 23) and tb > 0 and ((rr < 0 and cs > 0) or (rr > 0 and cs < 0)):
                tb >>= 1
            ar2 = 2 * (rr if rr >= 0 else -rr)
            if ar2 > tb or (ar2 == tb and tb > 0 and (mant & 1) == 1):
                bad[i] = 1
                break
            ad = cs + rr if cs + rr >= 0 else -(cs + rr)
            if (t > 0 and (t >= 62 or ad < (1 << t))):     # nonzero C but exact value below 2^-126
                bad[i] = 1
                break
            y = _red(y + _mulmod(_signed_mod(cs + rr), r[j]))
        if bad[i] == 0 and y != z:
            bad[i] = 3
    return bad


@nb.njit(cache=True, inline="always")
def _acc(s, x):
    """Signed int64 accumulator: add x (|x| < 2^61), reduce mod p when large."""
    s += x
    if s >= (1 << 62) or s <= -(1 << 62):
        s = s % P61
    return s


@nb.njit(cache=True, inline="always")
def _fold(hi, lo, shift):
    """(hi·2^shift + lo) mod p for signed int64 hi, lo."""
    h = _signed_mod(hi % P61)
    l = _signed_mod(lo % P61)
    return _red(_mulmod(h, np.uint64(1) << np.uint64(shift)) + l)


@nb.njit(cache=True, parallel=True)
def _check_small_r(Abits, Bi, ub, Cbits, R, K, r):
    """
    As _check, but r_j ∈ [0, 2^30): every product fits in int64 and only occasional
    reductions are needed. Soundness per repetition: 2^-30 (Schwartz–Zippel, |S| = 2^30).
    """
    m, n = Abits.shape
    k = Bi.shape[1]
    w = np.empty(n, np.uint64)
    for l in nb.prange(n):
        s = 0
        for j in range(k):
            s = _acc(s, Bi[l, j] * r[j])            # < 2^24 · 2^30
        w[l] = _signed_mod(s % P61)
    bad = np.zeros(m, np.int64)
    for i in nb.prange(m):
        emax = -1000
        for l in range(n):
            u = _bf16(Abits[i, l] & 0xFFFFFFFF)
            e = (u >> 23) & 0xFF
            if (u & 0x7FFFFFFF) != 0 and e - 127 > emax:
                emax = e - 127
        ua = 0 if emax == -1000 else emax - K - 7
        if emax != -1000 and emax - K < -126:
            bad[i] = 4
            continue
        zh = 0
        zl = 0
        if emax != -1000:
            for l in range(n):
                u = _bf16(Abits[i, l] & 0xFFFFFFFF)
                e = ((u >> 23) & 0xFF) - 127
                if (u & 0x7FFFFFFF) != 0 and e >= emax - K:
                    mm = 128 + ((u >> 16) & 0x7F)
                    v = mm << (e - emax + K)
                    if (u >> 31) & 1:
                        v = -v
                    wl = np.int64(w[l])
                    zh = _acc(zh, v * (wl >> 31))          # < 2^24 · 2^30
                    zl = _acc(zl, v * (wl & 0x7FFFFFFF))   # < 2^24 · 2^31
        z = _fold(zh, zl, 31)
        yh = 0
        yl = 0
        for j in range(k):
            cb = Cbits[i, j] & 0xFFFFFFFF
            eC = (cb >> 23) & 0xFF
            frac = cb & 0x7FFFFF
            if eC == 255 or (eC == 0 and frac != 0):
                bad[i] = 4
                break
            rr = R[i, j]
            if rr >= (1 << 59) or rr <= -(1 << 59):
                bad[i] = 1
                break
            t = -126 - (ua + ub[j])
            if eC == 0:
                # C = 0 allowed only if the exact value (= R) is below 2^-126
                ar = rr if rr >= 0 else -rr
                if (t <= 0 and ar != 0) or (0 < t < 62 and ar >= (1 << t)):
                    bad[i] = 1
                    break
                yh = _acc(yh, (rr >> 30) * r[j])
                yl = _acc(yl, (rr & 0x3FFFFFFF) * r[j])
                continue
            mant = frac | (1 << 23)
            sh = eC - 150 - (ua + ub[j])
            if sh > 59 - 24:
                bad[i] = 4
                break
            if sh < 0:
                if (mant & ((1 << (-sh)) - 1)) != 0:
                    bad[i] = 2
                    break
                cs = mant >> (-sh)
                tb = 0
            else:
                cs = mant << sh
                tb = 0 if sh == 0 else (1 << sh)
            if (cb >> 31) & 1:
                cs = -cs
            if mant == (1 << 23) and tb > 0 and ((rr < 0 and cs > 0) or (rr > 0 and cs < 0)):
                tb >>= 1
            ar2 = 2 * (rr if rr >= 0 else -rr)
            if ar2 > tb or (ar2 == tb and tb > 0 and (mant & 1) == 1):
                bad[i] = 1
                break
            ad = cs + rr if cs + rr >= 0 else -(cs + rr)
            if (t > 0 and (t >= 62 or ad < (1 << t))):     # nonzero C but exact value below 2^-126
                bad[i] = 1
                break
            d = cs + rr                                   # |d| < 2^60
            yh = _acc(yh, (d >> 30) * r[j])               # < 2^30 · 2^30
            yl = _acc(yl, (d & 0x3FFFFFFF) * r[j])
        if bad[i] == 0 and _fold(yh, yl, 30) != z:
            bad[i] = 3
    return bad


REASONS = {1: "R outside rounding interval", 2: "C not on the spec grid", 3: "Freivalds mod p failed",
           4: "outside spec (subnormal/range)"}


def certificate_check_fast(A, B, C, R, K, rng, small_r=False, reps=1):
    """
    2-D tensors/arrays: A (m,n) fp32, B (n,k) fp32, C (m,k) fp32, R (m,k) int64.
    small_r=False: r uniform in F_p (false accept ≤ 2^-61 per rep).
    small_r=True: r uniform in [0, 2^30) (false accept ≤ 2^-30 per rep), ~2-3× faster.
    """
    Ab = np.ascontiguousarray(np.asarray(A, dtype=np.float32)).view(np.int32).astype(np.int64)
    Bb = np.ascontiguousarray(np.asarray(B, dtype=np.float32)).view(np.int32).astype(np.int64)
    Cb = np.ascontiguousarray(np.asarray(C, dtype=np.float32)).view(np.int32).astype(np.int64)
    Rn = np.ascontiguousarray(np.asarray(R, dtype=np.int64))
    Bi, ub, ok = _quant_cols(Bb, K)
    if not ok:
        return False, REASONS[4]
    for _ in range(reps):
        if small_r:
            r = rng.integers(0, 1 << 30, size=B.shape[1], dtype=np.int64)
            bad = _check_small_r(Ab, Bi, ub, Cb, Rn, K, r)
        else:
            r = rng.integers(0, P61, size=B.shape[1], dtype=np.int64).astype(np.uint64)
            bad = _check(Ab, Bi, ub, Cb, Rn, K, r)
        nz = bad[bad != 0]
        if nz.size:
            return False, REASONS[int(nz[0])]
    return True, "ok"
