"""
Non-interactive commit-then-challenge protocol for one recorded training step.

  1. Prover: Merkle-commit every distinct tensor of the step (one leaf per row of its 2-D view,
     row = last dimension), publish the per-tensor roots and the global root.
  2. Challenges (Fiat–Shamir): every random choice — Freivalds vectors r, rows to open, op
     coordinates to recompute — is derived from SHAKE-256(global_root ‖ step ‖ label), so the
     prover cannot pick tensors after seeing the randomness without changing the root.
  3. Prover: for each GEMM C = A·B publishes the sketches
         v = B r,  w = C r,  z = A v,  q = |B||r|,  μ = |A| q        (float64 vectors)
     and Merkle openings of a random fraction `row_rate` of the rows of A, B and C; for each
     non-GEMM op, openings of the input/output rows that contain the sampled coordinates.
  4. Verifier (never sees full tensors): checks every opening against the committed roots,
     re-derives the sketch entries for opened rows (v_j, q_j from B rows; w_i from C rows;
     z_i, μ_i from A rows), checks the Freivalds relation |w − z| ≤ τ(μ) on *all* rows, and
     recomputes sampled op coordinates from opened rows.

Soundness sketch:
  * If the prover reports honest sketches of a wrong C, Freivalds catches it as in Phase 1
    (float: only above tolerance).
  * To hide a wrong row i the prover must lie about w_i (or z_i/μ_i, v_j, q_j). A lie in one
    entry is caught only if that row is opened: probability ≈ row_rate. Sparse cheating is
    therefore caught with probability ≈ row_rate per lied entry, dense cheating ~ always.
    This is the price of a sublinear verifier with plain (non-homomorphic) commitments.
  * Fiat–Shamir: a prover that changes any tensor changes the root and therefore all
    challenges; grinding over q roots succeeds w.p. ≤ q·ε where ε is the per-challenge error.
    For sparse cheating ε ≈ 1 − row_rate is large, so grinding is cheap there.

Also: an exact variant over F_p for integer GEMMs (`prove_int_gemm` / `verify_int_gemm`).
"""
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from freivalds.core import U_FP32, U_FP64, gamma, matvec_mod

# ------------------------------------------------------------------------------------------
# Merkle trees
# ------------------------------------------------------------------------------------------

H = lambda b: hashlib.sha256(b).digest()


def _rows(t):
    """2-D float32/int view of a tensor: (rows, last_dim) as a contiguous numpy array."""
    a = t.detach().contiguous().numpy()
    if a.ndim == 0:
        a = a.reshape(1, 1)
    return a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)


def leaf_hash(tid, i, row_bytes):
    return H(b"\x00" + struct.pack("<II", tid, i) + row_bytes)


class MerkleTree:
    def __init__(self, tid, rows2d):
        self.tid = tid
        level = [leaf_hash(tid, i, rows2d[i].tobytes()) for i in range(rows2d.shape[0])]
        self.n = len(level)
        self.levels = [level]
        while len(level) > 1:
            if len(level) % 2:
                level = level + [level[-1]]
            level = [H(b"\x01" + level[j] + level[j + 1]) for j in range(0, len(level), 2)]
            self.levels.append(level)
        self.root = self.levels[-1][0]

    def path(self, i):
        out = []
        for level in self.levels[:-1]:
            sib = i ^ 1
            out.append(level[sib] if sib < len(level) else level[i])
            i //= 2
        return out


def verify_path(tid, i, row_bytes, path, root):
    h = leaf_hash(tid, i, row_bytes)
    for sib in path:
        h = H(b"\x01" + h + sib) if i % 2 == 0 else H(b"\x01" + sib + h)
        i //= 2
    return h == root


# ------------------------------------------------------------------------------------------
# Fiat–Shamir challenges
# ------------------------------------------------------------------------------------------

class Transcript:
    def __init__(self, root, step=0, fs=True):
        # fs=False is a deliberately broken variant (challenges do not depend on the
        # commitment) used to demonstrate the choose-after-r attack.
        self.seed = H(b"fs-v1" + (root if fs else b"") + struct.pack("<Q", step))

    def _xof(self, label, nbytes):
        return hashlib.shake_256(self.seed + label.encode()).digest(nbytes)

    def rademacher(self, label, shape):
        n = int(np.prod(shape))
        bits = np.unpackbits(np.frombuffer(self._xof(label, (n + 7) // 8), dtype=np.uint8))[:n]
        return (bits.astype(np.float64) * 2 - 1).reshape(shape)

    def indices(self, label, count, upper):
        raw = np.frombuffer(self._xof(label, 8 * count), dtype=np.uint64)
        return np.unique((raw % np.uint64(upper)).astype(np.int64))   # bias ≤ upper/2^64

    def field(self, label, shape, q=61):
        p = (1 << q) - 1
        n = int(np.prod(shape))
        raw = np.frombuffer(self._xof(label, 8 * n), dtype=np.uint64)
        return (raw % np.uint64(p)).astype(np.int64).reshape(shape)


# ------------------------------------------------------------------------------------------
# public structure of a step (no data)
# ------------------------------------------------------------------------------------------

@dataclass
class Structure:
    shapes: dict                     # tid -> shape
    gemms: list                      # (name, tidA, tidB, tidC)
    ops: list                        # (name, kind, fn, {input: tid}, [out tids])


def build_structure(tr):
    """Assign one tensor id per distinct tensor object in the trace (producer = consumer)."""
    ids, tensors = {}, {}

    def tid(t):
        k = id(t)
        if k not in ids:
            ids[k] = len(ids)
            tensors[ids[k]] = t
        return ids[k]
    gemms = [(g.name, tid(g.A), tid(g.B), tid(g.C)) for g in tr.gemms]
    ops = [(o.name, o.kind, o.fn, {k: tid(v) for k, v in o.inputs.items() if torch.is_tensor(v)},
            [tid(v) for v in o.outputs]) for o in tr.ops]
    shapes = {i: tuple(t.shape) for i, t in tensors.items()}
    return Structure(shapes, gemms, ops), tensors


# ------------------------------------------------------------------------------------------
# prover
# ------------------------------------------------------------------------------------------

@dataclass
class Proof:
    sketches: dict = field(default_factory=dict)   # gemm name -> dict of vectors
    openings: dict = field(default_factory=dict)   # (tid, row) -> (row_bytes, path)

    def nbytes(self):
        sk = sum(v.nbytes for d in self.sketches.values() for v in d.values())
        rows = sum(len(b) for b, _ in self.openings.values())
        paths = sum(32 * len(p) for _, p in self.openings.values())
        return dict(sketch_bytes=sk, opened_row_bytes=rows, path_bytes=paths,
                    total=sk + rows + paths + 8 * len(self.openings))


def _gemm_views(t):
    """(batch, m, n) float64 view of a GEMM operand."""
    a = t.detach().double()
    return a.reshape(-1, a.shape[-2], a.shape[-1]) if a.dim() > 2 else a.unsqueeze(0)


def commit(tensors):
    t0 = time.perf_counter()
    trees = {i: MerkleTree(i, _rows(t)) for i, t in tensors.items()}
    roots = b"".join(trees[i].root + struct.pack("<I", i) for i in sorted(trees))
    global_root = H(b"\x02" + roots)
    return trees, global_root, time.perf_counter() - t0


def challenges_for_gemm(T, name, shapeA, shapeB, row_rate):
    bt = int(np.prod(shapeA[:-2])) if len(shapeA) > 2 else 1
    m, n, k = shapeA[-2], shapeA[-1], shapeB[-1]
    r = T.rademacher(f"r/{name}", (bt, k, 1))
    rowsA = T.indices(f"rowsA/{name}", max(1, math.ceil(row_rate * bt * m)), bt * m)
    rowsB = T.indices(f"rowsB/{name}", max(1, math.ceil(row_rate * bt * n)), bt * n)
    rowsC = T.indices(f"rowsC/{name}", max(1, math.ceil(row_rate * bt * m)), bt * m)
    return r, rowsA, rowsB, rowsC


def challenges_for_op(T, name, kind, out_shape, rho):
    """
    Sampling units. Openings are whole rows, so for 'elem' and 'row' ops we sample a fraction
    rho of *rows* and recompute them entirely (a single wrong element is caught w.p. ≈ rho and
    only ≈ rho of the data is revealed). For 'layout' ops (transposes) one output row maps to
    many input rows, so we sample ceil(rho·rows) individual *elements* instead (weaker: a single
    wrong element is caught w.p. ≈ rho / row_length).
    """
    numel = int(np.prod(out_shape))
    rows = numel // out_shape[-1]
    if kind == "layout":
        return T.indices(f"op/{name}", max(1, math.ceil(rho * rows)), numel)
    return T.indices(f"op/{name}", max(1, math.ceil(rho * rows)), rows)


def prove(tr, row_rate=0.01, rho=0.01, step=0, adversary=None, fs=True):
    """
    Run the prover side on a recorded trace.
    adversary: optional dict with hooks:
        'post_commit': fn(tensors) -> tensors   (modify data after commitment, before openings)
        'sketch': fn(name, sketch_dict) -> sketch_dict (lie about sketches)
    Returns (structure, tensor_roots, global_root, proof, timings).
    """
    S, tensors = build_structure(tr)
    trees, root, t_commit = commit(tensors)
    t0 = time.perf_counter()
    T = Transcript(root, step, fs)
    if adversary and "post_commit" in adversary:
        tensors = adversary["post_commit"](tensors)
    rows2d = {i: _rows(t) for i, t in tensors.items()}
    proof = Proof()

    def open_row(tid, row):
        if (tid, row) not in proof.openings:
            proof.openings[(tid, row)] = (rows2d[tid][row].tobytes(), trees[tid].path(row))

    for name, ta, tb, tc in S.gemms:
        r, rA, rB, rC = challenges_for_gemm(T, name, S.shapes[ta], S.shapes[tb], row_rate)
        A, B, C = _gemm_views(tensors[ta]), _gemm_views(tensors[tb]), _gemm_views(tensors[tc])
        rt = torch.from_numpy(r)
        v = B @ rt
        q = B.abs() @ rt.abs()
        sk = dict(v=v.numpy()[..., 0], w=(C @ rt).numpy()[..., 0], z=(A @ v).numpy()[..., 0],
                  q=q.numpy()[..., 0], mu=(A.abs() @ q).numpy()[..., 0],
                  a1=A.abs().sum(-1).numpy())          # ‖A_i‖₁, for the underflow term
        if adversary and "sketch" in adversary:
            sk = adversary["sketch"](name, sk)
        proof.sketches[name] = sk
        for row in rA:
            open_row(ta, int(row))
        for row in rB:
            open_row(tb, int(row))
        for row in rC:
            open_row(tc, int(row))
    for name, kind, fn, ins, outs in S.ops:
        oshape = S.shapes[outs[0]]
        units = challenges_for_op(T, name, kind, oshape, rho)
        last = oshape[-1] if len(oshape) else 1
        for u in units:
            if kind == "layout":
                orow = int(u) // last
                open_row(outs[0], orow)
                tid, srow, _ = _layout_source(S, name, fn, ins, outs, int(u))
                open_row(tid, srow)
            else:
                row = int(u)
                for tid in list(ins.values()) + outs:
                    open_row(tid, row)
    t_prove = time.perf_counter() - t0
    troots = {i: trees[i].root for i in trees}
    return S, troots, root, proof, dict(t_commit=t_commit, t_respond=t_prove)


_LMAP = {}


def _layout_source(S, name, fn, ins, outs, flat_out):
    """Map an output element of a layout op to (input tid, row index) through its index map."""
    key = (name, tuple(S.shapes[t] for t in ins.values()), S.shapes[outs[0]])
    if key not in _LMAP:
        off, idx_in, offs, tids = 0, {}, [0], []
        for k, t in ins.items():
            n = int(np.prod(S.shapes[t]))
            idx_in[k] = torch.arange(off, off + n, dtype=torch.float64).view(S.shapes[t])
            off += n; offs.append(off); tids.append(t)
        _LMAP[key] = (fn(**idx_in).reshape(-1).long().numpy(), np.array(offs), tids)
    m, offs, tids = _LMAP[key]
    src = int(m[flat_out])
    j = int(np.searchsorted(offs, src, side="right") - 1)
    local = src - offs[j]
    last = S.shapes[tids[j]][-1]
    return tids[j], local // last, local % last


# ------------------------------------------------------------------------------------------
# verifier
# ------------------------------------------------------------------------------------------

def verify(S, troots, root, proof, row_rate=0.01, rho=0.01, step=0, u=U_FP32, fs=True):
    """
    Returns (ok, failures list, stats). The verifier uses only S (public structure), the
    roots, and the proof.
    """
    fails = []
    roots = b"".join(troots[i] + struct.pack("<I", i) for i in sorted(troots))
    if H(b"\x02" + roots) != root:
        return False, ["global root mismatch"], {}
    T = Transcript(root, step, fs)
    cache = {}

    def row(tid, r):
        if (tid, r) in cache:
            return cache[(tid, r)]
        if (tid, r) not in proof.openings:
            fails.append(f"missing opening t{tid}[{r}]")
            return None
        b, path = proof.openings[(tid, r)]
        if not verify_path(tid, r, b, path, troots[tid]):
            fails.append(f"bad Merkle path t{tid}[{r}]")
            return None
        cache[(tid, r)] = np.frombuffer(b, dtype=np.float32).astype(np.float64)
        return cache[(tid, r)]

    t0 = time.perf_counter()
    seen_rows = set()
    for name, ta, tb, tc in S.gemms:
        sA, sB = S.shapes[ta], S.shapes[tb]
        r, rA, rB, rC = challenges_for_gemm(T, name, sA, sB, row_rate)
        n, k = sA[-1], sB[-1]
        m = sA[-2]
        sk = proof.sketches.get(name)
        if sk is None:
            fails.append(f"{name}: missing sketch"); continue
        v, w, z, q, mu, a1 = (sk[x] for x in ("v", "w", "z", "q", "mu", "a1"))
        tol64 = lambda mag, L: gamma(L + 2, U_FP64) * 4 * mag + 1e-300
        bad = False
        for rr in rB:            # rows of B: v_j = B_j r_b, q_j = |B_j||r_b|
            b, j = divmod(int(rr), n)
            Bj = row(tb, int(rr))
            if Bj is None:
                bad = True; continue
            rb = r[b, :, 0]
            ref_q = np.abs(Bj) @ np.abs(rb)
            if abs(Bj @ rb - v[b, j]) > tol64(ref_q, k) or abs(ref_q - q[b, j]) > tol64(ref_q, k):
                bad = True
        for rr in rC:            # rows of C: w_i = C_i r_b
            b, i = divmod(int(rr), m)
            Ci = row(tc, int(rr))
            if Ci is None:
                bad = True; continue
            rb = r[b, :, 0]
            if abs(Ci @ rb - w[b, i]) > tol64(np.abs(Ci) @ np.abs(rb), k):
                bad = True
        for rr in rA:            # rows of A: z_i = A_i v_b, mu_i = |A_i| q_b
            b, i = divmod(int(rr), m)
            Ai = row(ta, int(rr))
            if Ai is None:
                bad = True; continue
            ref_mu = np.abs(Ai) @ q[b]
            ref_a1 = np.abs(Ai).sum()
            if abs(Ai @ v[b] - z[b, i]) > tol64(np.abs(Ai) @ np.abs(v[b]), n) or \
                    abs(ref_mu - mu[b, i]) > tol64(ref_mu, n) or abs(ref_a1 - a1[b, i]) > tol64(ref_a1, n):
                bad = True
        # Freivalds relation on all rows, with the rigorous tolerance built from μ
        gt = gamma(n, u)
        tau = (gt + gamma(n + k + 2, U_FP64) * (2 + gt) * 2) * mu * (1 + 1e-6) + 1e-300
        # underflow term (see stepverify._underflow_term): η(‖A_i‖₁‖r‖₁ + Σ_k q_k + 2n‖r‖₁)
        r1 = k
        tau = tau + 2.0 ** -126 * (a1 * r1 + q.sum(-1, keepdims=True) + 2 * n * r1) * 1.01
        if bad or not np.all(np.abs(w - z) <= tau):
            fails.append(name)
    t_gemm = time.perf_counter() - t0

    t0 = time.perf_counter()
    for name, kind, fn, ins, outs in S.ops:
        oshape = S.shapes[outs[0]]
        units = challenges_for_op(T, name, kind, oshape, rho)
        last = oshape[-1]
        ok = True
        for uu in units:
            uu = int(uu)
            if kind == "layout":
                o_row = row(outs[0], uu // last)
                tid, srow, scol = _layout_source(S, name, fn, ins, outs, uu)
                s_row = row(tid, srow)
                if o_row is None or s_row is None or o_row[uu % last] != s_row[scol]:
                    ok = False
                continue
            rr = uu
            in_rows = {kk: row(t, rr) for kk, t in ins.items()}
            out_rows = [row(t, rr) for t in outs]
            if any(x is None for x in list(in_rows.values()) + out_rows):
                ok = False; continue
            sub = {kk: torch.tensor(x, dtype=torch.float32).unsqueeze(0) for kk, x in in_rows.items()}
            refs = [x[None] for x in out_rows]
            new = fn(**sub)
            new = new if isinstance(new, tuple) else (new,)
            for a, b_ in zip(new, refs):
                a = a.double().numpy()
                scale = max(np.abs(b_).max(), 1e-30)
                if not np.all(np.abs(a - b_) <= 1e-5 * np.abs(b_) + 1e-6 * scale):
                    ok = False
        if not ok:
            fails.append(name)
    t_ops = time.perf_counter() - t0
    total_rows = sum(int(np.prod(s[:-1])) if len(s) > 1 else 1 for s in S.shapes.values())
    total_bytes = sum(4 * int(np.prod(s)) for s in S.shapes.values())
    opened_bytes = sum(len(b) for b, _ in proof.openings.values())
    stats = dict(t_verify_gemm=t_gemm, t_verify_ops=t_ops, opened_rows=len(proof.openings),
                 total_rows=total_rows, opened_bytes=opened_bytes, total_bytes=total_bytes,
                 frac_bytes_seen=opened_bytes / total_bytes, **proof.nbytes())
    return not fails, fails, stats


# ------------------------------------------------------------------------------------------
# exact variant: one integer GEMM over F_p, sketch + openings
# ------------------------------------------------------------------------------------------

def prove_int_gemm(A, B, C, row_rate=0.05, q=61, lie_row=None):
    """
    Exact protocol for C = A·B over the integers (int8 A, B; int64 C), mod p = 2^q − 1.
    lie_row: if set, the prover sets z[lie_row] := w[lie_row] (hiding a wrong row of C).
    """
    tensors = {0: torch.from_numpy(A.astype(np.int32)), 1: torch.from_numpy(B.astype(np.int32)),
               2: torch.from_numpy(C.astype(np.int64))}
    trees, root, _ = commit(tensors)
    T = Transcript(root)
    r = T.field("r", (B.shape[1],), q)
    v = matvec_mod(B, r, q); w = matvec_mod(C, r, q); z = matvec_mod(A, v, q)
    if lie_row is not None:
        z = z.copy(); z[lie_row] = w[lie_row]
    m, n = A.shape
    rows = {0: T.indices("rowsA", max(1, math.ceil(row_rate * m)), m),
            1: T.indices("rowsB", max(1, math.ceil(row_rate * n)), n),
            2: T.indices("rowsC", max(1, math.ceil(row_rate * m)), m)}
    R = {i: _rows(t) for i, t in tensors.items()}
    openings = {(i, int(j)): (R[i][int(j)].tobytes(), trees[i].path(int(j)))
                for i in rows for j in rows[i]}
    return {i: trees[i].root for i in trees}, root, dict(v=v, w=w, z=z), openings


def verify_int_gemm(shapes, troots, root, sk, openings, row_rate=0.05, q=61):
    p = (1 << q) - 1
    roots = b"".join(troots[i] + struct.pack("<I", i) for i in sorted(troots))
    if H(b"\x02" + roots) != root:
        return False
    T = Transcript(root)
    (m, n), (_, k) = shapes
    r = T.field("r", (k,), q)
    rows = {0: T.indices("rowsA", max(1, math.ceil(row_rate * m)), m),
            1: T.indices("rowsB", max(1, math.ceil(row_rate * n)), n),
            2: T.indices("rowsC", max(1, math.ceil(row_rate * m)), m)}
    dt = {0: np.int32, 1: np.int32, 2: np.int64}
    for i, js in rows.items():
        for j in js:
            j = int(j)
            if (i, j) not in openings:
                return False
            b, path = openings[(i, j)]
            if not verify_path(i, j, b, path, troots[i]):
                return False
            x = np.frombuffer(b, dtype=dt[i]).astype(np.int64)[None, :]
            if i == 0 and int(matvec_mod(x, sk["v"], q)[0]) != int(sk["z"][j]):
                return False
            if i == 1 and int(matvec_mod(x, r, q)[0]) != int(sk["v"][j]):
                return False
            if i == 2 and int(matvec_mod(x, r, q)[0]) != int(sk["w"][j]):
                return False
    return bool(np.array_equal(sk["w"] % p, sk["z"] % p))
