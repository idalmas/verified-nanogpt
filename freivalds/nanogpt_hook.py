"""
Freivalds / certificate verification for the repo's real nanoGPT (`model.py`), unmodified.

`recording(model, rec)` is a context manager that, for the duration of a training step:
  * routes every nn.Linear through RecLinear, an autograd Function that records the three GEMMs
    Y = X·Wᵀ (+b), dX = dY·W, dW = dYᵀ·X;
  * replaces CausalSelfAttention.forward with the explicit (non-flash) attention of model.py,
    whose QKᵀ and PV matmuls (and their backward) go through RecBMM and are recorded.
Each recorded GEMM is either a plain fp32 matmul (verify with float Freivalds) or, with
rec.cr = True, a certified correctly-rounded GEMM from freivalds/certified.py (verify exactly).
The records use the same GemmRec format as freivalds/tinygpt.py, so freivalds.stepverify works.

Not covered by GEMM checks (they still need recomputation / replay): embeddings (gather and
scatter-add backward), LayerNorm, GELU, softmax, cross-entropy, residual adds, bias sums,
gradient clipping and AdamW.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from freivalds.tinygpt import GemmRec, Trace


@dataclass
class Recorder:
    cr: bool = False
    tamper3: dict = field(default_factory=dict)   # name -> fn(A, B, C) -> C or (C, R)
    gemms: list = field(default_factory=list)
    record: bool = True

    def mm(self, name, A, B):
        R = K = None
        if self.cr:
            from freivalds.certified import cr_matmul
            C, R, meta = cr_matmul(A, B)
            K = meta["K"]
        else:
            C = A @ B
        if name in self.tamper3:
            out = self.tamper3[name](A, B, C)
            C, R = out if isinstance(out, tuple) else (out, R)
        if self.record:
            # clones: the step later mutates these tensors in place (AdamW updates W, gradient
            # clipping rescales p.grad, which autograd may alias to the dW output)
            self.gemms.append(GemmRec(name, A.detach().clone(), B.detach().clone(), C.detach().clone(), R, K))
        return C

    def as_trace(self):
        return Trace(gemms=self.gemms, ops=[])


class RecLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, name, rec):
        X2 = x.reshape(-1, x.shape[-1])
        Y = rec.mm(name, X2, W.t())
        if b is not None:
            Y = Y + b
        ctx.save_for_backward(X2, W)
        ctx.has_b, ctx.name, ctx.rec, ctx.xshape = b is not None, name, rec, x.shape
        return Y.view(*x.shape[:-1], W.shape[0])

    @staticmethod
    def backward(ctx, dY):
        X2, W = ctx.saved_tensors
        dY2 = dY.reshape(-1, dY.shape[-1])
        dX = ctx.rec.mm(ctx.name + ".dX", dY2, W)
        dW = ctx.rec.mm(ctx.name + ".dW", dY2.t(), X2)
        db = dY2.sum(0) if ctx.has_b else None
        return dX.view(ctx.xshape), dW, db, None, None


class RecBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, name, rec):
        ctx.save_for_backward(a, b)
        ctx.name, ctx.rec = name, rec
        return rec.mm(name, a, b)

    @staticmethod
    def backward(ctx, dC):
        a, b = ctx.saved_tensors
        da = ctx.rec.mm(ctx.name + ".dA", dC, b.transpose(-1, -2))
        db = ctx.rec.mm(ctx.name + ".dB", a.transpose(-1, -2), dC)
        return da, db, None, None


def _attn_forward(self, x, name, rec):
    """model.py CausalSelfAttention.forward, explicit-attention branch, with recorded matmuls."""
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    att = RecBMM.apply(q, k.transpose(-2, -1), name + ".qk", rec) * (1.0 / math.sqrt(k.size(-1)))
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = RecBMM.apply(att, v, name + ".pv", rec)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))


@contextmanager
def recording(model, rec):
    from model import CausalSelfAttention
    saved = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            saved.append((m, m.forward))
            m.forward = (lambda x, m=m, nm=name: RecLinear.apply(x, m.weight, m.bias, nm, rec))
        elif isinstance(m, CausalSelfAttention):
            saved.append((m, m.forward))
            m.forward = (lambda x, m=m, nm=name: _attn_forward(m, x, nm, rec))
    try:
        yield rec
    finally:
        for m, f in saved:
            m.forward = f


def gemm_macs(gemms):
    tot = 0
    for g in gemms:
        b = math.prod(g.A.shape[:-2]) if g.A.dim() > 2 else 1
        tot += b * g.A.shape[-2] * g.A.shape[-1] * g.B.shape[-1]
    return tot
