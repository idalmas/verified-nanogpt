"""
A tiny char-level GPT whose full training step (forward, backward, AdamW) is written out by
hand, so that every tensor the step produces is recorded in a Trace:

  * GEMM records:  C = A @ B (optionally batched over leading dims), from forward (Y = X W),
    input-gradient (dX = dY Wᵀ) and weight-gradient (dW = Xᵀ dY) matmuls, plus attention's
    QKᵀ / PV matmuls and their backward, and the embedding lookups written as one-hot GEMMs.
  * OP records:    out = f(inputs) for non-linear / elementwise / row-wise ops
    (layer norm, softmax, GELU, cross-entropy gradient, residual adds, AdamW) and pure
    layout ops (head split/merge, which cost no FLOPs).

The model uses non-parametric layer norm (as in OLMo-1), no biases, untied embeddings and
no gradient clipping, so the backward pass stays short. Shapes are tiny (CPU).

Tampering: Trace(tamper={record_name: fn}) applies fn to that record's output before anything
downstream uses it, i.e. the adversary corrupts one value and then computes everything after
it honestly. This is the hardest case for a per-op checker (only one check can fail).
"""
import math
from dataclasses import dataclass, field

import torch

LN_EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)


# ------------------------------------------------------------------------------------------
# primitive functions (all work on row subsets / element subsets, see verifier)
# ------------------------------------------------------------------------------------------

def ln_fwd(x):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + LN_EPS)


def ln_bwd(x, dy):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + LN_EPS)
    y = (x - mu) * rstd
    return rstd * (dy - dy.mean(-1, keepdim=True) - y * (dy * y).mean(-1, keepdim=True))


def gelu_fwd(u):
    return 0.5 * u * (1.0 + torch.tanh(GELU_C * (u + 0.044715 * u ** 3)))


def gelu_bwd(u, dg):
    t = torch.tanh(GELU_C * (u + 0.044715 * u ** 3))
    dt = (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * u * u)
    return dg * (0.5 * (1 + t) + 0.5 * u * dt)


def softmax_fwd(s):
    s = s - s.max(-1, keepdim=True).values
    e = torch.exp(s)
    return e / e.sum(-1, keepdim=True)


def softmax_bwd(p, dp):
    return p * (dp - (dp * p).sum(-1, keepdim=True))


def ce_grad(logits, onehot_scaled):
    """dlogits for mean cross-entropy; onehot_scaled = onehot(target)/N (N rows)."""
    p = softmax_fwd(logits)
    return p * onehot_scaled.sum(-1, keepdim=True) - onehot_scaled


def add(a, b):
    return a + b


def scale_mask(s, maskbias, scale):
    return s * scale + maskbias


def mul_scalar(x, c):
    return x * c


# per-element FLOP model used for cost accounting of non-GEMM ops (an assumed op-cost model,
# stated in the report): elementwise ops count these many flops per output element.
OP_FLOPS_PER_ELEM = {"ln_fwd": 8, "ln_bwd": 14, "gelu_fwd": 12, "gelu_bwd": 20,
                     "softmax_fwd": 5, "softmax_bwd": 4, "ce_grad": 8, "add": 1,
                     "scale_mask": 2, "mul_scalar": 1, "adamw": 16, "layout": 0, "quant": 2,
                     "dequant": 1}


@dataclass
class GemmRec:
    name: str
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    R: torch.Tensor = None    # rounding certificate (CR-GEMM mode), see freivalds/certified.py
    K: int = None

    @property
    def mnk(self):
        batch = math.prod(self.A.shape[:-2]) if self.A.dim() > 2 else 1
        return batch, self.A.shape[-2], self.A.shape[-1], self.B.shape[-1]


@dataclass
class OpRec:
    name: str
    kind: str                 # 'elem', 'row', 'layout'
    fn_name: str
    fn: object                # callable(**inputs) -> out (or tuple of outs)
    inputs: dict
    outputs: tuple            # tuple of tensors


@dataclass
class Trace:
    tamper: dict = field(default_factory=dict)
    tamper3: dict = field(default_factory=dict)   # name -> fn(A, B, C) -> C (sees the operands)
    gemms: list = field(default_factory=list)
    int_gemms: list = field(default_factory=list)  # GemmRec with int64 A, B, C (exact)
    ops: list = field(default_factory=list)
    record: bool = True
    cr: bool = False          # every GEMM is a certified correctly-rounded GEMM

    def gemm(self, name, A, B):
        R = K = None
        if self.cr:
            from freivalds.certified import cr_matmul
            C, R, meta = cr_matmul(A, B)
            K = meta["K"]
        else:
            C = A @ B
        if name in self.tamper:
            C = self.tamper[name](C)
        if name in self.tamper3:
            out = self.tamper3[name](A, B, C)
            if isinstance(out, tuple):          # adversary may also supply its own certificate
                C, R = out
            else:
                C = out
        if self.record:
            self.gemms.append(GemmRec(name, A, B, C, R, K))
        return C

    def gemm_int8(self, name, A, B):
        """
        int8 GEMM with exact int accumulation: per-tensor absmax quantization of A and B to
        int8 (recorded as exact elementwise ops), C_int = A_q B_q exactly (int64; computed in
        float64, exact since n·127² < 2^53), dequantized C = C_int · s_A s_B in fp32.
        The tamper3 hook (if any) receives (A_q, B_q, C_int) and returns a new C_int.
        """
        sa = (A.abs().max() / 127).clamp_min(1e-30)
        sb = (B.abs().max() / 127).clamp_min(1e-30)
        qa = self.op(name + ".qA", "elem", "quant", lambda x, s=sa: torch.round(x / s), x=A)
        qb = self.op(name + ".qB", "elem", "quant", lambda x, s=sb: torch.round(x / s), x=B)
        Ci = (qa.double() @ qb.double()).round().long()
        if name in self.tamper3:
            Ci = self.tamper3[name](qa.long(), qb.long(), Ci)
        if self.record:
            self.int_gemms.append(GemmRec(name, qa.long(), qb.long(), Ci))
        return self.op(name + ".deq", "elem", "dequant",
                       lambda c, s=float(sa * sb): (c.double() * s).float(), c=Ci)

    def op(self, name, kind, fn_name, fn, **inputs):
        out = fn(**inputs)
        outs = out if isinstance(out, tuple) else (out,)
        if name in self.tamper:
            outs = tuple(self.tamper[name](o) if i == 0 else o for i, o in enumerate(outs))
        if self.record:
            self.ops.append(OpRec(name, kind, fn_name, fn, inputs, outs))
        return outs if isinstance(out, tuple) else outs[0]

    def tensors(self):
        """All distinct tensors in the trace by (record, slot) name, for commitments."""
        d = {}
        for g in self.gemms:
            d[f"{g.name}.A"], d[f"{g.name}.B"], d[f"{g.name}.C"] = g.A, g.B, g.C
        for o in self.ops:
            for k, v in o.inputs.items():
                if torch.is_tensor(v):
                    d[f"{o.name}.in.{k}"] = v
            for i, v in enumerate(o.outputs):
                d[f"{o.name}.out{i}"] = v
        return d


# ------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------

@dataclass
class Config:
    vocab: int = 65
    block: int = 64
    d: int = 128
    n_layer: int = 2
    n_head: int = 4
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    wd: float = 0.0
    int_dw: bool = False      # weight-gradient GEMMs in int8 with exact int accumulation
    cr: bool = False          # all GEMMs as certified correctly-rounded GEMMs (bf16 in, fp32 out)


def init_params(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    d, V = cfg.d, cfg.vocab

    def n(*s, std=0.02):
        return (torch.randn(*s, generator=g) * std).float()
    P = {"wte": n(V, d), "wpe": n(cfg.block, d), "wlm": n(d, V)}
    for l in range(cfg.n_layer):
        P[f"h{l}.wqkv"] = n(d, 3 * d)
        P[f"h{l}.wo"] = n(d, d, std=0.02 / math.sqrt(2 * cfg.n_layer))
        P[f"h{l}.w1"] = n(d, 4 * d)
        P[f"h{l}.w2"] = n(4 * d, d, std=0.02 / math.sqrt(2 * cfg.n_layer))
    return P


def init_opt(P):
    return {"t": 0, "m": {k: torch.zeros_like(v) for k, v in P.items()},
            "v": {k: torch.zeros_like(v) for k, v in P.items()}}


def adamw_fn(theta, m, v, g, lr, b1, b2, eps, wd, t):
    m2 = b1 * m + (1 - b1) * g
    v2 = b2 * v + (1 - b2) * g * g
    mhat = m2 / (1 - b1 ** t)
    vhat = v2 / (1 - b2 ** t)
    th2 = theta - lr * (mhat / (torch.sqrt(vhat) + eps) + wd * theta)
    return th2, m2, v2


def _split_heads(x, B, T, H):
    return x.view(B, T, H, -1).transpose(1, 2).contiguous()


def _merge_heads(x, B, T):
    return x.transpose(1, 2).contiguous().view(B * T, -1)


def training_step(P, opt, idx, tgt, cfg, tr=None, update_hook=None):
    """
    One full training step, recorded into tr.

    Inputs: P params dict, opt AdamW state, idx/tgt (B, T) long tensors, cfg Config,
        tr Trace (created if None), update_hook optional fn(new_params_dict) -> dict applied
        after AdamW (used by the Phase-4 adversary to add a perturbation to the update).
    Outputs: (newP, newopt, loss, tr).
    """
    tr = Trace() if tr is None else tr
    tr.cr = tr.cr or cfg.cr
    dW = tr.gemm_int8 if cfg.int_dw else tr.gemm      # weight-gradient GEMMs
    B, T = idx.shape
    d, H, V = cfg.d, cfg.n_head, cfg.vocab
    N = B * T
    dh = d // H
    scale = 1.0 / math.sqrt(dh)
    onehot = torch.nn.functional.one_hot(idx.reshape(-1), V).float()
    sel = torch.nn.functional.one_hot(torch.arange(T).repeat(B), cfg.block).float()
    maskbias = torch.full((T, T), float("-inf")).triu(1).expand(B, H, T, T).contiguous()
    maskbias = torch.where(torch.isinf(maskbias), torch.tensor(-1e9), maskbias)

    # ---------------- forward ----------------
    etok = tr.gemm("emb_tok", onehot, P["wte"])
    epos = tr.gemm("emb_pos", sel, P["wpe"])
    x = tr.op("emb_add", "elem", "add", add, a=etok, b=epos)
    cache = []
    for l in range(cfg.n_layer):
        p = f"h{l}."
        h = tr.op(p + "ln1", "row", "ln_fwd", ln_fwd, x=x)
        qkv = tr.gemm(p + "qkv", h, P[p + "wqkv"])
        q, k, v = (tr.op(p + f"split_{nm}", "layout", "layout",
                         lambda qkv, i=i: _split_heads(qkv[:, i * d:(i + 1) * d], B, T, H), qkv=qkv)
                   for i, nm in enumerate("qkv"))
        kt = tr.op(p + "kT", "layout", "layout", lambda k: k.transpose(-1, -2).contiguous(), k=k)
        s = tr.gemm(p + "att_s", q, kt)
        s2 = tr.op(p + "att_scale", "elem", "scale_mask",
                   lambda s, maskbias: scale_mask(s, maskbias, scale), s=s, maskbias=maskbias)
        att = tr.op(p + "softmax", "row", "softmax_fwd", softmax_fwd, s=s2)
        y = tr.gemm(p + "att_y", att, v)
        y2 = tr.op(p + "merge", "layout", "layout", lambda y: _merge_heads(y, B, T), y=y)
        o = tr.gemm(p + "proj", y2, P[p + "wo"])
        xm = tr.op(p + "res1", "elem", "add", add, a=x, b=o)
        h2 = tr.op(p + "ln2", "row", "ln_fwd", ln_fwd, x=xm)
        u = tr.gemm(p + "fc1", h2, P[p + "w1"])
        gg = tr.op(p + "gelu", "elem", "gelu_fwd", gelu_fwd, u=u)
        mm = tr.gemm(p + "fc2", gg, P[p + "w2"])
        xo = tr.op(p + "res2", "elem", "add", add, a=xm, b=mm)
        cache.append(dict(x=x, h=h, q=q, kt=kt, att=att, v=v, y2=y2, xm=xm, h2=h2, u=u, gg=gg))
        x = xo
    hf = tr.op("lnf", "row", "ln_fwd", ln_fwd, x=x)
    logits = tr.gemm("lm_head", hf, P["wlm"])
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(logits, tgt.reshape(-1))

    # ---------------- backward ----------------
    G = {}
    oh_t = torch.nn.functional.one_hot(tgt.reshape(-1), V).float() / N
    dlogits = tr.op("ce_grad", "row", "ce_grad", ce_grad, logits=logits, onehot_scaled=oh_t)
    hfT = tr.op("lnf_T", "layout", "layout", lambda hf: hf.t().contiguous(), hf=hf)
    G["wlm"] = dW("lm_head.dW", hfT, dlogits)
    wlmT = tr.op("wlm_T", "layout", "layout", lambda w: w.t().contiguous(), w=P["wlm"])
    dhf = tr.gemm("lm_head.dX", dlogits, wlmT)
    dx = tr.op("lnf.bwd", "row", "ln_bwd", ln_bwd, x=x, dy=dhf)
    for l in reversed(range(cfg.n_layer)):
        p, c = f"h{l}.", cache[l]
        T_ = lambda nm, t: tr.op(p + nm, "layout", "layout", lambda t: t.transpose(-1, -2).contiguous(), t=t)
        # MLP
        G[p + "w2"] = dW(p + "fc2.dW", T_("gg_T", c["gg"]), dx)
        dgg = tr.gemm(p + "fc2.dX", dx, T_("w2_T", P[p + "w2"]))
        du = tr.op(p + "gelu.bwd", "elem", "gelu_bwd", gelu_bwd, u=c["u"], dg=dgg)
        G[p + "w1"] = dW(p + "fc1.dW", T_("h2_T", c["h2"]), du)
        dh2 = tr.gemm(p + "fc1.dX", du, T_("w1_T", P[p + "w1"]))
        dxm_b = tr.op(p + "ln2.bwd", "row", "ln_bwd", ln_bwd, x=c["xm"], dy=dh2)
        dxm = tr.op(p + "res2.bwd", "elem", "add", add, a=dx, b=dxm_b)
        # attention
        G[p + "wo"] = dW(p + "proj.dW", T_("y2_T", c["y2"]), dxm)
        dy2 = tr.gemm(p + "proj.dX", dxm, T_("wo_T", P[p + "wo"]))
        dy = tr.op(p + "merge.bwd", "layout", "layout", lambda t: _split_heads(t, B, T, H), t=dy2)
        datt = tr.gemm(p + "att_y.dA", dy, T_("v_T", c["v"]))
        dv = tr.gemm(p + "att_y.dB", T_("att_T", c["att"]), dy)
        ds2 = tr.op(p + "softmax.bwd", "row", "softmax_bwd", softmax_bwd, p=c["att"], dp=datt)
        ds = tr.op(p + "att_scale.bwd", "elem", "mul_scalar",
                   lambda x: mul_scalar(x, scale), x=ds2)
        dq = tr.gemm(p + "att_s.dA", ds, T_("kT_T", c["kt"]))
        dkt = tr.gemm(p + "att_s.dB", T_("q_T", c["q"]), ds)
        dqkv = tr.op(p + "split.bwd", "layout", "layout",
                     lambda dq, dkt, dv: torch.cat([_merge_heads(dq, B, T),
                                                    _merge_heads(dkt.transpose(-1, -2), B, T),
                                                    _merge_heads(dv, B, T)], dim=1),
                     dq=dq, dkt=dkt, dv=dv)
        G[p + "wqkv"] = dW(p + "qkv.dW", T_("h_T", c["h"]), dqkv)
        dh = tr.gemm(p + "qkv.dX", dqkv, T_("wqkv_T", P[p + "wqkv"]))
        dx_b = tr.op(p + "ln1.bwd", "row", "ln_bwd", ln_bwd, x=c["x"], dy=dh)
        dx = tr.op(p + "res1.bwd", "elem", "add", add, a=dxm, b=dx_b)
    ohT = tr.op("onehot_T", "layout", "layout", lambda t: t.t().contiguous(), t=onehot)
    selT = tr.op("sel_T", "layout", "layout", lambda t: t.t().contiguous(), t=sel)
    G["wte"] = dW("emb_tok.dW", ohT, dx)
    G["wpe"] = dW("emb_pos.dW", selT, dx)

    # ---------------- AdamW ----------------
    t = opt["t"] + 1
    newP, newm, newv = {}, {}, {}
    for name in P:
        th2, m2, v2 = tr.op(f"adam.{name}", "elem", "adamw",
                            lambda theta, m, v, g, t=t: adamw_fn(theta, m, v, g, cfg.lr, cfg.beta1,
                                                                 cfg.beta2, cfg.eps, cfg.wd, t),
                            theta=P[name], m=opt["m"][name], v=opt["v"][name], g=G[name])
        newP[name], newm[name], newv[name] = th2, m2, v2
    if update_hook is not None:
        newP = update_hook(newP)
    return newP, {"t": t, "m": newm, "v": newv}, float(loss), tr


def reference_grads(P, idx, tgt, cfg):
    """Same model with torch autograd (for testing the hand-written backward)."""
    Q = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    B, T = idx.shape
    d, H = cfg.d, cfg.n_head
    x = Q["wte"][idx] + Q["wpe"][:T]
    x = x.view(B * T, d)
    for l in range(cfg.n_layer):
        p = f"h{l}."
        qkv = ln_fwd(x) @ Q[p + "wqkv"]
        q, k, v = (_split_heads(qkv[:, i * d:(i + 1) * d], B, T, H) for i in range(3))
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + _merge_heads(att, B, T) @ Q[p + "wo"]
        x = x + gelu_fwd(ln_fwd(x) @ Q[p + "w1"]) @ Q[p + "w2"]
    logits = ln_fwd(x) @ Q["wlm"]
    loss = torch.nn.functional.cross_entropy(logits, tgt.reshape(-1))
    loss.backward()
    return {k: v.grad for k, v in Q.items()}, float(loss)


def get_batch(data, B, T, gen):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + T] for i in ix])
    return x, y


def load_shakespeare():
    import os
    import numpy as np
    path = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare_char")
    tr = np.memmap(os.path.join(path, "train.bin"), dtype=np.uint16, mode="r")
    return torch.from_numpy(tr.astype(np.int64))
