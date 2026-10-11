"""
Matmul-oracle replay: complete verification of a real nanoGPT training step without doing
any matmul FLOPs.

Prover: runs `step.training_step` under `recording(model, Recorder(cr=...))`, publishing the
pre-step state (weights + optimizer), the batch, every GEMM record (A, B, C[, R]) in call order,
and the claimed post-step state.

Verifier: rebuilds the pre-step model/optimizer and replays the same training step code, but
inside `recording(model, OracleRecorder(records))`. At every matmul the verifier
  1. checks that the inputs it computed itself (A, B) are bit-identical to the committed ones —
     this verifies every non-matmul op upstream of that matmul (LayerNorm, GELU, softmax,
     embeddings, bias adds, the backward of all of these);
  2. checks the committed output C with its certificate (exact) or float Freivalds (tolerance);
  3. uses the committed C instead of multiplying.
At the end the verifier's weights and optimizer state must be bit-identical to the claimed
post-step state, which covers gradient clipping and AdamW.
Cost: the non-matmul ops (O(activations)) + O(n²) checks per matmul; no O(n³) work.
Requires the verifier to run the same op implementations as the prover (pinned stack; Phase 8).
Caveat for callers: torch's optimizer.load_state_dict keeps references to the given state
tensors, so always load deep copies of committed states.
"""
import copy

import numpy as np
import torch

from freivalds.nanogpt_hook import Recorder


def _bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))
    return bool(torch.equal(a, b))


class OracleRecorder(Recorder):
    def __init__(self, committed, cr, seed=0):
        super().__init__(cr=cr, record=False)
        self.committed = committed
        self.i = 0
        self.failures = []
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)

    def mm(self, name, A, B):
        if self.i >= len(self.committed):
            self.failures.append(f"{name}: no committed record")
            return A @ B
        g = self.committed[self.i]
        self.i += 1
        if g.name != name:
            self.failures.append(f"{name}: record order mismatch ({g.name})")
        if not _bits_equal(A.detach(), g.A) or not _bits_equal(B.detach(), g.B):
            self.failures.append(f"{name}: inputs differ from verifier's recomputation")
        if g.R is not None:
            from freivalds.certified_fast import certificate_check_fast
            A3 = g.A.reshape(-1, *g.A.shape[-2:]).numpy(); B3 = g.B.reshape(-1, *g.B.shape[-2:]).numpy()
            C3 = g.C.reshape(-1, *g.C.shape[-2:]).numpy(); R3 = g.R.reshape(-1, *g.R.shape[-2:]).numpy()
            if not all(certificate_check_fast(A3[b], B3[b], C3[b], R3[b], g.K, self.rng)[0] for b in range(A3.shape[0])):
                self.failures.append(f"{name}: certificate rejected")
        else:
            from freivalds.stepverify import check_gemm
            if not check_gemm(g, self.gen):
                self.failures.append(f"{name}: float Freivalds rejected")
        return g.C.clone()


def snapshot(model, opt):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())


def states_equal(model, opt, claimed_model_sd, claimed_opt_sd):
    bad = []
    for k, v in model.state_dict().items():
        if not _bits_equal(v, claimed_model_sd[k]):
            bad.append(f"weights:{k}")
    so = opt.state_dict()["state"]
    for pid, st in so.items():
        for k, v in st.items():
            if torch.is_tensor(v) and not _bits_equal(v, claimed_opt_sd["state"][pid][k]):
                bad.append(f"adam:{pid}:{k}")
    return bad
