"""
Replay one saved training step and check the weights against the next snapshot.

train.py writes a pair of checkpoints (iter t and iter t+1). This file loads t,
reruns that single step on CPU, and compares the resulting weights to t+1.
"""

import os
import sys
from contextlib import nullcontext

import torch

from step import rebuild_trainer, training_step

def load_checkpoint(path):
    """
    Inputs:
        path: filesystem path to a pair snapshot from train.py.
    Outputs:
        the checkpoint dict (model, optimizer, batch, iter_num, ...).
    Side effects:
        reads from disk. tensors are mapped to CPU.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def enable_deterministic_replay():
    """
    Match the runtime flags train.py sets when deterministic=True.

    CPU matmuls otherwise sum across several threads, and the weights land one
    float32 ulp away from the checkpoint. torch.equal then reports a mismatch.

    Inputs:
        none.
    Outputs:
        none.
    Side effects:
        forces one PyTorch thread and deterministic algorithms for this process.
        turns TF32 off. sets CUBLAS_WORKSPACE_CONFIG so a later CUDA replay
        uses the same workspace size train.py used.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def compare_model_state(state_a, state_b):
    """
    Inputs:
        state_a, state_b: model state_dicts (name -> tensor), not full checkpoints.
    Outputs:
        True if every tensor is bitwise equal, else False.
    Side effects:
        none.
    """
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        if not torch.equal(state_a[key], state_b[key]):
            return False
    return True


def verify(path_t, path_t1):
    """
    Replay one training_step from snapshot t and compare weights to snapshot t+1.

    Uses step.training_step, not train.py (importing train.py would start a full run).

    Inputs:
        path_t: snapshot at the start of iter t (must include batch and, ideally, lr).
        path_t1: snapshot at the start of iter t+1.
    Outputs:
        True if replayed weights are bitwise equal to t+1.
    Side effects:
        prints progress; builds a fresh model on CPU.
        when the snapshot was trained with deterministic=True, switches this
        process to the same single-thread deterministic math.
    """
    ck_t = load_checkpoint(path_t)
    ck_t1 = load_checkpoint(path_t1)

    if ck_t["config"].get("deterministic"):
        enable_deterministic_replay()

    if "batch" not in ck_t:
        print("checkpoint t has no batch; re-run training after the save change")
        return False

    x, y = ck_t["batch"]
    it_t = ck_t["iter_num"]
    it_t1 = ck_t1["iter_num"]
    print(f"t={it_t} t+1={it_t1} batch X {tuple(x.shape)} Y {tuple(y.shape)}")

    if it_t1 != it_t + 1:
        print("iters are not consecutive; pass checkpoint:_t then checkpoint:_t+1")
        return False

    if ck_t["model_args"].get("dropout", 0) != 0:
        print("warning: dropout is on; replay will not match unless RNG state is saved")

    model, optimizer, cfg = rebuild_trainer(ck_t)
    dtype = cfg.get("dtype", "float32")
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    ctx = nullcontext()
    accum = cfg["gradient_accumulation_steps"]

    training_step(model, optimizer, x, y, scaler, ctx, cfg["grad_clip"], accum)

    ok = compare_model_state(model.state_dict(), ck_t1["model"])
    print("replay matches t+1" if ok else "replay does not match t+1")
    return ok


if __name__ == "__main__":
    ok = verify(sys.argv[1], sys.argv[2])
    print("pair ok" if ok else "pair not ok")
    sys.exit(0 if ok else 1)
