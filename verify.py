"""
Replay one saved training step and check the weights against the next snapshot.

train.py writes a pair of checkpoints (iter t and iter t+1). This file loads t,
reruns that single step on the device and dtype training used (or a device you
pass), and compares the resulting weights to t+1.

usage: python verify.py <checkpoint t> <checkpoint t+1> [device]
"""

import os
import sys

import torch

from step import rebuild_trainer, training_step, make_ctx

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


def verify(path_t, path_t1, device=None):
    """
    Replay one training_step from snapshot t and compare weights to snapshot t+1.

    Uses step.training_step, the same function train.py calls each iter
    (importing train.py would start a full run).

    Inputs:
        path_t: snapshot at the start of iter t (must include batches and, ideally, lr).
        path_t1: snapshot at the start of iter t+1.
        device: where to replay. None means the device training ran on, which is
            what a bitwise match needs. e.g. 'cpu' replays a GPU run on CPU.
    Outputs:
        True if replayed weights are bitwise equal to t+1.
    Side effects:
        prints progress; builds a fresh model on device.
        when the snapshot was trained with deterministic=True, switches this
        process to the same single-thread deterministic math.
    """
    ck_t = load_checkpoint(path_t)
    ck_t1 = load_checkpoint(path_t1)

    cfg = ck_t["config"]
    if cfg.get("deterministic"):
        enable_deterministic_replay()
    else:
        # mirror train.py's non-deterministic defaults
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if "batches" in ck_t:
        batches = ck_t["batches"]
    elif "batch" in ck_t:
        # older snapshots saved only the first micro batch
        batches = [ck_t["batch"]]
        if cfg["gradient_accumulation_steps"] > 1:
            print("checkpoint t only has one of the micro batches; re-run training after the save change")
            return False
    else:
        print("checkpoint t has no batches; re-run training after the save change")
        return False

    device = device or cfg["device"]
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = cfg.get("dtype", "float32")
    batches = [(x.to(device), y.to(device)) for x, y in batches]

    it_t = ck_t["iter_num"]
    it_t1 = ck_t1["iter_num"]
    x, y = batches[0]
    print(f"t={it_t} t+1={it_t1} {len(batches)} micro batches of X {tuple(x.shape)} Y {tuple(y.shape)}")
    print(f"replaying on {device} with dtype {dtype} (trained on {cfg['device']})")

    if it_t1 != it_t + 1:
        print("iters are not consecutive; pass checkpoint:_t then checkpoint:_t+1")
        return False

    if ck_t["model_args"].get("dropout", 0) != 0:
        print("warning: dropout is on; replay will not match unless RNG state is saved")

    model, optimizer, cfg = rebuild_trainer(ck_t, device)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    if "scaler" in ck_t:
        scaler.load_state_dict(ck_t["scaler"])
    ctx = make_ctx(device_type, dtype)

    training_step(model, optimizer, batches, scaler, ctx, cfg["grad_clip"])

    replayed = {k: v.cpu() for k, v in model.state_dict().items()}
    ok = compare_model_state(replayed, ck_t1["model"])
    print("replay matches t+1" if ok else "replay does not match t+1")
    return ok


if __name__ == "__main__":
    device = sys.argv[3] if len(sys.argv) > 3 else None
    ok = verify(sys.argv[1], sys.argv[2], device)
    print("pair ok" if ok else "pair not ok")
    sys.exit(0 if ok else 1)
