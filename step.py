from contextlib import nullcontext

import torch

from model import GPT, GPTConfig


def make_ctx(device_type, dtype):
    """
    The autocast context train.py runs forward passes under.

    Inputs:
        device_type: 'cuda' or 'cpu'.
        dtype: 'float32', 'bfloat16' or 'float16' (the config string).
    Outputs:
        a context manager. nullcontext on CPU, autocast to dtype on CUDA.
    Side effects:
        none.
    """
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    return nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def rebuild_trainer(ck, device):
    """
    Turn a pair-snapshot dict back into a live GPT + AdamW on device.

    Inputs:
        ck: checkpoint from train.py (model, optimizer, model_args, config, lr).
        device: where to rebuild, e.g. 'cuda' or 'cpu'. Use the training device
            for a bitwise replay; a different device uses different kernels.
    Outputs:
        model, optimizer, config — ready for training_step. model is in train() mode.
    Side effects:
        constructs new modules; prints GPT param counts from configure_optimizers.
    """
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.to(device)
    model.train()
    cfg = ck["config"]
    lr = ck.get("lr", cfg["learning_rate"])
    # device_type picks fused AdamW on CUDA, same as train.py
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], lr, (cfg["beta1"], cfg["beta2"]), device_type,
    )
    optimizer.load_state_dict(ck["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    return model, optimizer, cfg


## Basically a part of the model which does exaclty one step, for verification pruposes
## train.py and verify.py both call this, so the two cannot drift apart
def training_step(model, optimizer, batches, scaler, ctx, grad_clip, ddp=False):
    """
    Inputs:
        batches: list of (X, Y), one per gradient accumulation micro step.
        ddp: True when model is wrapped in DDP; grads only sync on the last micro step.
    Outputs:
        loss of the last micro step, already divided by len(batches).
    Side effects:
        updates model weights, optimizer state and scaler.
    """
    accum = len(batches)
    for micro_step, (X, Y) in enumerate(batches):
        if ddp:
            model.require_backward_grad_sync = (micro_step == accum - 1)
        with ctx:
            ## foward pass
            _, loss = model(X, Y)
            loss = loss / accum
        ## backwards pass
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss
