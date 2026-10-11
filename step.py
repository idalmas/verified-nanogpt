import torch

from model import GPT, GPTConfig


def rebuild_trainer(ck):
    """
    Turn a pair-snapshot dict back into a live GPT + AdamW on CPU.

    Inputs:
        ck: checkpoint from train.py (model, optimizer, model_args, config, lr).
    Outputs:
        model, optimizer, config — ready for training_step. model is in train() mode.
    Side effects:
        constructs new modules; prints GPT param counts from configure_optimizers.
    """
    model = GPT(GPTConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    model.to("cpu")
    model.train()
    cfg = ck["config"]
    lr = ck.get("lr", cfg["learning_rate"])
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], lr, (cfg["beta1"], cfg["beta2"]), "cpu",
    )
    optimizer.load_state_dict(ck["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    return model, optimizer, cfg


## Basically a part of the model which does exaclty one step, for verification pruposes
def training_step(model, optimizer, X, Y, scaler, ctx, grad_clip, gradient_accumlation_steps):

    for step in range(gradient_accumlation_steps):
        with ctx:
            ## foward pass 
            _, loss = model(X, Y)
            loss = loss / gradient_accumlation_steps
            # NOT get_batch here
            ## backwards pass 
        scaler.scale(loss).backward()
        
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss 
