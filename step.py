import torch

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
