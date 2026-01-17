import torch

def build_optimizer(model, lr=3e-4, wd=0.1):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.95)
    )
