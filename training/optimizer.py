import torch


def build_optimizer(
    model,
    lr=3e-4,
    wd=0.1,
    lr_embed_mult: float = 0.5,
    lr_head_mult: float = 1.0,
    lr_moe_mult: float = 1.2,
):
    embed_params, head_params, moe_params, base_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name:
            embed_params.append(p)
        elif "lm_head" in name:
            head_params.append(p)
        elif "moe" in name or "experts" in name:
            moe_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": lr, "weight_decay": wd},
        {"params": embed_params, "lr": lr * lr_embed_mult, "weight_decay": wd},
        {"params": head_params, "lr": lr * lr_head_mult, "weight_decay": wd},
        {"params": moe_params, "lr": lr * lr_moe_mult, "weight_decay": wd},
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
