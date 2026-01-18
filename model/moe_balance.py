import torch

def load_balance_loss(router_probs):
    density = router_probs.mean(dim=(0,1))
    loss = density.pow(2).mean()
    return loss
