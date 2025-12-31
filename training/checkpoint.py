import torch
import os

def save(model, opt, step, path="ckpt.pt"):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step
    }, path)

def load(model, opt, path="ckpt.pt"):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]
