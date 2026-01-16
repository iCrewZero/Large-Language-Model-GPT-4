import torch

def quant_k(x):
    scale = x.abs().amax(dim=-1, keepdim=True) / 127 + 1e-6
    q = (x / scale).round().clamp(-128,127).to(torch.int8)
    return q, scale

def dequant_k(q, s):
    return q.float() * s
