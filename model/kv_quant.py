import torch

def quantize_k(x):
    max_val = x.abs().amax(dim=(-1, -2), keepdim=True)
    scale = max_val / 127.0 + 1e-6
    x_i8 = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return x_i8, scale

def dequantize_k(x_i8, scale):
    return x_i8.float() * scale
