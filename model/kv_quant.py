import torch

def quantize_kv(x):
    """
    x: [B, H, T, D] FP16
    returns: int8 tensor + scale
    """
    max_val = x.abs().amax(dim=(-1, -2), keepdim=True)  # per head
    scale = max_val / 127.0 + 1e-6
    x_int8 = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return x_int8, scale

def dequantize_kv(x_int8, scale):
    return x_int8.float() * scale
