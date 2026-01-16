import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, q, k, v, mask=None):
        att = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        return att @ v
