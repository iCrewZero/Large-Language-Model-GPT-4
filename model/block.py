import torch
import torch.nn as nn
from .attention import Attention

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn = Attention(dim, heads)

        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, -1, C // 8).transpose(1, 2)
        k = k.view(B, T, -1, C // 8).transpose(1, 2)
        v = v.view(B, T, -1, C // 8).transpose(1, 2)

        out = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        x = x + out
        x = x + self.ff(self.ln2(x))
        return x
