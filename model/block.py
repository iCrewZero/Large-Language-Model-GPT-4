import torch.nn as nn
from .attention import CausalSelfAttention

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt() * self.w

class Block(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, use_flash):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, n_kv_head, use_flash)
        self.n2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, rope, start_pos=0, kv_cache=None):
        a, kv = self.attn(self.n1(x), rope, start_pos, kv_cache)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, kv
