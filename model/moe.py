import torch
import torch.nn as nn
from .router import Router

class MoE(nn.Module):
    def __init__(self, dim, n_experts, topk):
        super().__init__()
        self.router = Router(dim, n_experts, topk)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim),
                nn.GELU(),
                nn.Linear(4*dim, dim)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        idx, w = self.router(x)
        out = torch.zeros_like(x)

        for k in range(idx.size(-1)):
            e = idx[...,k]
            mask = torch.nn.functional.one_hot(e, len(self.experts)).float()
            for i,expert in enumerate(self.experts):
                out += expert(x) * mask[...,i:i+1] * w[...,k:k+1]

        return out
