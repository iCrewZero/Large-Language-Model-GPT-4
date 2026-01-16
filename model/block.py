import torch.nn as nn
from model.attention import Attention
from model.moe import MoE

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.moe = MoE(cfg.dim, cfg.moe_experts, cfg.moe_top_k)

    def forward(self, x, *kv):
        x = x + self.attn(self.ln1(x), *kv)
        x = x + self.moe(self.ln2(x))
        return x
