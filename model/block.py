import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import CausalSelfAttention
from .mla import MLA
from .moe import MoE

class Block(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)

        self.attn = CausalSelfAttention(
            cfg.dim, cfg.n_head, cfg.n_kv_head, rope
        )

        self.mla = MLA(cfg.dim, cfg.n_head) if cfg.use_mla else None
        self.moe = MoE(cfg.dim, cfg.moe_experts, cfg.moe_topk) if cfg.use_moe else None

        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, 4*cfg.dim),
            nn.GELU(),
            nn.Linear(4*cfg.dim, cfg.dim)
        )

    def forward(self, x, pos):
        h = self.attn(self.norm1(x), pos)
        if self.mla:
            h = h + self.mla(h)

        x = x + h

        if self.moe:
            x = x + self.moe(self.norm2(x))
        else:
            x = x + self.ffn(self.norm2(x))

        return x
