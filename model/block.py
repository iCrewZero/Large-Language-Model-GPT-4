import torch.nn as nn

from .attention import CausalSelfAttention
from .mla import MultiHeadLatentAttention
from .moe import MoE
from .rmsnorm import RMSNorm


class DenseMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(
            cfg.dim,
            cfg.n_head,
            cfg.n_kv_head,
            cfg.rope_factor,
            rope_base=cfg.rope_base,
        )

        self.enable_mla = cfg.enable_mla
        self.enable_moe = cfg.enable_moe

        self.norm2 = RMSNorm(cfg.dim)
        if self.enable_mla:
            self.mla = MultiHeadLatentAttention(cfg.dim, cfg.n_head, cfg.mla_latent_dim, cfg.dropout)
        else:
            self.mla = None

        self.norm3 = RMSNorm(cfg.dim)
        if self.enable_moe:
            self.ffn = MoE(
                cfg.dim,
                cfg.moe_experts,
                cfg.moe_topk,
                cfg.moe_capacity_factor,
                cfg.moe_balance_momentum,
            )
        else:
            self.ffn = DenseMLP(cfg.dim)

    def forward(self, x, pos, kv_cache=None, layer_idx=None):
        stats = {}
        x = x + self.attn(self.norm1(x), start_pos=pos, kv_cache=kv_cache, layer_idx=layer_idx)

        if self.enable_mla:
            x = x + self.mla(self.norm2(x))

        ff_in = self.norm3(x)
        if self.enable_moe:
            ff_out, moe_stats = self.ffn(ff_in)
            stats.update(moe_stats)
        else:
            ff_out = self.ffn(ff_in)

        x = x + ff_out
        return x, stats
