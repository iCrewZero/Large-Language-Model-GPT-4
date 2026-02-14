import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mla import MultiHeadLatentAttention
from .moe import MoE
from .rmsnorm import RMSNorm
from .state_space import StateSpaceMixer


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
    """Hybrid block: transformer attention + state-space mixer with parallel residual pathways."""

    def __init__(self, cfg):
        super().__init__()
        self.enable_shape_checks = cfg.enable_shape_checks
        self.token_skip_threshold = cfg.token_skip_threshold

        self.norm1 = RMSNorm(cfg.dim)
        self.attn = CausalSelfAttention(
            cfg.dim,
            cfg.n_head,
            cfg.n_kv_head,
            cfg.rope_factor,
            rope_base=cfg.rope_base,
            local_window=cfg.local_window,
            global_stride=cfg.global_stride,
        )

        self.enable_mla = cfg.enable_mla
        self.enable_moe = cfg.enable_moe

        self.norm2 = RMSNorm(cfg.dim)
        self.mla = (
            MultiHeadLatentAttention(cfg.dim, cfg.n_head, cfg.mla_latent_dim, cfg.dropout)
            if self.enable_mla
            else None
        )

        self.norm3 = RMSNorm(cfg.dim)
        self.ffn = (
            MoE(
                cfg.dim,
                cfg.moe_experts,
                cfg.moe_topk,
                cfg.moe_capacity_factor,
                cfg.moe_balance_momentum,
            )
            if self.enable_moe
            else DenseMLP(cfg.dim)
        )
        self.ssm = StateSpaceMixer(cfg.dim)

        self.skip_gate = nn.Linear(cfg.dim, 1)
        self.res_mix = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x, pos, kv_cache=None, layer_idx=None):
        stats = {}
        x_norm = self.norm1(x)

        token_importance = torch.sigmoid(self.skip_gate(x_norm))  # [B, T, 1]
        skip_mask = (token_importance >= self.token_skip_threshold).to(x.dtype)

        # Parallel residual: attention and SSM branches.
        attn_out = self.attn(x_norm, start_pos=pos, kv_cache=kv_cache, layer_idx=layer_idx)
        ssm_out = self.ssm(x_norm)
        mix = torch.softmax(self.res_mix, dim=0)
        mixed = mix[0] * attn_out + mix[1] * ssm_out

        # Token-importance routing: low-importance tokens get less update.
        x = x + mixed * skip_mask

        if self.enable_mla:
            x = x + self.mla(self.norm2(x))

        ff_in = self.norm3(x)
        if self.enable_moe:
            ff_out, moe_stats = self.ffn(ff_in)
            stats.update(moe_stats)
        else:
            ff_out = self.ffn(ff_in)

        x = x + ff_out
        stats["skip_ratio"] = 1.0 - skip_mask.mean().detach()
        return x, stats
