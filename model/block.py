import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mla import MultiHeadLatentAttention
from .moe import MoE
from .rmsnorm import RMSNorm
from .state_space import StateSpaceMixer


class AdaptiveDenseFFN(nn.Module):
    """Dense FFN with channel-wise adaptive expansion gating."""

    def __init__(self, dim: int, multiplier: int = 4):
        super().__init__()
        hidden = dim * multiplier
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.channel_gate = nn.Linear(dim, hidden)

    def forward(self, x):
        h = self.fc1(x)
        gate = torch.sigmoid(self.channel_gate(x))
        h = self.act(h) * gate
        return self.fc2(h)


class Block(nn.Module):
    """Hybrid block with parallel transformer/SSM pathways and token-adaptive depth routing."""

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
            MultiHeadLatentAttention(
                cfg.dim,
                cfg.n_head,
                cfg.mla_latent_dim,
                cfg.dropout,
                multires_levels=cfg.mla_multires_levels,
            )
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
                expert_dropout=cfg.moe_expert_dropout,
                adapter_rank=cfg.moe_adapter_rank,
            )
            if self.enable_moe
            else AdaptiveDenseFFN(cfg.dim, multiplier=cfg.adaptive_ffn_multiplier)
        )
        self.ssm = StateSpaceMixer(cfg.dim)

        self.skip_gate = nn.Linear(cfg.dim, 1)
        self.branch_gate = nn.Linear(cfg.dim, 1)

    def forward(self, x, pos, kv_cache=None, layer_idx=None):
        stats = {}
        x_norm = self.norm1(x)

        token_importance = torch.sigmoid(self.skip_gate(x_norm))  # [B, T, 1]
        skip_mask = (token_importance >= self.token_skip_threshold).to(x.dtype)

        attn_out = self.attn(x_norm, start_pos=pos, kv_cache=kv_cache, layer_idx=layer_idx)
        ssm_out = self.ssm(x_norm)

        # Gated residual mixing (token-wise) between transformer and SSM branches.
        gate = torch.sigmoid(self.branch_gate(x_norm))
        mixed = gate * attn_out + (1.0 - gate) * ssm_out

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
        stats["branch_gate_mean"] = gate.mean().detach()
        return x, stats
