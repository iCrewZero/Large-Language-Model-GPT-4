import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """MLA with multi-resolution pooling and latent causal attention."""

    def __init__(self, dim: int, n_head: int, latent_dim: int, dropout: float = 0.0, multires_levels: int = 3):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.head_dim = dim // n_head
        self.latent_dim = latent_dim
        self.multires_levels = multires_levels

        in_dim = dim * multires_levels
        self.to_latent = nn.Linear(in_dim, latent_dim, bias=False)
        self.latent_q = nn.Linear(latent_dim, dim, bias=False)
        self.latent_k = nn.Linear(latent_dim, dim, bias=False)
        self.latent_v = nn.Linear(latent_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _multires_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for level in range(1, self.multires_levels):
            stride = 2**level
            pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=stride, stride=stride, ceil_mode=True)
            up = F.interpolate(pooled, size=x.size(1), mode="nearest").transpose(1, 2)
            feats.append(up)
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        x_mr = self._multires_features(x)
        latent = self.to_latent(x_mr)

        q = self.latent_q(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.latent_k(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = self.latent_v(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        latent_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        latent_out = latent_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.dropout(self.to_out(latent_out))
