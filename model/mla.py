import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """MLA compresses sequence features to latent tokens, performs latent attention, then projects back."""

    def __init__(self, dim: int, n_head: int, latent_dim: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.head_dim = dim // n_head
        self.latent_dim = latent_dim

        self.to_latent = nn.Linear(dim, latent_dim, bias=False)
        self.latent_q = nn.Linear(latent_dim, dim, bias=False)
        self.latent_k = nn.Linear(latent_dim, dim, bias=False)
        self.latent_v = nn.Linear(latent_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        latent = self.to_latent(x)

        q = self.latent_q(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.latent_k(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = self.latent_v(latent).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        # Latent attention is causal to preserve autoregressive semantics.
        latent_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        latent_out = latent_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.dropout(self.to_out(latent_out))
