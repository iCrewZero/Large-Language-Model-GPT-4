import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentProjector(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.to_latent = nn.Linear(dim, latent_dim, bias=False)
        self.from_latent = nn.Linear(latent_dim, dim, bias=False)

    def forward(self, x):
        z = self.to_latent(x)
        return self.from_latent(z)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, heads, latent_dim):
        super().__init__()
        self.h = heads
        self.d = dim // heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.latent = LatentProjector(dim, latent_dim)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1,2).reshape(B, T, C)

        latent_out = self.latent(attn)
        return self.o(latent_out)
