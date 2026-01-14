import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, use_flash):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.use_flash = use_flash

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2 // n_head * n_kv_head, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, start_pos=0, kv_cache=None):
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.n_head, self.head_dim)
        kv = self.kv(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        q, k = rope(q, k, start_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        if self.use_flash and torch.cuda.is_available():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            att = att.softmax(dim=-1)
            out = att @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o(out), (k, v)
