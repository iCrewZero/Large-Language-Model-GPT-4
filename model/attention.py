import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import YaRNRoPE


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, rope_factor, rope_base=10000, use_flash=True):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.use_flash = use_flash

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * n_kv_head * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.rope = YaRNRoPE(self.head_dim, base=rope_base, factor=rope_factor)

    def forward(self, x, start_pos=0, kv_cache=None, layer_idx=None):
        bsz, seqlen, dim = x.shape
        q = self.q(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv(x).view(bsz, seqlen, 2, self.n_kv_head, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        q, k = self.rope.apply(q, k.transpose(1, 2), start_pos)
        k = k.transpose(1, 2)

        if kv_cache is not None and layer_idx is not None:
            past_k = kv_cache[layer_idx]["k"]
            past_v = kv_cache[layer_idx]["v"]
            if past_k is not None:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            kv_cache[layer_idx]["k"] = k.detach()
            kv_cache[layer_idx]["v"] = v.detach()

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.o(out)
