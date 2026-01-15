import torch
import torch.nn as nn
import torch.nn.functional as F
from .paged_kv import append_tokens, write_kv_block, iter_kv_pages


class YaRNRoPE(nn.Module):
    def __init__(self, dim, base=10000, factor=8.0):
        super().__init__()
        half = dim // 2
        idx = torch.arange(half)

        low = half
        high = half * 32

        scale = torch.ones(half)
        scale[idx > high] = factor
        mask = (idx >= low) & (idx <= high)
        scale[mask] = 1 + (factor - 1) * (idx[mask] - low) / (high - low)

        inv_freq = 1.0 / (base ** (idx / half))
        self.register_buffer("inv_freq", inv_freq * scale)

    def forward(self, q, k, pos):
        T = q.size(-2)
        t = torch.arange(pos, pos + T, device=q.device)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        def rotate(x):
            x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        return q, k


def quantize_k(k):
    max_val = k.abs().amax(dim=(2, 3), keepdim=True)
    scale = max_val / 127.0 + 1e-6
    q = (k / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_k(q, scale):
    return q.float() * scale


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, allocator, use_flash=True):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.use_flash = use_flash
        self.allocator = allocator

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rope = YaRNRoPE(self.head_dim)

    def forward(self, x, state, K_pool, V_pool):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q, k = self.rope(q, k, state.seq_len)

        append_tokens(state, self.allocator, T)

        k = k.transpose(1, 2)
        write_kv_block(K_pool, V_pool, state, k, v, self.allocator)

        outputs = []
        for K_page, V_page in iter_kv_pages(
            K_pool, V_pool, state, state.seq_len, self.allocator
        ):
            K_page = K_page.repeat_interleave(
                self.n_head // self.n_kv_head, dim=1
            )
            V_page = V_page.repeat_interleave(
                self.n_head // self.n_kv_head, dim=1
            )

            out = F.scaled_dot_product_attention(
                q, K_page, V_page, is_causal=True
            )
            outputs.append(out)

        out = sum(outputs)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)
