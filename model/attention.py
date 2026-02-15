import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import YaRNRoPE


class CausalSelfAttention(nn.Module):
    """Hierarchical attention with local window + global tokens + head specialization routing."""

    def __init__(
        self,
        dim,
        n_head,
        n_kv_head,
        rope_factor,
        rope_base=10000,
        use_flash=True,
        local_window: int = 256,
        global_stride: int = 64,
    ):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.use_flash = use_flash
        self.local_window = local_window
        self.global_stride = global_stride

        # MQA/GQA hybrid: q has n_head, k/v have n_kv_head and are broadcast to groups.
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * n_kv_head * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        # Head specialization routing: token-wise head scaling in [0,1].
        self.head_router = nn.Linear(dim, n_head, bias=True)

        self.rope = YaRNRoPE(self.head_dim, base=rope_base, factor=rope_factor)

    def _local_causal_mask(self, seq_q: int, seq_k: int, device):
        q_idx = torch.arange(seq_q, device=device).unsqueeze(-1)
        k_idx = torch.arange(seq_k, device=device).unsqueeze(0)
        causal = k_idx <= q_idx + (seq_k - seq_q)
        if self.local_window > 0:
            lower = q_idx + (seq_k - seq_q) - self.local_window
            causal = causal & (k_idx >= lower)
        return causal

    def _apply_hierarchical(self, q, k, v):
        tk = k.size(2)
        tq = q.size(2)

        local_mask = self._local_causal_mask(tq, tk, q.device)
        attn_local = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask)

        if self.global_stride <= 0 or tk < self.global_stride:
            return attn_local

        g_idx = torch.arange(0, tk, self.global_stride, device=q.device)
        k_global = k.index_select(2, g_idx)
        v_global = v.index_select(2, g_idx)
        attn_global = F.scaled_dot_product_attention(q, k_global, v_global, is_causal=False)
        return 0.7 * attn_local + 0.3 * attn_global

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

        # head specialization routing (per token/head weights)
        head_weight = torch.sigmoid(self.head_router(x)).transpose(1, 2).unsqueeze(-1)  # [B,H,T,1]
        q = q * head_weight

        out = self._apply_hierarchical(q, k, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.o(out)
