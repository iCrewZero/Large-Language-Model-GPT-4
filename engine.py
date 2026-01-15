import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

def quantize_k(x):
    max_val = x.abs().amax(dim=(-1, -2), keepdim=True)
    scale = max_val / 127.0 + 1e-6
    x_i8 = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
    return x_i8, scale

def dequantize_k(x_i8, scale):
    return x_i8.float() * scale


class KVAllocator:
    def __init__(self, num_pages, page_size, num_heads, head_dim, device):
        self.page_size = page_size
        self.free_pages = list(range(num_pages))

        self.K = torch.empty(
            num_pages, page_size, num_heads, head_dim,
            dtype=torch.int8, device=device
        )
        self.V = torch.empty_like(self.K)

        self.K_scale = torch.empty(
            num_pages, 1, num_heads, 1,
            dtype=torch.float16, device=device
        )
        self.V_scale = torch.empty_like(self.K_scale)

    def alloc(self):
        assert self.free_pages, "OOM: KV pages exhausted"
        return self.free_pages.pop()

    def free(self, page_id):
        self.free_pages.append(page_id)



@dataclass
class KVState:
    pages: List[int]
    seqlen: int


class PagedAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, state: KVState, allocator: KVAllocator):
        Ks, Vs = [], []

        for pid in state.pages:
            k = dequantize_k(
                allocator.K[pid], allocator.K_scale[pid]
            )
            v = dequantize_k(
                allocator.V[pid], allocator.V_scale[pid]
            )
            Ks.append(k)
            Vs.append(v)

        K = torch.cat(Ks, dim=0).transpose(0, 1).unsqueeze(0)
        V = torch.cat(Vs, dim=0).transpose(0, 1).unsqueeze(0)

        return F.scaled_dot_product_attention(
            q, K, V, is_causal=True
        )


class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = PagedAttention(num_heads, self.head_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, state, allocator):
        B, T, C = x.shape
        qkv = self.qkv(x[:, -1:])
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        if state.seqlen % allocator.page_size == 0:
            pid = allocator.alloc()
            state.pages.append(pid)

        page_id = state.pages[-1]
        offset = state.seqlen % allocator.page_size

        k = k.view(self.num_heads, self.head_dim)
        v = v.view(self.num_heads, self.head_dim)

        k_q, ks = quantize_k(k)
        v_q, vs = quantize_k(v)

        allocator.K[page_id, offset] = k_q
        allocator.V[page_id, offset] = v_q
        allocator.K_scale[page_id] = ks
        allocator.V_scale[page_id] = vs

        state.seqlen += 1

        out = self.attn(q, state, allocator)
        out = out.transpose(1, 2).reshape(B, 1, C)
        return x[:, -1:] + self.proj(out)


class GPT(nn.Module):
    def __init__(self, vocab, d_model, layers, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, heads) for _ in range(layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, token, state, allocator):
        x = self.embed(token)
        for blk in self.blocks:
            x = blk(x, state, allocator)
        return self.lm_head(x)


@torch.no_grad()
def speculative_decode(
    target: GPT,
    draft: GPT,
    start_token,
    allocator,
    steps=32,
    speculate_k=4
):
    device = start_token.device
    t_state = KVState([], 0)
    d_state = KVState([], 0)

    token = start_token
    out = [token.item()]

    for _ in range(steps):
        draft_tokens = []
        for _ in range(speculate_k):
            logits = draft(token.unsqueeze(0), d_state, allocator)
            token = logits.argmax(-1)
            draft_tokens.append(token)
          
        accept = 0
        for t in draft_tokens:
            logits = target(t.unsqueeze(0), t_state, allocator)
            pred = logits.argmax(-1)
            if pred.item() == t.item():
                out.append(t.item())
                accept += 1
            else:
                out.append(pred.item())
                token = pred
                break

        if accept == len(draft_tokens):
            token = draft_tokens[-1]

    return out
