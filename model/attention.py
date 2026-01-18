import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import YaRNRoPE

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head, rope_factor, use_flash=True):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        self.use_flash = use_flash

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * n_kv_head * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.rope = YaRNRoPE(self.head_dim, factor=rope_factor)

    def forward(self, x, start_pos=0):
        B,T,C = x.shape
        q = self.q(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        kv = self.kv(x).view(B,T,2,self.n_kv_head,self.head_dim)
        k,v = kv[:,:,0], kv[:,:,1]

        q,k = self.rope.apply(q, k.transpose(1,2), start_pos)
        k = k.transpose(1,2)

        k = k.repeat_interleave(self.n_head//self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head//self.n_kv_head, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).reshape(B,T,C)
        return self.o(out)
