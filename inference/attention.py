import torch
import torch.nn as nn
import torch.nn.functional as F

class PagedAttention(nn.Module):
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, q, K, V):
        return F.scaled_dot_product_attention(q, K, V, is_causal=True)
