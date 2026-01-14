import torch.nn as nn
from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        n_kv_head,
        rope,
        mlp,
        use_flash=False,
    ):
        super().__init__()

        self.attn = CausalSelfAttention(
            dim=dim,
            n_head=n_head,
            n_kv_head=n_kv_head,
            rope=rope,
            use_flash=use_flash,
        )

        self.mlp = mlp

        self.attn_norm = nn.RMSNorm(dim)
        self.mlp_norm = nn.RMSNorm(dim)

    def forward(self, x, start_pos=0):
        x = x + self.attn(
            self.attn_norm(x),
            start_pos=start_pos,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
