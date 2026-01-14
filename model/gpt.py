import torch.nn as nn
from .blocks import Block
from .rmsnorm import RMSNorm
from .rope import RoPE

class GPT(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim // heads)

        self.blocks = nn.ModuleList(
            [Block(dim, heads) for _ in range(layers)]
        )

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, start_pos=0):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.rope, start_pos)
        return self.lm_head(self.norm(x))
