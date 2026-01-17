import torch
import torch.nn as nn
from .block import Block
from .rope import YaRNRoPE

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = YaRNRoPE(cfg.dim//cfg.n_head, cfg.rope_base, cfg.rope_factor)

        self.blocks = nn.ModuleList([
            Block(cfg, self.rope) for _ in range(cfg.n_layer)
        ])

        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for i,blk in enumerate(self.blocks):
            x = blk(x, pos=0)
        x = self.norm(x)
        return self.lm_head(x)
