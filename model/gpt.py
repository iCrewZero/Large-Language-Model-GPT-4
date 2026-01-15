import torch
import torch.nn as nn
from model.blocks import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, state, allocator):
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, state, allocator)
        return self.lm_head(x)
