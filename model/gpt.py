import torch, torch.nn as nn
from model.block import Block
from model.verifier import Verifier
from kv.paged_kv import KVState, PageAllocator

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, False)
        self.verifier = Verifier(cfg.dim)

    def forward(self, ids):
        x = self.emb(ids)
        state = KVState()
        alloc = PageAllocator(4096)

        for b in self.blocks:
            x = b(x, state, alloc, 16)

        h = self.ln(x)
        return {
            "logits": self.head(h),
            "verifier": self.verifier(h)
        }
