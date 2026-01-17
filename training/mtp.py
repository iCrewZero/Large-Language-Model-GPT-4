import torch
import torch.nn as nn

class MultiTokenHead(nn.Module):
    def __init__(self, dim, vocab, n_future=4):
        super().__init__()
        self.n = n_future
        self.heads = nn.ModuleList(
            [nn.Linear(dim, vocab) for _ in range(n_future)]
        )

    def forward(self, hidden):
        return [h(hidden) for h in self.heads]
