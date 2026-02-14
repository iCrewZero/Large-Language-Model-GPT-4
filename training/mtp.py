import torch.nn as nn


class MTPHead(nn.Module):
    """Predicts multiple future token logits from current hidden state."""

    def __init__(self, dim: int, vocab: int, steps: int):
        super().__init__()
        self.steps = steps
        self.proj = nn.ModuleList([nn.Linear(dim, vocab, bias=False) for _ in range(steps)])

    def forward(self, hidden):
        return [head(hidden) for head in self.proj]
