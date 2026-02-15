import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationHead(nn.Module):
    """Projection head for representation distillation."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RetrievalHead(nn.Module):
    """Vector retrieval hook. Returns similarity logits against provided memory bank."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor, memory_bank: torch.Tensor | None = None):
        # hidden [B, T, C], memory_bank [M, C]
        q = F.normalize(self.query(hidden[:, -1]), dim=-1)
        if memory_bank is None:
            return None
        k = F.normalize(memory_bank, dim=-1)
        return q @ k.t()
