import torch
import torch.nn as nn


class StateSpaceMixer(nn.Module):
    """Lightweight diagonal state-space style token mixer in residual form."""

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.in_proj(x)
        state = torch.zeros_like(h[:, :1])
        ys = []
        a = torch.sigmoid(self.a).view(1, 1, -1)
        b = self.b.view(1, 1, -1)
        for t in range(h.size(1)):
            state = a * state + b * h[:, t : t + 1]
            ys.append(state)
        y = torch.cat(ys, dim=1)
        return self.out_proj(y)
