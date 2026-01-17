import torch
import torch.nn as nn

class PRM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, hidden):
        return self.scorer(hidden).squeeze(-1)
