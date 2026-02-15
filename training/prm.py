import torch
import torch.nn as nn


class ProcessRewardModel(nn.Module):
    """Token-level reward model for reasoning-chain supervision and GRPO advantages."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, hidden):
        # returns per-token rewards [B, T]
        return self.net(hidden).squeeze(-1)

    def sequence_reward(self, hidden):
        return self.forward(hidden).mean(dim=-1)
