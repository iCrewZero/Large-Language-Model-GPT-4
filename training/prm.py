import torch.nn as nn

class ProcessRewardModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, hidden):
        return self.net(hidden).squeeze(-1).mean(dim=-1)
