import torch
import torch.nn as nn

class ProcessVerifier(nn.Module):
def init(self, dim):
super().init()
self.head = nn.Sequential(
nn.Linear(dim, dim),
nn.GELU(),
nn.Linear(dim, 1)
)

def forward(self, hidden_states):
    return self.head(hidden_states)
