import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, dim, n_experts, k):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        scores, idx = torch.topk(logits, self.k, dim=-1)
        probs = scores.softmax(dim=-1)
        return probs, idx
