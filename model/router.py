import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, dim, n_experts, topk):
        super().__init__()
        self.linear = nn.Linear(dim, n_experts, bias=False)
        self.topk = topk

    def forward(self, x):
        scores = self.linear(x)
        probs = torch.softmax(scores, dim=-1)
        topv, topi = torch.topk(probs, self.topk, dim=-1)
        return topi, topv
