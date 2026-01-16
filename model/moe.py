import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenRouter(nn.Module):
def init(self, dim, num_experts, k=2):
super().init()
self.num_experts = num_experts
self.k = k
self.router = nn.Linear(dim, num_experts, bias=False)

def forward(self, x):
    logits = self.router(x)
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
    return topk_scores, topk_idx


class Expert(nn.Module):
def init(self, dim, hidden_dim):
super().init()
self.net = nn.Sequential(
nn.Linear(dim, hidden_dim),
nn.GELU(),
nn.Linear(hidden_dim, dim)
)

def forward(self, x):
    return self.net(x)


class SparseMoE(nn.Module):
def init(self, dim, hidden_dim, num_experts=8, k=2):
super().init()
self.router = TokenRouter(dim, num_experts, k)
self.experts = nn.ModuleList(
[Expert(dim, hidden_dim) for _ in range(num_experts)]
)

def forward(self, x):
    B, T, D = x.shape
    scores, idx = self.router(x)

    output = torch.zeros_like(x)

    for b in range(B):
        for t in range(T):
            token_out = 0
            for i in range(idx.size(-1)):
                expert_id = idx[b, t, i].item()
                weight = scores[b, t, i]
                token_out += weight * self.experts[expert_id](x[b, t])
            output[b, t] = token_out

    return output
