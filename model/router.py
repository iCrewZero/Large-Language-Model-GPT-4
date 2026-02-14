import torch
import torch.nn as nn


class TopKRouter(nn.Module):
    """Top-k router with auxiliary-loss-free balancing via EMA expert bias updates."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        k: int,
        balance_momentum: float = 0.95,
    ):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.balance_momentum = balance_momentum

        self.register_buffer("expert_bias", torch.zeros(n_experts), persistent=True)
        self.register_buffer("ema_load", torch.zeros(n_experts), persistent=True)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x) + self.expert_bias
        topk_scores, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_prob = torch.softmax(topk_scores, dim=-1)

        with torch.no_grad():
            one_hot = torch.zeros(*topk_idx.shape[:-1], self.n_experts, device=x.device, dtype=x.dtype)
            one_hot.scatter_(-1, topk_idx, 1.0)
            load = one_hot.mean(dim=(0, 1))
            self.ema_load.mul_(self.balance_momentum).add_(load * (1.0 - self.balance_momentum))
            target = torch.full_like(self.ema_load, 1.0 / self.n_experts)
            # Positive bias for under-utilized experts, negative for over-utilized.
            self.expert_bias.add_(0.01 * (target - self.ema_load))

        return topk_prob, topk_idx
