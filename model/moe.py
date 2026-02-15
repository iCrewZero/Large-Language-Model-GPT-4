import torch
import torch.nn as nn

from .router import TopKRouter


class ExpertMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoE(nn.Module):
    """Sparse MoE layer with top-k dispatch and auxiliary-loss-free balancing."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        topk: int,
        capacity_factor: float = 1.25,
        balance_momentum: float = 0.95,
    ):
        super().__init__()
        self.router = TopKRouter(dim, n_experts, topk, balance_momentum=balance_momentum)
        self.experts = nn.ModuleList([ExpertMLP(dim) for _ in range(n_experts)])
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor):
        probs, idx = self.router(x)
        out = torch.zeros_like(x)

        bsz, seqlen, _ = x.shape
        capacity = max(1, int((bsz * seqlen * self.capacity_factor) / len(self.experts)))

        for k_slot in range(self.router.k):
            slot_expert = idx[..., k_slot]
            slot_weight = probs[..., k_slot].unsqueeze(-1)

            for e_id, expert_net in enumerate(self.experts):
                token_mask = slot_expert == e_id
                token_positions = token_mask.nonzero(as_tuple=False)
                if token_positions.numel() == 0:
                    continue

                # Capacity cap per expert to stay memory safe during peak routing skew.
                token_positions = token_positions[:capacity]
                token_x = x[token_positions[:, 0], token_positions[:, 1]]
                token_out = expert_net(token_x)

                out[token_positions[:, 0], token_positions[:, 1]] += (
                    token_out * slot_weight[token_positions[:, 0], token_positions[:, 1]]
                )

        stats = {
            "router_probs": probs.detach(),
            "router_idx": idx.detach(),
            "expert_load": self.router.ema_load.detach().clone(),
        }
        return out, stats
