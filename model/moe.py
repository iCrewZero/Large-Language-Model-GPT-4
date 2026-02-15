import torch
import torch.nn as nn

from .router import TopKRouter


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, adapter_rank: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        # Low-rank adapter inside expert FFN.
        self.down = nn.Linear(dim, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.fc2(self.act(self.fc1(x)))
        adapter = self.up(self.act(self.down(x)))
        return base + adapter


class MoE(nn.Module):
    """Sparse MoE with top-k dispatch, shared expert, capacity cap, and expert dropout."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        topk: int,
        capacity_factor: float = 1.25,
        balance_momentum: float = 0.95,
        expert_dropout: float = 0.05,
        adapter_rank: int = 32,
    ):
        super().__init__()
        self.router = TopKRouter(dim, n_experts, topk, balance_momentum=balance_momentum)
        self.experts = nn.ModuleList([ExpertMLP(dim, adapter_rank=adapter_rank) for _ in range(n_experts)])
        self.shared_expert = ExpertMLP(dim, adapter_rank=adapter_rank)
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout

    def forward(self, x: torch.Tensor):
        probs, idx, z_loss = self.router(x)
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

                token_positions = token_positions[:capacity]
                token_x = x[token_positions[:, 0], token_positions[:, 1]]

                if self.training and self.expert_dropout > 0.0:
                    keep = (torch.rand(token_x.size(0), device=token_x.device) > self.expert_dropout).float().unsqueeze(-1)
                    token_x = token_x * keep

                token_out = expert_net(token_x)
                out[token_positions[:, 0], token_positions[:, 1]] += (
                    token_out * slot_weight[token_positions[:, 0], token_positions[:, 1]]
                )

        # Shared global expert always contributes for stability.
        out = out + 0.2 * self.shared_expert(x)

        stats = {
            "router_probs": probs.detach(),
            "router_idx": idx.detach(),
            "expert_load": self.router.ema_load.detach().clone(),
            "router_z_loss": z_loss.detach(),
        }
        return out, stats
