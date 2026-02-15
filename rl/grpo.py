import torch


def grpo_loss(log_probs: torch.Tensor, rewards: torch.Tensor, group_size: int = 4) -> torch.Tensor:
    """Group Relative Policy Optimization objective.

    Args:
        log_probs: [B, T] log probabilities for sampled actions.
        rewards: [B] scalar reward per sampled sequence.
    """
    batch = rewards.size(0)
    if batch % group_size != 0:
        group_size = batch

    grouped_rewards = rewards.view(-1, group_size)
    baseline = grouped_rewards.mean(dim=-1, keepdim=True)
    advantage = (grouped_rewards - baseline).reshape(-1)

    token_logp = log_probs.mean(dim=-1)
    return -(advantage.detach() * token_logp).mean()
