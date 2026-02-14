import torch


def load_imbalance(router_indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Monitoring metric only (not training loss): coefficient of variation of expert usage."""
    one_hot = torch.nn.functional.one_hot(router_indices, num_classes=n_experts).float()
    load = one_hot.mean(dim=(0, 1, 2))
    return load.std() / (load.mean() + 1e-6)
