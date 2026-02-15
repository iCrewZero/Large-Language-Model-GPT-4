import torch
import torch.nn.functional as F

from training.mtp_loss import mtp_loss


def contrastive_reasoning_loss(hidden: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    # Uses adjacent tokens as positives and random roll as negatives.
    if hidden.size(1) < 2:
        return hidden.new_zeros(())
    h = F.normalize(hidden, dim=-1)
    pos = (h[:, :-1] * h[:, 1:]).sum(dim=-1)
    neg = (h[:, :-1] * h.roll(shifts=1, dims=1)[:, :-1]).sum(dim=-1)
    return F.relu(margin - pos + neg).mean()


def distillation_loss(student_repr: torch.Tensor, teacher_repr: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_repr, teacher_repr)


def dynamic_weighted_loss(loss_dict: dict[str, torch.Tensor], ema_state: dict[str, float], momentum: float = 0.95):
    weighted = 0.0
    weights = {}
    for k, v in loss_dict.items():
        value = float(v.detach().item())
        prev = ema_state.get(k, value)
        ema = momentum * prev + (1.0 - momentum) * value
        ema_state[k] = ema
        inv = 1.0 / max(ema, 1e-6)
        weights[k] = inv
    norm = sum(weights.values()) + 1e-8
    for k, v in loss_dict.items():
        weighted = weighted + v * (weights[k] / norm)
    return weighted, weights


def next_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def compute_all_losses(out, labels, teacher_repr=None):
    losses = {}
    losses["ce"] = next_token_ce(out["logits"], labels)
    if "mtp_logits" in out:
        losses["mtp"] = mtp_loss(out["mtp_logits"], labels)
    if "hidden" in out:
        losses["contrastive"] = contrastive_reasoning_loss(out["hidden"])
    if teacher_repr is not None:
        losses["distill"] = distillation_loss(out["distill"], teacher_repr)
    return losses
