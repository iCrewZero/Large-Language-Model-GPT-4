import torch
import torch.nn.functional as F

from rl.grpo import grpo_loss
from training.losses import compute_all_losses, dynamic_weighted_loss
from training.precision import PrecisionManager
from training.prm import ProcessRewardModel
from training.scheduler import CurriculumLengthScheduler


class ContinuousBatcher:
    """Packs variable microbatches into a fixed token budget for continuous batching."""

    def __init__(self, max_tokens_per_step: int):
        self.max_tokens_per_step = max_tokens_per_step
        self._queue = []

    def add(self, sample):
        self._queue.append(sample)

    def pop_batch(self):
        if not self._queue:
            return None
        batch, used = [], 0
        while self._queue:
            nxt = self._queue[0]
            tokens = nxt["input_ids"].numel()
            if batch and used + tokens > self.max_tokens_per_step:
                break
            batch.append(self._queue.pop(0))
            used += tokens
        return batch


def _crop(item, seq_len: int):
    return {
        "input_ids": item["input_ids"][:seq_len],
        "labels": item["labels"][:seq_len],
    }


def _collate(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, labels = [], []
    for item in batch:
        inp = item["input_ids"]
        lbl = item["labels"]
        pad = max_len - inp.size(0)
        input_ids.append(F.pad(inp, (0, pad), value=0))
        labels.append(F.pad(lbl, (0, pad), value=-100))
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


def _inject_grad_noise(model, std: float):
    if std <= 0:
        return
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * std)


def train_step(
    model,
    batch,
    optimizer,
    scheduler=None,
    prm: ProcessRewardModel = None,
    group_size: int = 4,
    teacher_model=None,
    precision: PrecisionManager | None = None,
    grad_noise_std: float = 0.0,
    loss_ema_state: dict[str, float] | None = None,
):
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    loss_ema_state = loss_ema_state if loss_ema_state is not None else {}
    precision = precision if precision is not None else PrecisionManager()

    with precision.autocast():
        out = model(input_ids, return_hidden=True, return_router=True)
        teacher_repr = None
        if teacher_model is not None:
            with torch.no_grad():
                t_out = teacher_model(input_ids, return_hidden=True)
                teacher_repr = t_out["hidden"].detach()

        losses = compute_all_losses(out, labels, teacher_repr=teacher_repr)

        if prm is not None:
            token_logp = F.log_softmax(out["logits"][:, :-1, :], dim=-1)
            actions = labels[:, 1:].clamp_min(0)
            chosen_logp = token_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            rewards = prm.sequence_reward(out["hidden"].detach())
            losses["grpo"] = grpo_loss(chosen_logp, rewards, group_size=group_size)

        loss, dynamic_weights = dynamic_weighted_loss(losses, loss_ema_state)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    _inject_grad_noise(model, grad_noise_std)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler:
        scheduler.step()

    metrics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    metrics["loss"] = float(loss.detach().cpu().item())
    metrics["weights"] = dynamic_weights
    return metrics


def train_continuous(
    model,
    optimizer,
    stream,
    max_tokens_per_step: int,
    total_steps: int,
    curriculum_start_len: int,
    curriculum_end_len: int,
    scheduler=None,
    prm=None,
    **train_step_kwargs,
):
    batcher = ContinuousBatcher(max_tokens_per_step=max_tokens_per_step)
    curriculum = CurriculumLengthScheduler(curriculum_start_len, curriculum_end_len, total_steps)
    metrics = []

    for step, sample in enumerate(stream):
        seq_len = curriculum(step)
        batcher.add(_crop(sample, seq_len))
        packed = batcher.pop_batch()
        if packed is None:
            continue
        collated = _collate(packed)
        metrics.append(train_step(model, collated, optimizer, scheduler=scheduler, prm=prm, **train_step_kwargs))
    return metrics
