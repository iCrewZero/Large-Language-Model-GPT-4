import torch
import torch.nn.functional as F

from rl.grpo import grpo_loss
from training.contracts import TrainBatch, TrainMetrics
from training.losses import compute_all_losses, dynamic_weighted_loss
from training.precision import PrecisionManager
from training.prm import ProcessRewardModel
from training.scheduler import CurriculumLengthScheduler
from utils.tensor_checks import assert_dtype, assert_rank, assert_same_device


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


def train_step(
    model,
    batch: TrainBatch,
    optimizer,
    scheduler=None,
    prm: ProcessRewardModel = None,
    group_size: int = 4,
    teacher_model=None,
    precision: PrecisionManager | None = None,
    grad_noise_std: float = 0.0,
    loss_ema_state: dict[str, float] | None = None,
    loss_ema_momentum: float = 0.95,
    grad_clip_norm: float = 1.0,
    router_reg_weight: float = 0.01,
    activation_sparsity_weight: float = 0.001,
    token_entropy_weight: float = 0.001,
    loss_ce_weight: float = 1.0,
    loss_mtp_weight: float = 0.1,
    loss_contrastive_weight: float = 0.05,
    loss_distill_weight: float = 0.1,
    logit_clip: float = 30.0,
) -> TrainMetrics:
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    assert_rank(input_ids, 2, "batch.input_ids")
    assert_rank(labels, 2, "batch.labels")
    assert_dtype(input_ids, [torch.long, torch.int64], "batch.input_ids")
    assert_dtype(labels, [torch.long, torch.int64], "batch.labels")
    assert_same_device(input_ids, labels)

    loss_ema_state = loss_ema_state if loss_ema_state is not None else {}
    precision = precision if precision is not None else PrecisionManager()

    with precision.autocast():
        out = model(input_ids, return_hidden=True, return_router=True)
        out["logits"] = out["logits"].clamp(min=-logit_clip, max=logit_clip)
        teacher_repr = None
        if teacher_model is not None:
            with torch.no_grad():
                t_out = teacher_model(input_ids, return_hidden=True)
                teacher_repr = t_out["hidden"].detach()

        losses = compute_all_losses(out, labels, teacher_repr=teacher_repr)

        if "ce" in losses:
            losses["ce"] = losses["ce"] * loss_ce_weight
        if "mtp" in losses:
            losses["mtp"] = losses["mtp"] * loss_mtp_weight
        if "contrastive" in losses:
            losses["contrastive"] = losses["contrastive"] * loss_contrastive_weight
        if "distill" in losses:
            losses["distill"] = losses["distill"] * loss_distill_weight

        if prm is not None:
            token_logp = F.log_softmax(out["logits"][:, :-1, :], dim=-1)
            actions = labels[:, 1:].clamp_min(0)
            chosen_logp = token_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            rewards = prm.sequence_reward(out["hidden"].detach())
            losses["grpo"] = grpo_loss(chosen_logp, rewards, group_size=group_size)

        if "router_reg" in losses:
            losses["router_reg"] = losses["router_reg"] * router_reg_weight
        if "activation_sparsity" in losses:
            losses["activation_sparsity"] = losses["activation_sparsity"] * activation_sparsity_weight
        if "token_entropy" in losses:
            losses["token_entropy"] = losses["token_entropy"] * token_entropy_weight

        loss, dynamic_weights = dynamic_weighted_loss(losses, loss_ema_state, momentum=loss_ema_momentum)

    optimizer.zero_grad(set_to_none=True)
    precision.backward_step(
        loss,
        optimizer,
        grad_clip=grad_clip_norm,
        params=model.parameters(),
        grad_noise_std=grad_noise_std,
    )
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
