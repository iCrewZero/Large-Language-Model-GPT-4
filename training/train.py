import torch
import torch.nn.functional as F

from rl.grpo import grpo_loss
from training.mtp_loss import mtp_loss
from training.prm import ProcessRewardModel


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


def train_step(model, batch, optimizer, scheduler=None, prm: ProcessRewardModel = None, group_size: int = 4):
    model.train()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    out = model(input_ids, return_hidden=True, return_router=True)
    logits = out["logits"]

    ce = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    loss = ce

    if "mtp_logits" in out:
        loss = loss + mtp_loss(out["mtp_logits"], labels)

    if prm is not None:
        token_logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        actions = labels[:, 1:].clamp_min(0)
        chosen_logp = token_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        rewards = prm.sequence_reward(out["hidden"].detach())
        loss = loss + grpo_loss(chosen_logp, rewards, group_size=group_size)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler:
        scheduler.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "ce": float(ce.detach().cpu().item()),
    }


def train_continuous(model, optimizer, stream, max_tokens_per_step: int, scheduler=None, prm=None):
    batcher = ContinuousBatcher(max_tokens_per_step=max_tokens_per_step)
    metrics = []
    for sample in stream:
        batcher.add(sample)
        packed = batcher.pop_batch()
        if packed is None:
            continue
        collated = _collate(packed)
        metrics.append(train_step(model, collated, optimizer, scheduler=scheduler, prm=prm))
    return metrics
