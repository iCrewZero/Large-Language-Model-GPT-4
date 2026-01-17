import torch
import torch.nn.functional as F
from training.mtp_loss import mtp_loss
from training.prm import ProcessRewardModel

def grpo_loss(logits, actions, rewards):
    logp = F.log_softmax(logits, dim=-1)
    selected = logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return -(selected * rewards).mean()

def train_step(
    model,
    batch,
    optimizer,
    scheduler=None,
    prm: ProcessRewardModel = None
):
    model.train()
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    out = model(input_ids, return_hidden=True, return_router=True)

    logits = out["logits"]
    hidden = out["hidden"]
    router_stats = out.get("router_probs", None)

    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1)
    )

    loss = ce

    # 🔹 Multi-Token Prediction
    loss += mtp_loss(hidden, labels)

    # 🔹 MoE load balance (aux-loss-free)
    if router_stats is not None:
        load = router_stats.mean(dim=0)
        loss += load.var() * 0.01

    # 🔹 Process Reward Model (optional RL phase)
    if prm is not None:
        rewards = prm(hidden.detach())
        loss += grpo_loss(logits[:, :-1], labels[:, 1:], rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler:
        scheduler.step()

    return loss.item()
