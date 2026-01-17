import torch

def grpo_loss(logprobs, rewards, groups):
    loss = 0
    for g in groups:
        r = rewards[g]
        baseline = r.mean()
        advantage = r - baseline
        loss += -(advantage * logprobs[g]).mean()
    return loss
