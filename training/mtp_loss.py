import torch
import torch.nn.functional as F

def mtp_loss(hidden, labels, k=3):
    loss = 0.0
    for i in range(1, k+1):
        pred = hidden[:, :-i]
        target = labels[:, i:]
        loss += F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1)
        )
    return loss * 0.1
