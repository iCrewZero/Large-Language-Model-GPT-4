import torch
import torch.nn.functional as F


def mtp_loss(mtp_logits, labels, weight: float = 0.1):
    """Multi-token prediction CE for shifted horizons."""
    if not mtp_logits:
        return labels.new_zeros((), dtype=torch.float32)

    total = 0.0
    for step, logits in enumerate(mtp_logits, start=1):
        if logits.size(1) <= step:
            continue
        pred = logits[:, :-step, :].contiguous()
        target = labels[:, step:].contiguous()
        total = total + F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
    return total * weight
