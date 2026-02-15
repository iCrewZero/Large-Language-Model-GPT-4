from typing import Dict, TypedDict

import torch


class TrainBatch(TypedDict):
    input_ids: torch.Tensor  # [B, T] long
    labels: torch.Tensor     # [B, T] long, -100 ignored


class TrainMetrics(TypedDict, total=False):
    loss: float
    ce: float
    mtp: float
    contrastive: float
    distill: float
    grpo: float
    weights: Dict[str, float]
