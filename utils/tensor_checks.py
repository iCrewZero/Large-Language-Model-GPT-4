from typing import Iterable, Sequence

import torch


def assert_rank(t: torch.Tensor, rank: int, name: str) -> None:
    if t.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {tuple(t.shape)}")


def assert_shape_prefix(t: torch.Tensor, prefix: Sequence[int], name: str) -> None:
    if len(prefix) > t.dim():
        raise ValueError(f"{name} prefix length exceeds tensor rank")
    for i, p in enumerate(prefix):
        if p != -1 and t.shape[i] != p:
            raise ValueError(f"{name} dim[{i}] expected {p}, got {t.shape[i]}")


def assert_dtype(t: torch.Tensor, allowed: Iterable[torch.dtype], name: str) -> None:
    allowed_set = set(allowed)
    if t.dtype not in allowed_set:
        raise ValueError(f"{name} dtype {t.dtype} not in {allowed_set}")


def assert_same_device(*tensors: torch.Tensor) -> None:
    devices = {t.device for t in tensors if t is not None}
    if len(devices) > 1:
        raise ValueError(f"tensors on different devices: {devices}")


def safe_clamp_logits(logits: torch.Tensor, clip_value: float = 30.0) -> torch.Tensor:
    if clip_value <= 0:
        return logits
    return logits.clamp(min=-clip_value, max=clip_value)
