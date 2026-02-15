from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class InferRequest:
    prompt: str
    max_new_tokens: int = 128
    reasoning_mode: Optional[str] = None


@dataclass
class InferResponse:
    text: str
    token_ids: torch.Tensor
