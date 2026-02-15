from typing import List, Optional, Protocol, TypedDict

import torch


class KVLayerState(TypedDict):
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]


KVCacheState = List[KVLayerState]


class ModelOutput(TypedDict, total=False):
    logits: torch.Tensor      # [B, T, V]
    value: torch.Tensor       # [B, T]
    prm: torch.Tensor         # [B, T]
    distill: torch.Tensor     # [B, T, C]
    retrieval_scores: Optional[torch.Tensor]  # [B, M] or None
    mtp_logits: List[torch.Tensor]            # list[[B, T, V]]
    hidden: torch.Tensor      # [B, T, C]
    router: list


class BaseLLM(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCacheState] = None,
        return_hidden: bool = False,
        return_router: bool = False,
        memory_bank: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        ...


class BaseReasoner(Protocol):
    def search(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...


class BaseSpecDecoder(Protocol):
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        ...


class BasePRM(Protocol):
    def sequence_reward(self, hidden: torch.Tensor) -> torch.Tensor:
        ...


class BaseKVCache(Protocol):
    def as_model_cache(self) -> KVCacheState:
        ...

    def evict_if_needed(self) -> None:
        ...

    def clear(self) -> None:
        ...
