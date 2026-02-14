from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from model.config import ModelConfig


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    max_steps: int = 500_000
    log_interval: int = 50
    save_interval: int = 1_000

    max_tokens_per_step: int = 8_192
    enable_grpo: bool = True
    group_size: int = 4
    enable_prm: bool = True
    enable_continuous_batching: bool = True
    enable_fp8: bool = False
    grad_noise_std: float = 0.0

    curriculum_start_len: int = 256
    curriculum_end_len: int = 4096

    loss_ce_weight: float = 1.0
    loss_mtp_weight: float = 0.1
    loss_contrastive_weight: float = 0.05
    loss_distill_weight: float = 0.1


@dataclass
class InferenceConfig:
    enable_speculative: bool = True
    enable_multi_draft: bool = True
    enable_mcts: bool = True
    max_batch_size: int = 16

    speculative_steps: int = 4
    mcts_simulations: int = 32
    mcts_depth: int = 8
    beam_width: int = 4

    kv_max_tokens: int = 8192
    kv_evict_keep: int = 2048

    prefill_chunk_size: int = 1024
    decode_chunk_size: int = 1


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(model_path: str | Path, train_path: str | Path):
    model_data = _load_yaml(model_path)
    train_data = _load_yaml(train_path)
    model_cfg = ModelConfig(**model_data)
    train_cfg = TrainingConfig(**train_data)
    infer_cfg = InferenceConfig(**{k: v for k, v in train_data.items() if k in InferenceConfig.__annotations__})
    return model_cfg, train_cfg, infer_cfg
