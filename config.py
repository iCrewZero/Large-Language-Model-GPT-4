from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from model.config import ModelConfig

T = TypeVar("T")


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

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.max_tokens_per_step < 1:
            raise ValueError("max_tokens_per_step must be >= 1")


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

    def validate(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.speculative_steps < 1:
            raise ValueError("speculative_steps must be >= 1")
        if self.kv_evict_keep > self.kv_max_tokens:
            raise ValueError("kv_evict_keep must be <= kv_max_tokens")


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        if any(ch in v for ch in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def _load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    """Minimal key: value parser for flat config files used in this repo.

    Supports booleans, ints, floats and strings. Ignores comments and blank lines.
    """
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
    return data


def _filter_for_dataclass(data: Dict[str, Any], cls: Type[T]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def load_configs(model_path: str | Path, train_path: str | Path):
    model_data = _load_simple_yaml(model_path)
    train_data = _load_simple_yaml(train_path)

    model_cfg = ModelConfig(**_filter_for_dataclass(model_data, ModelConfig))
    train_cfg = TrainingConfig(**_filter_for_dataclass(train_data, TrainingConfig))
    infer_cfg = InferenceConfig(**_filter_for_dataclass(train_data, InferenceConfig))

    model_cfg.validate()
    train_cfg.validate()
    infer_cfg.validate()
    return model_cfg, train_cfg, infer_cfg
