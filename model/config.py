from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Core transformer configuration with feature toggles for advanced training/inference."""

    vocab_size: int = 50280
    dim: int = 2048
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 8
    block_size: int = 4096

    rope_base: int = 10000
    rope_factor: float = 8.0

    dropout: float = 0.0

    # MLA
    enable_mla: bool = True
    mla_latent_dim: int = 512

    # MoE
    enable_moe: bool = True
    moe_experts: int = 8
    moe_topk: int = 2
    moe_capacity_factor: float = 1.25
    moe_balance_momentum: float = 0.95

    # MTP head
    enable_mtp: bool = True
    mtp_steps: int = 3

    # Systems / perf toggles
    enable_fp8_hooks: bool = False
    enable_nvfp4_hooks: bool = False
    enable_shape_checks: bool = True
