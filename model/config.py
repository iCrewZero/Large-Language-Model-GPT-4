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

    local_window: int = 256
    global_stride: int = 64

    enable_mla: bool = True
    mla_latent_dim: int = 512
    mla_multires_levels: int = 3

    enable_moe: bool = True
    moe_experts: int = 8
    moe_topk: int = 2
    moe_capacity_factor: float = 1.25
    moe_balance_momentum: float = 0.95
    moe_expert_dropout: float = 0.05
    moe_adapter_rank: int = 32

    enable_mtp: bool = True
    mtp_steps: int = 3

    token_skip_threshold: float = 0.15

    adaptive_ffn_multiplier: int = 4

    enable_fp8_hooks: bool = False
    enable_nvfp4_hooks: bool = False
    enable_shape_checks: bool = True

    def validate(self) -> None:
        if self.dim % self.n_head != 0:
            raise ValueError("dim must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.moe_topk > self.moe_experts:
            raise ValueError("moe_topk must be <= moe_experts")
        if self.mtp_steps < 1:
            raise ValueError("mtp_steps must be >= 1")
        if not (0.0 <= self.token_skip_threshold <= 1.0):
            raise ValueError("token_skip_threshold must be in [0,1]")
        if self.mla_multires_levels < 1:
            raise ValueError("mla_multires_levels must be >= 1")
        if self.moe_adapter_rank < 1:
            raise ValueError("moe_adapter_rank must be >= 1")
