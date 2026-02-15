from contextlib import contextmanager

import torch


class PrecisionManager:
    """Runtime precision switch with FP8 hook points and overflow-safe backward wrapper."""

    def __init__(
        self,
        enable_fp8: bool = False,
        default_dtype: torch.dtype = torch.bfloat16,
        grad_scale_init: float = 2.0**12,
    ):
        self.enable_fp8 = enable_fp8
        self.default_dtype = default_dtype
        self.scaler = torch.cuda.amp.GradScaler(init_scale=grad_scale_init, enabled=torch.cuda.is_available())

    @contextmanager
    def autocast(self):
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self.default_dtype):
            yield

    def maybe_apply_fp8_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.enable_fp8:
            return tensor.clamp(min=-448.0, max=448.0)
        return tensor

    def backward_step(self, loss: torch.Tensor, optimizer, grad_clip: float | None = None, params=None) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        if grad_clip is not None and params is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
