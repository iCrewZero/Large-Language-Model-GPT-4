from contextlib import contextmanager

import torch


class PrecisionManager:
    """Runtime precision switch with FP8 hook points."""

    def __init__(self, enable_fp8: bool = False, default_dtype: torch.dtype = torch.bfloat16):
        self.enable_fp8 = enable_fp8
        self.default_dtype = default_dtype

    @contextmanager
    def autocast(self):
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self.default_dtype):
            yield

    def maybe_apply_fp8_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        # Hook placeholder for future backend-specific FP8 kernels.
        if self.enable_fp8:
            return tensor.clamp(min=-448.0, max=448.0)
        return tensor
