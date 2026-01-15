import torch

class KVQuantizer:
    def __init__(self):
        self.scale = None
        self.initialized = False

    def _init_scale(self, k):
        with torch.no_grad():
            max_val = k.abs().amax(dim=(0, 2, 3), keepdim=True)
            scale = max_val / 127.0
            scale = torch.clamp(scale, min=1e-6)

        self.scale = scale
        self.initialized = True

    def fake_quant_k(self, k):
        if not self.initialized:
            self._init_scale(k)

        k_q = torch.round(k / self.scale).clamp(-128, 127)
        k_q = k_q * self.scale
        return k_q

    def real_quant_k(self, k):
        if not self.initialized:
            self._init_scale(k)

        k_i8 = torch.round(k / self.scale).clamp(-128, 127).to(torch.int8)
        return k_i8, self.scale

    def dequantize_k(self, k_i8):
        return k_i8.float() * self.scale
