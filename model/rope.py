import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, start_pos=0):
        B, T, H, D = q.shape
        pos = torch.arange(start_pos, start_pos + T, device=q.device)
        freqs = torch.einsum("t,d->td", pos, self.inv_freq)

        sin = freqs.sin()[None, :, None, :]
        cos = freqs.cos()[None, :, None, :]

        def rotate(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.cat(
                [x1 * cos - x2 * sin,
                 x1 * sin + x2 * cos],
                dim=-1
            )

        return rotate(q), rotate(k)
