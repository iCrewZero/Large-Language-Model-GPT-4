import torch

class RoPE:
    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base

    def _freqs(self, seq_len, device):
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device) / self.dim))
        t = torch.arange(seq_len, device=device)
        return torch.einsum("i,j->ij", t, inv)

    def __call__(self, q, k, start_pos):
        # q,k: [B, T, H, D]
        T = q.size(1)
        freqs = self._freqs(T + start_pos, q.device)[start_pos:]
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]

        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        return q, k
