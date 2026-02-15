import torch


class YaRNRoPE:
    def __init__(self, dim, base=10000, factor=8.0):
        half = dim // 2
        idx = torch.arange(half, dtype=torch.float32)

        low = max(1, half // 4)
        high = max(low + 1, half)

        scale = torch.ones(half, dtype=torch.float32)
        mask = (idx >= low) & (idx <= high)
        scale[idx > high] = factor
        scale[mask] = 1 + (factor - 1) * (idx[mask] - low) / (high - low)

        inv_freq = 1.0 / (base ** (idx / half))
        self.inv_freq = inv_freq * scale

    def apply(self, q, k, start_pos):
        inv_freq = self.inv_freq.to(device=q.device, dtype=q.dtype)
        t = torch.arange(start_pos, start_pos + q.size(2), device=q.device, dtype=q.dtype)
        freqs = torch.einsum("t,d->td", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[None, None, :, :]
        cos, sin = emb.cos(), emb.sin()

        def rotate(x):
            x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        return q, k
