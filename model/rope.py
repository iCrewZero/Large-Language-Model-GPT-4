import torch

class YaRNRoPE:
    def __init__(self, dim, base=10000, factor=8.0):
        half = dim // 2
        idx = torch.arange(half)

        low = half
        high = half * 32

        scale = torch.ones(half)
        scale[idx > high] = factor

        mask = (idx >= low) & (idx <= high)
        scale[mask] = 1 + (factor - 1) * (idx[mask] - low) / (high - low)

        inv = 1.0 / (base ** (idx / half))
        self.inv_freq = (inv * scale).cuda()

    def apply(self, q, k, pos):
        t = torch.arange(pos, pos + q.size(2), device=q.device)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()

        def rotate(x):
            x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
            return torch.cat([-x2, x1], dim=-1)

        return q*cos + rotate(q)*sin, k*cos + rotate(k)*sin
