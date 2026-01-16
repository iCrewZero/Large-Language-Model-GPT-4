import torch

class RoPE:
    def __init__(self, dim):
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.inv = inv

    def apply(self, q, k, pos):
        t = torch.arange(pos, pos + q.size(-2), device=q.device)
        f = torch.einsum("t,d->td", t, self.inv)
        emb = torch.cat([f,f], -1)
        cos, sin = emb.cos(), emb.sin()

        def rot(x):
            a,b = x.chunk(2,-1)
            return torch.cat([-b,a],-1)

        return q*cos + rot(q)*sin, k*cos + rot(k)*sin
