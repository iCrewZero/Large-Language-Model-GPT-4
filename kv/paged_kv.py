import torch


class KVPage:
    def __init__(self, k, v):
        self.k = k
        self.v = v
        self.score = 0.0
        self.age = 0


class PagedKVCache:
    """Simple dynamic KV page cache with score-age eviction."""

    def __init__(self, max_pages=512):
        self.pages = []
        self.max_pages = max_pages

    def add(self, k, v):
        if len(self.pages) >= self.max_pages:
            self.evict()
        self.pages.append(KVPage(k, v))

    def evict(self):
        if not self.pages:
            return
        scores = torch.tensor([p.score - 0.01 * p.age for p in self.pages])
        idx = torch.argmin(scores).item()
        del self.pages[idx]

    def feedback(self, attn_weights):
        delta = float(attn_weights.mean().item())
        for page in self.pages:
            page.score += delta
            page.age += 1

    def get(self):
        if not self.pages:
            return None, None
        ks = [p.k for p in self.pages]
        vs = [p.v for p in self.pages]
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)
