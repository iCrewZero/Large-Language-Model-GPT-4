import torch

class KVPage:
def init(self, k, v):
self.k = k
self.v = v
self.score = 0.0
self.age = 0

class PagedKVCache:
def init(self, max_pages=512):
self.pages = []
self.max_pages = max_pages

def add(self, k, v):
    if len(self.pages) >= self.max_pages:
        self.evict()
    self.pages.append(KVPage(k, v))

def evict(self):
    scores = torch.tensor(
        [p.score - 0.01 * p.age for p in self.pages]
    )
    idx = torch.argmin(scores).item()
    del self.pages[idx]

def feedback(self, attn_weights):
    for i, page in enumerate(self.pages):
        page.score += attn_weights.mean().item()
        page.age += 1

def get(self):
    ks = [p.k for p in self.pages]
    vs = [p.v for p in self.pages]
    return torch.cat(ks, dim=2), torch.cat(vs, dim=2)
