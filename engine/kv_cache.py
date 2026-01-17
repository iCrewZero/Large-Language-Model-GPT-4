import torch
from .paged_kv import KVState, PageAllocator

class PagedKVCache:
    def __init__(self, max_pages, page_size, n_head, head_dim, device):
        self.page_size = page_size
        self.alloc = PageAllocator(max_pages)

        self.K = torch.zeros(
            max_pages, n_head, page_size, head_dim,
            dtype=torch.int8, device=device
        )
        self.V = torch.zeros(
            max_pages, n_head, page_size, head_dim,
            dtype=torch.float16, device=device
        )
        self.K_scale = torch.zeros(
            max_pages, n_head, 1, 1, device=device
        )

    def append(self, state, k, v):
        if state.length % self.page_size == 0:
            state.pages.append(self.alloc.alloc())

        pid = state.pages[-1]
        off = state.length % self.page_size

        scale = k.abs().amax(dim=-1, keepdim=True) / 127.0 + 1e-6
        self.K[pid,:,off] = (k / scale).round().clamp(-128,127)
        self.K_scale[pid] = scale
        self.V[pid,:,off] = v

        state.length += 1

    def gather(self, state):
        ks, vs = [], []
        for pid in state.pages:
            ks.append(self.K[pid].float() * self.K_scale[pid])
            vs.append(self.V[pid])
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)
