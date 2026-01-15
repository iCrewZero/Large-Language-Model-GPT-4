import torch
from dataclasses import dataclass
from typing import List

@dataclass
class KVState:
    pages: List[int]
    seqlen: int

    def rollback(self, new_len, allocator):
        keep_pages = (new_len + allocator.page_size - 1) // allocator.page_size
        for pid in self.pages[keep_pages:]:
            allocator.free(pid)
        self.pages = self.pages[:keep_pages]
        self.seqlen = new_len

class KVAllocator:
    def __init__(self, num_pages, n_heads, head_dim, page_size):
        self.page_size = page_size
        self.free = list(range(num_pages))
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.K = torch.empty(num_pages, page_size, n_heads, head_dim, dtype=torch.int8, device="cuda")
        self.V = torch.empty_like(self.K)
        self.K_scale = torch.empty(num_pages, 1, n_heads, 1, dtype=torch.float16, device="cuda")
        self.V_scale = torch.empty_like(self.K_scale)

    def alloc(self):
        assert self.free, "OOM: KV pages exhausted"
        return self.free.pop()

    def free(self, pid):
        self.free.append(pid)

def append_token_state(state: KVState, allocator: KVAllocator, page_size):
    if state.seqlen % page_size == 0:
        pid = allocator.alloc()
        state.pages.append(pid)
    state.seqlen += 1

def write_kv_token(K_pool, V_pool, state: KVState, k_tensor, v_tensor, page_size):
    token_idx = state.seqlen - 1
    page_idx = token_idx // page_size
    offset = token_idx % page_size
    pid = state.pages[page_idx]
    K_pool[pid, offset] = k_tensor
    V_pool[pid, offset] = v_tensor

def gather_kv(K_pool, V_pool, state: KVState, upto_len, page_size):
    needed_pages = (upto_len + page_size - 1) // page_size
    ks, vs = [], []
    for pid in state.pages[:needed_pages]:
        ks.append(K_pool[pid].float())
        vs.append(V_pool[pid])
    K = torch.cat(ks, dim=1)
    V = torch.cat(vs, dim=1)
    if K.size(1) > upto_len:
        K = K[:, :upto_len, :]
        V = V[:, :upto_len, :]
    return K, V
