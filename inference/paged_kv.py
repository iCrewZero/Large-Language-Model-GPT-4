import torch

class PageAllocator:
    def __init__(self, num_pages: int, page_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.free_pages = list(range(num_pages))

    def alloc(self) -> int:
        if not self.free_pages:
            raise RuntimeError("Out of KV pages")
        return self.free_pages.pop()

    def free(self, pid: int):
        self.free_pages.append(pid)


class KVState:
    def __init__(self):
        self.pages = []
        self.seq_len = 0

    def reset(self, allocator: PageAllocator):
        for pid in self.pages:
            allocator.free(pid)
        self.pages.clear()
        self.seq_len = 0


def append_tokens(
    state: KVState,
    allocator: PageAllocator,
    num_tokens: int
):
    start = state.seq_len
    end = start + num_tokens

    first_page = start // allocator.page_size
    last_page = (end - 1) // allocator.page_size

    while len(state.pages) <= last_page:
        state.pages.append(allocator.alloc())

    state.seq_len = end


def write_kv_block(
    K_pool: torch.Tensor,
    V_pool: torch.Tensor,
    state: KVState,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    allocator: PageAllocator
):
    B, T, H, D = k_block.shape
    base = state.seq_len - T

    for t in range(T):
        token_idx = base + t
        page_idx = token_idx // allocator.page_size
        offset = token_idx % allocator.page_size
        pid = state.pages[page_idx]

        K_pool[pid, offset].copy_(k_block[:, t])
        V_pool[pid, offset].copy_(v_block[:, t])


def iter_kv_pages(
    K_pool: torch.Tensor,
    V_pool: torch.Tensor,
    state: KVState,
    upto_len: int,
    allocator: PageAllocator
):
    max_pages = (upto_len + allocator.page_size - 1) // allocator.page_size
    for i in range(max_pages):
        pid = state.pages[i]
        start = i * allocator.page_size
        end = min(start + allocator.page_size, upto_len)
        yield (
            K_pool[pid, : end - start],
            V_pool[pid, : end - start]
        )
