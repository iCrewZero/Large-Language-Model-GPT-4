import torch

class PageAllocator:
    def __init__(self, n_pages):
        self.free = list(range(n_pages))

    def alloc(self):
        if not self.free:
            raise RuntimeError("KV pages exhausted")
        return self.free.pop()

    def free_page(self, pid):
        self.free.append(pid)


class KVState:
    def __init__(self):
        self.pages = []
        self.length = 0

    def reset(self):
        self.pages.clear()
        self.length = 0
