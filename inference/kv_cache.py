import torch


class SequenceKVCache:
    """Per-layer KV cache with token budget eviction and feedback trimming."""

    def __init__(self, n_layers: int, max_tokens: int = 8192, keep_tokens_on_evict: int = 2048):
        self.layers = [{"k": None, "v": None} for _ in range(n_layers)]
        self.max_tokens = max_tokens
        self.keep_tokens_on_evict = keep_tokens_on_evict

    def as_model_cache(self):
        return self.layers

    def append_feedback(self, layer_idx: int, keep_last_tokens: int):
        layer = self.layers[layer_idx]
        if layer["k"] is None:
            return
        layer["k"] = layer["k"][:, -keep_last_tokens:, :, :].contiguous()
        layer["v"] = layer["v"][:, -keep_last_tokens:, :, :].contiguous()

    def evict_if_needed(self):
        for layer in self.layers:
            if layer["k"] is None:
                continue
            if layer["k"].size(1) > self.max_tokens:
                layer["k"] = layer["k"][:, -self.keep_tokens_on_evict :, :, :].contiguous()
                layer["v"] = layer["v"][:, -self.keep_tokens_on_evict :, :, :].contiguous()

    def clear(self):
        for layer in self.layers:
            layer["k"] = None
            layer["v"] = None

    def to(self, device):
        for layer in self.layers:
            if layer["k"] is not None:
                layer["k"] = layer["k"].to(device)
                layer["v"] = layer["v"].to(device)
        return self
