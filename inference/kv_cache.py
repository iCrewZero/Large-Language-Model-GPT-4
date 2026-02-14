import torch


class SequenceKVCache:
    """Simple per-layer KV cache with optional feedback trimming for reasoning revisions."""

    def __init__(self, n_layers: int):
        self.layers = [{"k": None, "v": None} for _ in range(n_layers)]

    def as_model_cache(self):
        return self.layers

    def append_feedback(self, layer_idx: int, keep_last_tokens: int):
        layer = self.layers[layer_idx]
        if layer["k"] is None:
            return
        layer["k"] = layer["k"][:, -keep_last_tokens:, :, :].contiguous()
        layer["v"] = layer["v"][:, -keep_last_tokens:, :, :].contiguous()

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
