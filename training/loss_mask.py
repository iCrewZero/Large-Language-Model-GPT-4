import torch

def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()
