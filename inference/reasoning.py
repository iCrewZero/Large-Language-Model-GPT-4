import torch

def reasoning_chain(model, input_ids, steps=5):
    chain = input_ids
    for _ in range(steps):
        logits = model(chain)
        token = torch.argmax(logits[:, -1], dim=-1)
        chain = torch.cat([chain, token.unsqueeze(1)], dim=1)
    return chain
