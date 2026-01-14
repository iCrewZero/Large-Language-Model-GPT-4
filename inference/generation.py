import torch
from model.gpt import GPT

model = GPT(32000, 512, 6, 8).cuda()
model.eval()

idx = torch.tensor([[1]], device="cuda")

for _ in range(50):
    logits = model(idx)
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)

print(idx)
