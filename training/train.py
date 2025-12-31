import torch
from model.gpt import GPT

def train(model, loader, optimizer):
    model.train()
    for x, y in loader:
        logits, loss = model(x.cuda(), y.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
