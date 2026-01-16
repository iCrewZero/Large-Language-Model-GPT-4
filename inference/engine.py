import torch

class Engine:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50):
        for _ in range(max_new_tokens):
            logits = self.model(input_ids)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
