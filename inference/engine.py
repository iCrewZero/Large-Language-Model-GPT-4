import torch

class Engine:
    def __init__(self, model, tok):
        self.m = model
        self.t = tok

    @torch.no_grad()
    def generate(self, prompt, max_new=128):
        ids = torch.tensor([self.t.encode(prompt)]).cuda()
        for _ in range(max_new):
            out = self.m(ids)
            nxt = out["logits"][:,-1].argmax(-1,keepdim=True)
            ids = torch.cat([ids,nxt],1)
        return self.t.decode(ids[0].tolist())
