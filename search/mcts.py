import math
import torch


class MCTSNode:
    def __init__(self, tokens, parent=None):
        self.tokens = tokens
        self.parent = parent
        self.children = []
        self.value = 0.0
        self.visits = 0

    def ucb(self, c=1.4):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(math.log((self.parent.visits if self.parent else 1) + 1) / self.visits)


class MCTS:
    def __init__(self, model, verifier, tokenizer, depth=5):
        self.model = model
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.depth = depth

    @torch.no_grad()
    def search(self, input_ids):
        root = MCTSNode(input_ids)

        for _ in range(64):
            node = self.select(root)
            self.expand(node)
            score = self.evaluate(node)
            self.backprop(node, score)

        if not root.children:
            return input_ids
        return max(root.children, key=lambda n: n.visits).tokens

    def select(self, node):
        depth = 0
        while node.children and depth < self.depth:
            node = max(node.children, key=lambda n: n.ucb())
            depth += 1
        return node

    @torch.no_grad()
    def expand(self, node):
        out = self.model(node.tokens)
        topk = torch.topk(out["logits"][:, -1], 5, dim=-1).indices
        for t in topk[0]:
            child = MCTSNode(torch.cat([node.tokens, t.view(1, 1)], dim=1), parent=node)
            node.children.append(child)

    @torch.no_grad()
    def evaluate(self, node):
        out = self.model(node.tokens, return_hidden=True)
        score = out["value"].mean().item()
        return score

    def backprop(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent
