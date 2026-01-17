import math
import torch

class MCTSNode:
def init(self, tokens, parent=None):
self.tokens = tokens
self.parent = parent
self.children = []
self.value = 0.0
self.visits = 0

def ucb(self, c=1.4):
    if self.visits == 0:
        return float("inf")
    return self.value / self.visits + c * math.sqrt(
        math.log(self.parent.visits + 1) / self.visits
    )


class MCTS:
def init(self, model, verifier, tokenizer, depth=5):
self.model = model
self.verifier = verifier
self.tokenizer = tokenizer
self.depth = depth

def search(self, input_ids):
    root = MCTSNode(input_ids)

    for _ in range(64):
        node = self.select(root)
        self.expand(node)
        score = self.evaluate(node)
        self.backprop(node, score)

    return max(root.children, key=lambda n: n.visits).tokens

def select(self, node):
    while node.children:
        node = max(node.children, key=lambda n: n.ucb())
    return node

def expand(self, node):
    logits = self.model(node.tokens)
    topk = torch.topk(logits[:, -1], 5, dim=-1).indices
    for t in topk[0]:
        child = MCTSNode(
            torch.cat([node.tokens, t.view(1, 1)], dim=1),
            parent=node
        )
        node.children.append(child)

def evaluate(self, node):
    hidden = self.model(node.tokens, return_hidden=True)
    score = self.verifier(hidden).mean().item()
    return score

def backprop(self, node, score):
    while node:
        node.visits += 1
        node.value += score
        node = node.parent
