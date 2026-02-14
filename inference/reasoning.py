import math
import torch


class MCTSNode:
    def __init__(self, tokens, prior=1.0, parent=None):
        self.tokens = tokens
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value(self):
        return self.value_sum / max(1, self.visits)


class MCTSReasoner:
    """Token-level MCTS over chain-of-thought expansions scored by PRM/value head."""

    def __init__(self, model, prm=None, simulations: int = 32, depth: int = 8, topk_expand: int = 4):
        self.model = model
        self.prm = prm
        self.simulations = simulations
        self.depth = depth
        self.topk_expand = topk_expand

    def _ucb(self, parent, child):
        explore = math.sqrt(math.log(parent.visits + 1) / (child.visits + 1e-6))
        return child.value + 1.4 * child.prior * explore

    @torch.no_grad()
    def _score(self, tokens):
        out = self.model(tokens, return_hidden=True)
        if self.prm is not None:
            return float(self.prm.sequence_reward(out["hidden"]).item())
        return float(out["value"][:, -1].mean().item())

    @torch.no_grad()
    def search(self, input_ids):
        root = MCTSNode(input_ids)

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.children and len(path) < self.depth:
                node = max(node.children.values(), key=lambda c: self._ucb(path[-1], c))
                path.append(node)

            if len(path) < self.depth:
                out = self.model(node.tokens)
                probs = torch.softmax(out["logits"][:, -1, :], dim=-1)
                topk_prob, topk_idx = torch.topk(probs, self.topk_expand, dim=-1)
                for i in range(self.topk_expand):
                    tok = topk_idx[0, i].view(1, 1)
                    child_tokens = torch.cat([node.tokens, tok], dim=1)
                    node.children[int(tok.item())] = MCTSNode(
                        child_tokens,
                        prior=float(topk_prob[0, i].item()),
                        parent=node,
                    )
                node = max(node.children.values(), key=lambda c: c.prior)
                path.append(node)

            reward = self._score(node.tokens)
            for p in path:
                p.visits += 1
                p.value_sum += reward

        if not root.children:
            return input_ids
        best = max(root.children.values(), key=lambda c: c.visits)
        return best.tokens
