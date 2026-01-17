class RolloutEngine:
    def __init__(self, model, prm, mcts):
        self.model = model
        self.prm = prm
        self.mcts = mcts

    def generate(self, tokens):
        if self.mcts:
            action = self.mcts.search(tokens)
            return action
        else:
            return self.model(tokens).argmax(-1)
