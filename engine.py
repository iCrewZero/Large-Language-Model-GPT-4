from inference.engine import Engine


class LLMEngine:
    """High-level facade over inference.Engine."""

    def __init__(self, model, tokenizer, cfg, draft_models=None, prm=None):
        self.engine = Engine(model, tokenizer, cfg, draft_models=draft_models, prm=prm)

    def generate(self, prompts, max_new=128, mode="default"):
        reasoning_mode = None
        if mode in {"mcts", "beam", "refine"}:
            reasoning_mode = mode
        return self.engine.generate(prompts, max_new=max_new, reasoning_mode=reasoning_mode)
