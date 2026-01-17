class LLMEngine:
    def __init__(self, model, draft, verifier, tokenizer):
        self.model = model
        self.draft = draft
        self.verifier = verifier
        self.tokenizer = tokenizer

    def generate(self, input_ids, mode="speculative"):
        if mode == "speculative":
            return SpeculativeDecoder(
                self.draft, self.model, self.verifier
            ).generate(input_ids)

        if mode == "reason":
            return reasoning_chain(self.model, input_ids)

        if mode == "mcts":
            return MCTS(
                self.model, self.verifier, self.tokenizer
            ).search(input_ids)
