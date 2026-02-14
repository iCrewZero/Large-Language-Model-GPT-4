import torch

from inference.kv_cache import SequenceKVCache
from inference.reasoning import MCTSReasoner
from inference.speculative import SpeculativeDecoder


class Engine:
    """Unified generation engine with KV caching, continuous batching, speculative decoding, and MCTS."""

    def __init__(self, model, tokenizer, cfg, draft_model=None, prm=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.speculative = None
        if draft_model is not None and cfg.get("enable_speculative", True):
            self.speculative = SpeculativeDecoder(draft_model, model, verifier=model.verifier)

        self.reasoner = None
        if cfg.get("enable_mcts", False):
            self.reasoner = MCTSReasoner(
                model,
                prm=prm,
                simulations=cfg.get("mcts_simulations", 16),
                depth=cfg.get("mcts_depth", 6),
            )

    @torch.no_grad()
    def generate(self, prompts, max_new=128, use_mcts=False):
        # Continuous batching over prompt list.
        device = next(self.model.parameters()).device
        encoded = [torch.tensor(self.tokenizer.encode(p), device=device).unsqueeze(0) for p in prompts]
        results = []

        for ids in encoded:
            if use_mcts and self.reasoner is not None:
                ids = self.reasoner.search(ids)

            kv_cache = SequenceKVCache(n_layers=len(self.model.blocks)).as_model_cache()
            if self.speculative is not None:
                out_ids = self.speculative.generate(ids, max_new_tokens=max_new)
            else:
                out_ids = ids
                start_pos = 0
                for _ in range(max_new):
                    out = self.model(out_ids[:, -1:], start_pos=start_pos, kv_cache=kv_cache)
                    nxt = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                    out_ids = torch.cat([out_ids, nxt], dim=1)
                    start_pos += 1

            results.append(self.tokenizer.decode(out_ids[0].tolist()))
        return results
