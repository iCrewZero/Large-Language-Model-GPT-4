import torch

from inference.kv_cache import SequenceKVCache
from inference.reasoning import MCTSReasoner, self_refine_chain, tree_of_thought_beam
from inference.speculative import SpeculativeDecoder


class ContinuousRequestBatcher:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.queue = []

    def add_prompt(self, prompt):
        self.queue.append(prompt)

    def next_batch(self):
        if not self.queue:
            return []
        batch = self.queue[: self.max_batch_size]
        self.queue = self.queue[self.max_batch_size :]
        return batch


class Engine:
    """Unified generation engine with continuous batching and prefill/decode split hooks."""

    def __init__(self, model, tokenizer, cfg, draft_models=None, prm=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.batcher = ContinuousRequestBatcher(cfg.get("max_batch_size", 16))

        self.speculative = None
        if draft_models is not None and cfg.get("enable_speculative", True):
            self.speculative = SpeculativeDecoder(draft_models, model, draft_steps=cfg.get("speculative_steps", 4))

        self.reasoner = None
        if cfg.get("enable_mcts", False):
            self.reasoner = MCTSReasoner(
                model,
                prm=prm,
                simulations=cfg.get("mcts_simulations", 16),
                depth=cfg.get("mcts_depth", 6),
            )

    @torch.no_grad()
    def _prefill(self, ids, kv_cache):
        _ = self.model(ids, start_pos=0, kv_cache=kv_cache)

    @torch.no_grad()
    def _decode(self, ids, kv_cache, max_new):
        out_ids = ids
        start_pos = ids.size(1)
        for _ in range(max_new):
            out = self.model(out_ids[:, -1:], start_pos=start_pos, kv_cache=kv_cache)
            nxt = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            out_ids = torch.cat([out_ids, nxt], dim=1)
            start_pos += 1
        return out_ids

    @torch.no_grad()
    def generate(self, prompts, max_new=128, reasoning_mode=None, memory_bank=None):
        for p in prompts:
            self.batcher.add_prompt(p)

        device = next(self.model.parameters()).device
        outputs = []

        while True:
            batch_prompts = self.batcher.next_batch()
            if not batch_prompts:
                break

            for prompt in batch_prompts:
                ids = torch.tensor(self.tokenizer.encode(prompt), device=device).unsqueeze(0)

                if reasoning_mode == "mcts" and self.reasoner is not None:
                    ids = self.reasoner.search(ids)
                elif reasoning_mode == "beam":
                    ids = tree_of_thought_beam(self.model, ids, beam_width=self.cfg.get("beam_width", 4))
                elif reasoning_mode == "refine":
                    ids = self_refine_chain(self.model, ids, memory_bank=memory_bank)

                kv = SequenceKVCache(
                    n_layers=len(self.model.blocks),
                    max_tokens=self.cfg.get("kv_max_tokens", 8192),
                    keep_tokens_on_evict=self.cfg.get("kv_evict_keep", 2048),
                )
                kv_cache = kv.as_model_cache()
                self._prefill(ids, kv_cache)

                if self.speculative is not None:
                    out_ids = self.speculative.generate(ids, max_new_tokens=max_new)
                else:
                    out_ids = self._decode(ids, kv_cache, max_new=max_new)

                kv.evict_if_needed()
                outputs.append(self.tokenizer.decode(out_ids[0].tolist()))

        return outputs
