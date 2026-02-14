import torch
import torch.nn as nn

from model.block import Block
from model.config import ModelConfig
from model.heads import DistillationHead, RetrievalHead
from model.rmsnorm import RMSNorm
from model.verifier import VerifierHead
from training.mtp import MTPHead
from training.prm import ProcessRewardModel


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.dim)

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.verifier = VerifierHead(cfg.dim)
        self.prm_head = ProcessRewardModel(cfg.dim)
        self.distill_head = DistillationHead(cfg.dim)
        self.retrieval_head = RetrievalHead(cfg.dim)
        self.mtp_head = MTPHead(cfg.dim, cfg.vocab_size, cfg.mtp_steps) if cfg.enable_mtp else None

    def init_kv_cache(self, batch_size: int):
        _ = batch_size
        return [{"k": None, "v": None} for _ in range(self.cfg.n_layer)]

    def _validate_shapes(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {input_ids.shape}")
        if input_ids.size(1) > self.cfg.block_size:
            raise ValueError("sequence length exceeds configured block_size")

    def forward(
        self,
        input_ids,
        start_pos: int = 0,
        kv_cache=None,
        return_hidden: bool = False,
        return_router: bool = False,
        memory_bank=None,
    ):
        if self.cfg.enable_shape_checks:
            self._validate_shapes(input_ids)

        x = self.embed(input_ids)
        router_stats = []

        for i, block in enumerate(self.blocks):
            x, stats = block(x, pos=start_pos, kv_cache=kv_cache, layer_idx=i)
            if stats:
                router_stats.append(stats)

        x = self.norm(x)
        logits = self.lm_head(x)
        out = {
            "logits": logits,
            "value": self.verifier(x),
            "prm": self.prm_head(x),
            "distill": self.distill_head(x),
            "retrieval_scores": self.retrieval_head(x, memory_bank=memory_bank),
        }

        if self.mtp_head is not None:
            out["mtp_logits"] = self.mtp_head(x)

        if return_hidden:
            out["hidden"] = x

        if return_router and router_stats:
            out["router"] = router_stats

        return out
