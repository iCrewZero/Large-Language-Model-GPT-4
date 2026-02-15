import torch


class SpeculativeDecoder:
    """Draft/target speculative decoding with support for multi-draft competition."""

    def __init__(self, draft_models, main_model, draft_steps: int = 4):
        if not isinstance(draft_models, (list, tuple)):
            draft_models = [draft_models]
        self.drafts = list(draft_models)
        self.main = main_model
        self.draft_steps = draft_steps

    @torch.no_grad()
    def _propose(self, input_ids):
        proposals = []
        for draft in self.drafts:
            proposal = input_ids
            for _ in range(self.draft_steps):
                draft_logits = draft(proposal)["logits"]
                draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
                proposal = torch.cat([proposal, draft_token], dim=1)
            proposals.append(proposal)
        return proposals

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32):
        device = next(self.main.parameters()).device
        input_ids = input_ids.to(device)
        target_len = input_ids.size(1) + max_new_tokens

        while input_ids.size(1) < target_len:
            proposals = self._propose(input_ids)
            best_candidate = None
            best_score = -float("inf")

            for proposal in proposals:
                out = self.main(proposal)
                logits = out["logits"][:, -self.draft_steps :, :]
                chosen = proposal[:, -self.draft_steps :]
                token_logp = torch.log_softmax(logits, dim=-1)
                score = token_logp.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).mean().item()
                if score > best_score:
                    best_score = score
                    best_candidate = proposal

            input_ids = best_candidate
            if input_ids.size(1) >= target_len:
                break

        return input_ids[:, :target_len]
