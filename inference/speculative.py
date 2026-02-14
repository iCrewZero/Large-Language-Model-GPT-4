import torch


class SpeculativeDecoder:
    """Draft/target speculative decoding with accept-reject verification."""

    def __init__(self, draft_model, main_model, verifier=None, draft_steps: int = 4):
        self.draft = draft_model
        self.main = main_model
        self.verifier = verifier
        self.draft_steps = draft_steps

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32):
        device = next(self.main.parameters()).device
        input_ids = input_ids.to(device)

        target_len = input_ids.size(1) + max_new_tokens
        while input_ids.size(1) < target_len:
            proposal = input_ids
            for _ in range(self.draft_steps):
                draft_logits = self.draft(proposal)["logits"]
                draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
                proposal = torch.cat([proposal, draft_token], dim=1)

            main_out = self.main(proposal)
            main_logits = main_out["logits"][:, -self.draft_steps :, :]
            main_tokens = torch.argmax(main_logits, dim=-1)
            proposed_tokens = proposal[:, -self.draft_steps :]

            matches = (main_tokens == proposed_tokens).all(dim=-1)
            if self.verifier is not None:
                verify = main_out["value"][:, -1].sigmoid() > 0.5
                matches = matches & verify

            if matches.item():
                input_ids = proposal
            else:
                next_token = main_tokens[:, :1]
                input_ids = torch.cat([input_ids, next_token], dim=1)

            if input_ids.size(1) >= target_len:
                break

        return input_ids
