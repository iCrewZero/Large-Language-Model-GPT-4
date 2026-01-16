import torch

class SpeculativeDecoder:
def init(self, draft_model, main_model, verifier):
self.draft = draft_model
self.main = main_model
self.verifier = verifier

def generate(self, input_ids, max_new_tokens=32):
    for _ in range(max_new_tokens):
        draft_logits = self.draft(input_ids)
        draft_token = torch.argmax(draft_logits[:, -1], dim=-1)

        candidate = torch.cat(
            [input_ids, draft_token.unsqueeze(1)], dim=1
        )

        main_logits = self.main(candidate)
        verify_score = self.verifier(main_logits[:, -1])

        if verify_score.mean() > 0.5:
            input_ids = candidate
        else:
            true_token = torch.argmax(main_logits[:, -1], dim=-1)
            input_ids = torch.cat(
                [input_ids, true_token.unsqueeze(1)], dim=1
            )

    return input_ids
