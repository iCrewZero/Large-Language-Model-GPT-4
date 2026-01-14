import torch

@torch.no_grad()
def speculative_decode(
    target,
    draft,
    input_ids,
    max_new_tokens,
    k=4,
    temperature=1.0
):
    device = input_ids.device
    target.eval()
    draft.eval()

    seq = input_ids
    t_kv = None
    d_kv = None

    while seq.size(1) < max_new_tokens:
        draft_tokens = []
        cur = seq
        d_kv_step = d_kv
        for _ in range(k):
            logits, d_kv_step = draft(cur[:, -1:], kv_cache=d_kv_step)
            next_tok = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            draft_tokens.append(next_tok)
            cur = torch.cat([cur, next_tok], dim=1)
          
        draft_block = torch.cat(draft_tokens, dim=1)
        logits, new_t_kv = target(draft_block, kv_cache=t_kv)
        target_preds = torch.argmax(logits, dim=-1)
        accept = 0
      
        for i in range(draft_block.size(1)):
            if target_preds[:, i].item() == draft_block[:, i].item():
                accept += 1
            else:
                break

        if accept > 0:
            seq = torch.cat([seq, draft_block[:, :accept]], dim=1)
            t_kv = truncate_kv(new_t_kv, seq.size(1))
        else:
            probs = torch.softmax(logits[:, 0] / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
            seq = torch.cat([seq, tok], dim=1)
            t_kv = truncate_kv(new_t_kv, seq.size(1))

        if seq.size(1) >= max_new_tokens:
            break

    return seq
