import torch
from .transformer import Transformer
from .paged_kv import KVState, PageAllocator, PAGE_SIZE, MAX_PAGES

@torch.no_grad()
def speculative_decode(target:Transformer,draft:Transformer,input_ids:torch.Tensor,max_new_tokens:int,k:int=4,temperature:float=1.0):
    device=input_ids.device
    target.eval(); draft.eval()
    seq=input_ids
    t_state,d_state=KVState(),KVState()
    K_pool_t=V_pool_t=torch.zeros(MAX_PAGES,16,PAGE_SIZE,128,device=device)
    K_pool_d=V_pool_d=torch.zeros(MAX_PAGES,16,PAGE_SIZE,128,device=device)
    allocator_t=PageAllocator(MAX_PAGES)
    allocator_d=PageAllocator(MAX_PAGES)
    while seq.size(1)<max_new_tokens:
        draft_tokens=[]
        for _ in range(k):
            logits=draft(seq,state=d_state,K_pool=K_pool_d,V_pool=V_pool_d,allocator=allocator_d)
            next_tok=torch.argmax(logits[:,-1],dim=-1,keepdim=True)
            draft_tokens.append(next_tok)
            seq=torch.cat([seq,next_tok],dim=1)
        draft_block=torch.cat(draft_tokens,dim=1)
        logits=target(draft_block,state=t_state,K_pool=K_pool_t,V_pool=V_pool_t,allocator=allocator_t)
        target_preds=torch.argmax(logits,dim=-1)
        accept=0
        for i in range(draft_block.size(1)):
            if target_preds[0,i].item()==draft_block[0,i].item(): accept+=1
            else: break
        if accept>0: seq=torch.cat([seq,draft_block[:,:accept]],dim=1)
    return seq
