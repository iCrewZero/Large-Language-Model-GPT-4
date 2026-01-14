import torch
import torch.nn as nn
import torch.nn.functional as F
from .paged_kv import append_token_state, write_kv_token, gather_kv, KVState, PageAllocator

class YaRNRoPE:
    def __init__(self, dim, base=10000, factor=8.0):
        self.dim=dim; self.base=base; self.factor=factor
        half=dim//2; idx=torch.arange(half); low=half; high=half*32
        scale=torch.ones(half); scale[idx>high]=factor
        mask=(idx>=low)&(idx<=high); scale[mask]=1+(factor-1)*(idx[mask]-low)/(high-low)
        inv_freq=1.0/(base**(idx/half)); self.inv_freq=(inv_freq*scale).cuda()
    def __call__(self,q,k,pos):
        t=torch.arange(pos,pos+q.size(2),device=q.device)
        freqs=torch.einsum("t,d->td",t,self.inv_freq)
        emb=torch.cat([freqs,freqs],dim=-1); cos,sin=emb.cos(),emb.sin()
        def rotate(x): x1,x2=x[...,:self.dim//2],x[...,self.dim//2:]; return torch.cat([-x2,x1],dim=-1)
        q=q*cos+rotate(q)*sin; k=k*cos+rotate(k)*sin; return q,k

def quantize_k(x):
    scale=x.abs().amax(dim=-1,keepdim=True)/127.0+1e-6
    q=(x/scale).round().clamp(-128,127).to(torch.int8)
    return q,scale
def dequantize_k(q,scale): return q.float()*scale

class CausalSelfAttention(nn.Module):
    def __init__(self,n_head=16,n_kv_head=8,head_dim=128,use_flash=True):
        super().__init__()
        self.n_head=n_head; self.n_kv_head=n_kv_head; self.head_dim=head_dim; self.use_flash=use_flash
        self.q=nn.Linear(head_dim*n_head,head_dim*n_head,bias=False)
        self.kv=nn.Linear(head_dim*n_head,2*n_kv_head*head_dim,bias=False)
        self.o=nn.Linear(head_dim*n_head,head_dim*n_head,bias=False)
        self.rope=YaRNRoPE(head_dim)
    def forward(self,x,state:KVState,K_pool,V_pool,allocator:PageAllocator):
        B,T,C=x.shape
        q=self.q(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        kv=self.kv(x).view(B,T,2,self.n_kv_head,self.head_dim); k,v=kv[:,:,0],kv[:,:,1]
        q,k=self.rope(q,k.transpose(1,2),state.seq_len); k=k.transpose(1,2)
        for t in range(T):
            append_token_state(state,allocator,PAGE_SIZE)
            write_kv_token(K_pool,V_pool,state,k[:,t],v[:,t],PAGE_SIZE)
        K,V=gather_kv(K_pool,V_pool,state,state.seq_len,PAGE_SIZE)
        K=K.repeat_interleave(self.n_head//self.n_kv_head,dim=1)
        V=V.repeat_interleave(self.n_head//self.n_kv_head,dim=1)
        out=F.scaled_dot_product_attention(q,K,V,is_causal=True)
        out=out.transpose(1,2).reshape(B,T,C)
        return self.o(out)
