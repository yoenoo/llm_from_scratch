# mini_transformer_debug.py
"""
Interview-style debugging exercise: **one file, many hidden bugs**.

Your task:
1. Open this file.
2. Fix *every* line marked with `# BUG`.
3. Run: `python mini_transformer_debug.py` and make all tests pass.

No additional libraries required beyond PyTorch (CPU is fine).
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
from einops import einsum, rearrange

def _assert_close(a, b, tol=1e-6, msg=""):
    err = (a - b).abs().max().item()
    assert err < tol, f"{msg} | max diff {err:.6f}"

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        # denom = x.mean(dim=-1, keepdim=True) + self.eps  # BUG
        # y = (x / denom) * self.weight  # BUG
        # return y  # BUG

        input_dtype = x.dtype
        x = x.float() # upcast to float32
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight) # no need to downcast?

def build_sinusoidal_positions(seq_len, d_model, theta=10000.0, device=None, dtype=torch.float32):
    # return torch.empty(seq_len, d_model, device=device, dtype=dtype)  # BUG
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2, device=device, dtype=dtype) / d_model))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    angles = einsum(pos, inv_freq, "n,d -> n d")
    cos = angles.cos()
    sin = angles.sin()
    
    out = torch.empty((seq_len, d_model))
    out[:,::2] = sin; out[:,1::2] = cos
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.q=nn.Linear(d_model,d_model,bias=False)
        self.k=nn.Linear(d_model,d_model,bias=False)
        self.v=nn.Linear(d_model,d_model,bias=False)
        self.o=nn.Linear(d_model,d_model,bias=False)
    def _split(self,x):
        B,T,_=x.shape
        return x.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
    def forward(self,x,cache=None,incremental=False):
        B,T,_=x.shape
        q=self._split(self.q(x)) # (batch_size, n_heads, seq_len, head_dim)
        k=self._split(self.k(x))
        v=self._split(self.v(x))

        if incremental:
            if cache is not None:
                # cache["k"] = torch.cat((cache["k"], k), dim=2)
                # cache["v"] = torch.cat((cache["v"], v), dim=2)
                
                # k = cache["k"]
                # v = cache["v"]

                k=torch.cat([cache["k"],k],dim=2)  # BUG
                v=torch.cat([cache["v"],v],dim=2)  # BUG
        
        scores=(q @ k.transpose(-2,-1))/math.sqrt(k.shape[-1])  # BUG
        att=F.softmax(scores,dim=-1)  # BUG
        y=att @ v
        y=y.transpose(1,2).contiguous().view(B,T,self.d_model)
        y=self.o(y)
        return y,{"k":k,"v":v}

class TransformerLM(nn.Module):
    def __init__(self,vocab=60,d_model=32,n_heads=4,context=64):
        super().__init__()
        self.emb=nn.Embedding(vocab,d_model)
        self.pos=nn.Parameter(build_sinusoidal_positions(context,d_model))
        self.mha=MultiHeadAttention(d_model,n_heads)
        self.norm=RMSNorm(d_model)
        self.lm=nn.Linear(d_model,vocab,bias=False)
    def forward(self,toks,cache=None,incremental=False):
        pos_emb=self.pos[:toks.size(1)]
        x=self.emb(toks)+pos_emb
        y,cache=self.mha(x,cache=cache,incremental=incremental)
        y=self.norm(y)
        logits=self.lm(y)
        return logits,cache

def lm_loss(logits,targets):
    probs=torch.softmax(logits,dim=-1)  # BUG
    return F.cross_entropy(probs.view(-1,probs.size(-1)),targets.view(-1))

def generate(model,prefix,steps):
    toks=prefix.clone(); cache=None
    for _ in range(steps):
        logits,cache=model(toks[-1],cache=cache,incremental=True)  # BUG
        next_id=logits[:,-1].argmax(dim=-1,keepdim=True)
        toks=torch.cat([toks,next_id],dim=1)
    return toks

# ---- reference helpers for tests
def _build_sin_ref(L,D,theta=10000.0):
    assert D%2==0
    pos=torch.arange(L,dtype=torch.float32)[:,None]
    i=torch.arange(0,D,2,dtype=torch.float32)[None,:]
    angle=pos/(theta**(i/D))
    pe=torch.zeros(L,D)
    pe[:,0::2]=torch.sin(angle);pe[:,1::2]=torch.cos(angle)
    return pe

def _test_rms():
    torch.manual_seed(0)
    x=torch.randn(3,5,16,dtype=torch.bfloat16)
    m=RMSNorm(16).to(torch.bfloat16)
    ref=_build_rms_ref(x,m.eps,m.weight)
    _assert_close(m(x).float(),ref.float(),1e-5,"RMSNorm")

def _build_rms_ref(x,eps,w):
    x32=x.float()
    rms=torch.sqrt((x32*x32).mean(-1,keepdim=True)+eps)
    return (x32/rms)*w.float()

def _test_pos():
    _assert_close(build_sinusoidal_positions(12,8),_build_sin_ref(12,8),1e-6,"Positional")

def _test_attn():
    torch.manual_seed(0)
    B,T,D,H=1,6,32,4
    x=torch.randn(B,T,D)
    m=MultiHeadAttention(D,H)
    full,_=m(x)
    cache=None; outs=[]
    for t in range(T):
        o,cache=m(x[:,t:t+1],cache=cache,incremental=True)
        outs.append(o)
    inc=torch.cat(outs,1)
    _assert_close(full,inc,1e-6,"KV cache")

def _test_loss():
    B,T,V=2,7,30
    logits=torch.randn(B,T,V); targets=torch.randint(0,V,(B,T))
    l1=lm_loss(logits,targets)
    ref=F.cross_entropy(logits[:,:-1].reshape(-1,V),targets[:,1:].reshape(-1))
    _assert_close(l1,ref,1e-6,"CrossEntropy shift")

def run_tests():
    _test_rms(); _test_pos(); _test_attn(); _test_loss()
    print("All tests passed!")

if __name__=="__main__":
    run_tests()
