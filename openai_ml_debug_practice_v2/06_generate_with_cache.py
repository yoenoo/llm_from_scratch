# 06_generate_with_cache.py
# Bug: generate() ignores cache, leading to O(L^2) work.
import torch, math, torch.nn as nn, torch.nn.functional as F

def sdp(q,k,v):
    return (q @ k.transpose(-2,-1) / math.sqrt(q.size(-1))).softmax(dim=-1) @ v

class CacheMHA(nn.Module):
    def __init__(self, d_model=24, n_heads=3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model=d_model; self.n_heads=n_heads; self.head_dim=d_model//n_heads
        self.q = nn.Linear(d_model,d_model,bias=False)
        self.k = nn.Linear(d_model,d_model,bias=False)
        self.v = nn.Linear(d_model,d_model,bias=False)
        self.o = nn.Linear(d_model,d_model,bias=False)
    def split(self,x):
        B,T,_ = x.shape
        return x.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
    def forward(self,x,cache=None,incremental=False):
        q=self.split(self.q(x))
        k=self.split(self.k(x))
        v=self.split(self.v(x))
        if incremental:
            if cache is not None:
                k=torch.cat([cache['k'],k],dim=-2)
                v=torch.cat([cache['v'],v],dim=-2)
        y=sdp(q,k,v)
        y=y.transpose(1,2).contiguous().view(x.size(0),x.size(1),self.d_model)
        return self.o(y),{'k':k,'v':v}

class TinyLM(nn.Module):
    def __init__(self,vocab=40,d_model=24,n_heads=3):
        super().__init__()
        self.emb=nn.Embedding(vocab,d_model)
        self.mha=CacheMHA(d_model,n_heads)
        self.lm=nn.Linear(d_model,vocab,bias=False)
    def forward(self,tok,cache=None,incremental=False):
        x=self.emb(tok)
        y,cache=self.mha(x,cache=cache,incremental=incremental)
        return self.lm(y),cache

def generate_buggy(model,prefix,steps):
    # always feeds full prefix
    tokens=prefix.clone()
    for _ in range(steps):
        logits,_=model(tokens,cache=None,incremental=False)  # BUG
        next=logits[:,-1].argmax(dim=-1,keepdim=True)
        tokens=torch.cat([tokens,next],1)
    return tokens

def generate_ref(model,prefix,steps):
    tokens=prefix.clone(); cache=None
    for _ in range(steps):
        logits,cache=model(tokens[:,-1:],cache=cache,incremental=True)
        next=logits[:,-1].argmax(dim=-1,keepdim=True)
        tokens=torch.cat([tokens,next],1)
    return tokens

def _run_tests():
    torch.manual_seed(0)
    model=TinyLM()
    prefix=torch.randint(0,40,(1,6))
    bug=generate_buggy(model,prefix,6)
    ref=generate_ref(model,prefix,6)
    assert torch.equal(bug,ref), "Results should match."
    print("✅ Exercise 6 logic fix done once you thread cache through generate().")

if __name__=="__main__":
    _run_tests()
