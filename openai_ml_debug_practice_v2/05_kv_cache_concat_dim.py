# 05_kv_cache_concat_dim.py
# Bug: KV cache concatenated on feature dim instead of time.
import torch, math, torch.nn as nn, torch.nn.functional as F

def sdp(q,k,v):
    return (q @ k.transpose(-2,-1) / math.sqrt(q.size(-1))).softmax(dim=-1) @ v

class MHA(nn.Module):
    def __init__(self, d_model=24, n_heads=3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model,d_model,bias=False)
        self.k = nn.Linear(d_model,d_model,bias=False)
        self.v = nn.Linear(d_model,d_model,bias=False)
        self.o = nn.Linear(d_model,d_model,bias=False)

    def split(self, x):
        B,T,_ = x.shape
        return x.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

    def forward(self,x,cache=None,incremental=False):
        B,T,_ = x.shape
        q = self.split(self.q(x))
        k = self.split(self.k(x))
        v = self.split(self.v(x))
        if incremental:
            if cache is not None:
                k = torch.cat([cache["k"], k], dim=-1)  # BUG
                v = torch.cat([cache["v"], v], dim=-1)  # BUG
            cache = {"k": k, "v": v}
        y = sdp(q,k,v)
        y = y.transpose(1,2).contiguous().view(B,T,self.d_model)
        return self.o(y), cache

def _run_tests():
    torch.manual_seed(0)
    B,T,D,nH = 1,5,24,3
    m = MHA(D,nH)
    x = torch.randn(B,T,D)
    y_full,_ = m(x,incremental=False)
    cache=None; outs=[]
    for t in range(T):
        y,cache = m(x[:,t:t+1],cache=cache,incremental=True)
        outs.append(y)
    y_inc = torch.cat(outs,dim=1)
    diff = (y_full - y_inc).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff < 1e-6, "Concatenate cache along time dim (-2)."

    print("✅ Exercise 5 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
