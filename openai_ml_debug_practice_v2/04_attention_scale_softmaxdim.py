# 04_attention_scale_softmaxdim.py
import torch, math, torch.nn.functional as F

def attn_buggy(q,k,v):
    B,H,T,D = q.shape
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(T)  # BUG
    p = F.softmax(scores, dim=2)    # BUG
    return p @ v

def attn_ref(q,k,v):
    B,H,T,D = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    p = F.softmax(scores, dim=-1)
    return p @ v

def _run_tests():
    torch.manual_seed(0)
    B,H,T,D = 2,3,5,4
    q = torch.randn(B,H,T,D); k = torch.randn_like(q); v = torch.randn_like(q)
    diff = (attn_buggy(q,k,v) - attn_ref(q,k,v)).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff < 1e-6, "Use sqrt(d_k) and softmax over key axis."

    print("✅ Exercise 4 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
