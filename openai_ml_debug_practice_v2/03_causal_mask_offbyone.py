# 03_causal_mask_offbyone.py
# Bug: mask blocks attending to self.
import torch, math, torch.nn.functional as F

def attn_buggy(q, k, v):
    B, H, T, D = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()  # BUG
    scores = scores.masked_fill(mask, float("-inf"))
    p = F.softmax(scores, dim=-1)
    return p @ v

def attn_ref(q, k, v):
    B, H, T, D = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    p = F.softmax(scores, dim=-1)
    return p @ v

def _run_tests():
    torch.manual_seed(0)
    B,H,T,D = 2,3,6,4
    q = torch.randn(B,H,T,D); k = torch.randn_like(q); v = torch.randn_like(q)
    yb = attn_buggy(q,k,v)
    yr = attn_ref(q,k,v)
    diff = (yb-yr).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff < 1e-6, "Causal mask off‑by‑one."

    print("✅ Exercise 3 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
