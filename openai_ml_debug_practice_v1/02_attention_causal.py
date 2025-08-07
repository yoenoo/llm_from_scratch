# 02_attention_causal.py
# Goal: Fix scaled dot-product attention with optional causal masking.
# Common bugs:
# - Wrong scaling factor (should be sqrt(d_k), not sqrt(d_model) or a seq dimension).
# - Applying mask AFTER softmax instead of BEFORE with -inf fill.
# - Softmax over the wrong dimension.
# - Using an upper-triangular mask in the wrong direction.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, causal=False):
    """
    q, k, v: [B, H, T, D]
    returns: [B, H, T, D]
    """
    B, H, T, D = q.shape
    # BUG
    # - Wrong scale: divides by sqrt(T) instead of sqrt(D)
    # - Applies mask AFTER softmax
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # BUG

    if causal:
        # causal mask: disallow attending to future positions j > i
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, -torch.inf)  # BUG
    
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

def _run_tests():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 5, 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Test 1: shape
    out = attention(q, k, v, causal=False)
    assert out.shape == (B, H, T, D), "Output shape is wrong."

    # Test 2: numerical sanity (no mask): if k==q and v is identity-like, attention should prefer self
    q2 = torch.eye(T).repeat(B, H, 1, 1)[:, :, :, :D]
    k2 = q2.clone()
    v2 = torch.arange(T * D).float().view(1, 1, T, D).repeat(B, H, 1, 1)
    out2 = attention(q2, k2, v2, causal=False)
    print(out2)
    diag_vals = out2[0, 0, torch.arange(T), :].detach()
    off_vals = out2[0, 0].mean(dim=0)
    assert diag_vals.mean() > off_vals.mean(), "Scaling/softmax seems off (diagonal not dominant)."

    # Test 3: causal masking enforces no future attention
    out3 = attention(q, k, v, causal=True)
    B0, H0 = 0, 0
    T0 = T
    # Recompute scores to infer mask effect
    scores = torch.matmul(q[B0:B0+1, H0:H0+1], k[B0:B0+1, H0:H0+1].transpose(-2, -1)) / math.sqrt(D)
    mask = torch.triu(torch.ones(T0, T0, device=q.device), diagonal=1).bool()
    scores_masked = scores.masked_fill(mask, float("-inf"))
    attn_true = F.softmax(scores_masked, dim=-1)
    out_true = torch.matmul(attn_true, v[B0:B0+1, H0:H0+1])
    diff = (out3[B0:B0+1, H0:H0+1] - out_true).abs().max().item()
    assert diff < 1e-5, "Causal masking/scaling is incorrect. Fix attention()."

    print("✅ Exercise 2 tests passed. (Once you've fixed the bugs!)")

if __name__ == "__main__":
    _run_tests()
