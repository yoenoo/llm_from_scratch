# 05_rope.py
# Goal: Apply rotary position embeddings (RoPE) correctly to q/k.
# Bugs:
# - Wrong pairing of even/odd features.
# - Miscomputed angles (duplicated instead of interleaved).
# - dtype issues when using bf16/fp16.

import torch
import torch.nn as nn
import math

def rope_angles(head_dim, seq_len, theta_base=10000.0, device=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim must be even."
    half = head_dim // 2
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, half]
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin  # each [T, half]

def apply_rope_buggy(x, cos, sin):
    """
    x: [B, H, T, D], cos/sin: [T, D/2]
    """
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    # BUG
    # - Duplicates cos/sin instead of interleaving
    # - Swaps signs incorrectly
    cos_full = torch.cat([cos, cos], dim=-1)  # BUG
    sin_full = torch.cat([sin, sin], dim=-1)  # BUG
    # x_rot = x * cos_full.unsqueeze(0).unsqueeze(0) + x.flip(-1) * sin_full.unsqueeze(0).unsqueeze(0)  # BUG

    x_rotated = torch.cat((-x2, x1), dim=-1)
    x_rot = x * cos_full[None,None,:,:] + x_rotated * sin_full[None,None,:,:]
    return x_rot

def apply_rope_correct(x, cos, sin):
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x2 * cos + x1 * sin
    return torch.cat([x1_rot, x2_rot], dim=-1)

def _run_tests():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 5, 8
    x = torch.randn(B, H, T, D)
    cos, sin = rope_angles(D, T, theta_base=10000.0, device=x.device)
    y_buggy = apply_rope_buggy(x, cos, sin)
    y_ref = apply_rope_correct(x, cos, sin)
    max_err = (y_buggy - y_ref).abs().max().item()
    print(f"Max abs err (BUGGY vs REF): {max_err:.6f}")
    assert max_err < 1e-6, "RoPE application is wrong. Fix apply_rope_buggy()."
    print("✅ Exercise 5 tests passed. (Once you've fixed the bugs!)")

if __name__ == "__main__":
    _run_tests()
