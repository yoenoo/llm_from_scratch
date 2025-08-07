# 02_pos_embed_empty.py
# Bug: Positional encoding created with torch.empty (uninitialized memory).
import torch, math

def build_positional_buggy(seq_len, d_model, device=None, dtype=torch.float32):
    # returns uninitialized tensor -> garbage values
    return torch.empty(seq_len, d_model, device=device, dtype=dtype)  # BUG

def build_positional_ref(seq_len, d_model, device=None, dtype=torch.float32, theta=10000.0):
    assert d_model % 2 == 0
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)[None, :]
    angle = pos / (theta ** (i / d_model))
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe.to(dtype)

def _run_tests():
    L, D = 10, 16
    pb = build_positional_buggy(L, D)
    pr = build_positional_ref(L, D)
    err = (pb - pr).abs().nan_to_num().mean().item()
    print(f"Mean abs err: {err:.6f}")
    assert err < 1e-6, "Initialize positional embeddings properly."

    print("✅ Exercise 2 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
