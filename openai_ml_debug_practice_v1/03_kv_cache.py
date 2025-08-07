# 03_kv_cache.py
# Goal: Implement incremental decoding attention with a K/V cache.
# Bugs to find:
# - Concatenating along wrong dimension.
# - Not updating the cache in-place / returning wrong present shapes.
# - Using cache during training (should be for inference/incremental only).
# - Off-by-one errors in the time index.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, causal=False):
    B, H, Tq, D = q.shape
    _, _, Tk, _ = k.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    if causal:
        # causal mask based on query/key lengths
        # allow attending only to keys up to current query position
        # This builds a [Tq, Tk] mask where positions j>i are -inf.
        max_off = Tk - Tq
        mask = torch.triu(torch.ones(Tq, Tk, device=q.device), diagonal=1+max_off).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class MHAWithCache(nn.Module):
    def __init__(self, n_heads, d_model, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads*head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads*head_dim, d_model, bias=False)

    def _split(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        return x

    def forward(self, x, *, cache=None, incremental=False):
        """
        x: [B, T, d_model]
        cache: dict with "k": [B, H, T_cache, D], "v": [B, H, T_cache, D]
        incremental: if True, we assume T==1 (one token at a time) and we append to cache.
        """
        B, T, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        if incremental:
            assert T == 1, "Incremental mode expects T==1."

            if cache is not None:
                k_prev = cache["k"]
                v_prev = cache["v"]
                # BUG
                k_cat = torch.cat([k_prev, k], dim=-2)  # BUG
                v_cat = torch.cat([v_prev, v], dim=-2)  # BUG
            else:
                k_cat, v_cat = k, v

            present = {"k": k_cat, "v": v_cat}
            out = scaled_dot_product(q, k_cat, v_cat, causal=True)
        else:
            present = None
            out = scaled_dot_product(q, k, v, causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), present

def _run_tests():
    torch.manual_seed(0)
    B, T, nH, D = 2, 6, 3, 12
    head_dim = D // nH
    x = torch.randn(B, T, D)
    mha = MHAWithCache(nH, D, head_dim)

    # Full pass (no cache)
    out_full, _ = mha(x, incremental=False)

    # Incremental pass, one token at a time with cache
    cache = None
    outs = []
    for t in range(T):
        y, cache = mha(x[:, t:t+1, :], cache=cache, incremental=True)
        outs.append(y)
    out_inc = torch.cat(outs, dim=1)

    diff = (out_full - out_inc).abs().max().item()
    print(f"Max diff full vs incremental (BUGGY): {diff:.6f}")
    assert diff < 1e-6, "KV-cache incremental decoding does not match full attention. Fix concatenation/logic."

    print("✅ Exercise 3 tests passed. (Once you've fixed the bugs!)")

if __name__ == "__main__":
    _run_tests()
