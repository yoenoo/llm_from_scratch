# 01_block_wo_double.py
# Bug: Output projection W_o is applied twice.
import math, torch, torch.nn as nn, torch.nn.functional as F

class TinyMHA(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.o(y)        # correct application
        return y

def _ref_forward(model, x):
    with torch.no_grad():
        B, T, D = x.shape
        q = model.q(x).view(B, T, model.n_heads, model.head_dim).transpose(1, 2)
        k = model.k(x).view(B, T, model.n_heads, model.head_dim).transpose(1, 2)
        v = model.v(x).view(B, T, model.n_heads, model.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(model.head_dim)
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = model.o(y)
        return y

def _run_tests():
    torch.manual_seed(0)
    m = TinyMHA()
    x = torch.randn(2, 5, 32)
    y_bug = m(x)
    y_ref = _ref_forward(m, x)
    err = (y_bug - y_ref).abs().max().item()
    print(f"Max abs err: {err:.6f}")
    assert err < 1e-6, "Apply W_o exactly once."

    print("✅ Exercise 1 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
