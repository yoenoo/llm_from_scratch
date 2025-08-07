# 01_rmsnorm.py
# Goal: Fix RMSNorm forward pass for numerical stability and correctness.
# Hints:
# - Upcast to float32 for the reduction.
# - Use RMS = sqrt(mean(a^2)) per feature vector.
# - Apply epsilon inside the sqrt (add to the mean of squares).
# - Broadcast the learnable gain `weight` correctly.
# - Return in the input dtype.

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # denom = x.mean(dim=-1, keepdim=True) + self.eps  # BUG
        # y = (x / denom) * self.weight  # BUG: broadcast happens, but denom is wrong metric
        # return y  # BUG: possibly dtype mismatch

        input_dtype = x.dtype        

        x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).to(input_dtype)

def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    # RMS = sqrt(mean(a^2))
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    y = (x32 / rms) * weight
    return y.to(dtype=x.dtype)

def _run_tests():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 7, 16, dtype=dtype)
        m = RMSNorm(16, eps=1e-5).to(dtype)
        with torch.no_grad():
            # set weight to non-trivial values for a stronger test
            m.weight.copy_(torch.linspace(0.5, 1.5, 16).to(dtype))

        y_buggy = m(x)
        y_ref = reference_rmsnorm(x, m.weight, m.eps)

        # Expect close agreement
        max_err = (y_buggy.float() - y_ref.float()).abs().max().item()
        print(f"[{dtype}] max abs err (BUGGY vs REF): {max_err:.6f}")
        assert max_err < 1e-5, "RMSNorm is incorrect. Fix the forward() implementation."

    print("✅ Exercise 1 tests passed. (Once you've fixed the bugs!)")

if __name__ == "__main__":
    _run_tests()
