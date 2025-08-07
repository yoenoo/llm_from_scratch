# 10_device_dtype.py
import torch

def rms_buggy(x,eps=1e-5):
    rms=torch.sqrt((x*x).mean(dim=-1,keepdim=True)+eps)  # in bf16
    return x/rms

def rms_ref(x,eps=1e-5):
    x32=x.float()
    rms=torch.sqrt((x32*x32).mean(dim=-1,keepdim=True)+eps)
    return (x32/rms).to(x.dtype)

def _run_tests():
    torch.manual_seed(0)
    x=torch.randn(6,8,32,dtype=torch.bfloat16)
    diff=(rms_buggy(x).float()-rms_ref(x).float()).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff<1e-5, "Upcast reductions to fp32, return same dtype."

    print("✅ Exercise 10 passed (after fix).")

if __name__=="__main__":
    _run_tests()
