# 12_rope_offbyone_pairing.py
# Bug: RoPE indexing off-by-one and wrong pairing.
import torch, math

def rope_angles_ref(D,T,theta=10000.0,device=None,dtype=torch.float32):
    half=D//2
    inv=1.0/(theta**(torch.arange(0,half,device=device,dtype=torch.float32)/half))
    t=torch.arange(T,device=device,dtype=torch.float32)
    freqs=torch.outer(t,inv)
    cos=torch.cos(freqs).to(dtype)
    sin=torch.sin(freqs).to(dtype)
    return cos,sin

def rope_angles_buggy(D,T):
    cos,sin=rope_angles_ref(D,T+1)
    return cos[1:],sin[1:]  # BUG drops position 0

def apply_rope_correct(x,cos,sin):
    B,H,T,D=x.shape; half=D//2
    x1,x2=x[...,:half],x[...,half:]
    cos=cos[None,None,:,:].to(x.dtype)
    sin=sin[None,None,:,:].to(x.dtype)
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin],dim=-1)

def apply_rope_buggy(x,cos,sin):
    cos_full=torch.cat([cos,cos],dim=-1)
    sin_full=torch.cat([sin,sin],dim=-1)
    return x*cos_full[None,None,:,:] + torch.flip(x,[-1])*sin_full[None,None,:,:]  # BUG

def _run_tests():
    torch.manual_seed(0)
    B,H,T,D=1,2,6,12
    x=torch.randn(B,H,T,D)
    cosb,sinb=rope_angles_buggy(D,T)
    cosr,sinr=rope_angles_ref(D,T)
    diff=(apply_rope_buggy(x,cosb,sinb)-apply_rope_correct(x,cosr,sinr)).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff<1e-6, "Fix RoPE indexing/pairing."

    print("✅ Exercise 12 passed (after fix).")

if __name__=="__main__":
    _run_tests()
