# 09_sampling_logits_filter.py
# Bug: temperature & top-k applied on probabilities.
import torch

def prepare_logits_buggy(logits,temperature=1.0,top_k=None):
    probs=torch.softmax(logits,dim=-1)
    probs=probs/temperature           # BUG
    if top_k is not None:
        topk_vals,topk_idx=torch.topk(probs,k=top_k,dim=-1)
        mask=torch.ones_like(probs)
        mask.scatter_(dim=-1,index=topk_idx, src=torch.zeros_like(topk_vals))
        probs=probs*(1-mask)
        probs=probs/probs.sum(dim=-1,keepdim=True).clamp_min(1e-12)
    return probs

def prepare_logits_ref(logits,temperature=1.0,top_k=None):
    logits=logits/temperature
    if top_k is not None:
        top_vals,top_idx=torch.topk(logits,k=top_k,dim=-1)
        cutoff=top_vals[...,-1:].clone()
        logits=logits.masked_fill(logits<cutoff,float('-inf'))
    return torch.softmax(logits,dim=-1)

def _run_tests():
    torch.manual_seed(0)
    b,v=3,12
    logits=torch.randn(b,v)
    for T in [0.7,1.0,1.3]:
        for k in [None,3,5]:
            diff=(prepare_logits_buggy(logits,T,k)-prepare_logits_ref(logits,T,k)).abs().max().item()
            print(f"T={T},k={k},diff={diff:.6f}")
            assert diff<1e-6, "Apply temperature/top‑k on logits then softmax."

    print("✅ Exercise 9 passed (after fix).")

if __name__=="__main__":
    _run_tests()
