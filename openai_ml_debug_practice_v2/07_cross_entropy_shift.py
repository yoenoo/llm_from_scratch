# 07_cross_entropy_shift.py
import torch, torch.nn.functional as F

def lm_loss_buggy(logits, targets, pad_id=None):
    probs = torch.softmax(logits, dim=-1)  # BUG
    return F.cross_entropy(probs.view(-1, probs.size(-1)),
                           targets.view(-1),
                           ignore_index=pad_id if pad_id is not None else -100)

def lm_loss_ref(logits, targets, pad_id=None):
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1),
                           ignore_index=pad_id if pad_id is not None else -100)

def _run_tests():
    torch.manual_seed(0)
    B,T,V = 2,7,15
    logits = torch.randn(B,T,V)
    targets = torch.randint(0,V,(B,T))
    pad=V-1; targets[:,-2:] = pad
    err = abs(lm_loss_buggy(logits,targets,pad)-lm_loss_ref(logits,targets,pad)).item()
    print(f"Loss diff: {err:.6f}")
    assert err < 1e-6, "Shift logits/targets & use raw logits."

    print("✅ Exercise 7 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
