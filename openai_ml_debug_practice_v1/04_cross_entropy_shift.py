# 04_cross_entropy_shift.py
# Goal: Fix next-token language modeling loss.
# Bugs:
# - Not shifting logits/targets (predict token t+1 with token t).
# - Using probabilities with CrossEntropyLoss.
# - Ignoring padding mask when present.
# - Flattening dimensions incorrectly.

import torch
import torch.nn as nn
import torch.nn.functional as F

def lm_loss_buggy(logits, targets, pad_id=None):
    """
    logits: [B, T, V]
    targets: [B, T] (token ids)
    """
    # BUG
    # - no shift
    # - applies softmax before CE
    # probs = F.softmax(logits, dim=-1)  # BUG
    # loss = F.cross_entropy(probs.view(-1, probs.size(-1)), targets.view(-1), ignore_index=pad_id if pad_id is not None else -100)
    loss = F.cross_entropy(logits[:,:-1].transpose(-1,-2), targets[:,1:], ignore_index=pad_id if pad_id is not None else -100)
    return loss

def lm_loss_correct(logits, targets, pad_id=None):
    # shift for next-token prediction: predict targets[:, 1:] from logits[:, :-1]
    logits_shifted = logits[:, :-1, :]
    targets_shifted = targets[:, 1:]
    loss = F.cross_entropy(
        logits_shifted.reshape(-1, logits.size(-1)),
        targets_shifted.reshape(-1),
        ignore_index=pad_id if pad_id is not None else -100
    )
    return loss

def _run_tests():
    torch.manual_seed(0)
    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V)
    targets = torch.randint(low=0, high=V, size=(B, T))

    # Basic: loss must equal correct implementation after you fix lm_loss_buggy()
    loss_buggy = lm_loss_buggy(logits, targets, pad_id=None)
    loss_ref = lm_loss_correct(logits, targets, pad_id=None)
    diff = abs(loss_buggy.item() - loss_ref.item())
    print(f"Loss diff (BUGGY vs REF): {diff:.6f}")
    assert diff < 1e-6, "Language modeling loss is wrong. Fix lm_loss_buggy()."

    # With padding: masked tokens should be ignored
    pad_id = V - 1
    targets_pad = targets.clone()
    targets_pad[:, -2:] = pad_id  # pad last two tokens
    loss_buggy2 = lm_loss_buggy(logits, targets_pad, pad_id=pad_id)
    loss_ref2 = lm_loss_correct(logits, targets_pad, pad_id=pad_id)
    diff2 = abs(loss_buggy2.item() - loss_ref2.item())
    print(f"Loss diff w/ pad (BUGGY vs REF): {diff2:.6f}")
    assert diff2 < 1e-6, "Padding handling is wrong. Fix lm_loss_buggy()."

    print("✅ Exercise 4 tests passed. (Once you've fixed the bugs!)")

if __name__ == "__main__":
    _run_tests()
