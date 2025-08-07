# 08_training_loop_modes.py
# Bug: model left in eval mode; no backward/step/zero_grad.
import torch, torch.nn as nn, torch.nn.functional as F

class Toy(nn.Module):
    def __init__(self,D=10):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(D,32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_buggy(model,x,y,steps=80,lr=1e-2):
    opt=torch.optim.SGD(model.parameters(),lr=lr)
    losses=[]
    for _ in range(steps):
        model.eval()  # BUG
        out=model(x)
        loss=F.mse_loss(out,y)
        losses.append(loss.item())  # BUG
    return losses

def train_ref(model,x,y,steps=80,lr=1e-2):
    opt=torch.optim.SGD(model.parameters(),lr=lr)
    losses=[]
    for _ in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(x),y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def _run_tests():
    torch.manual_seed(0)
    N,D=128,10
    X=torch.randn(N,D)
    y=(X@torch.randn(D,1)).squeeze()+0.1*torch.randn(N)
    ref=train_ref(Toy(D),X,y)
    bug=train_buggy(Toy(D),X,y)
    print(f"Loss ref: {ref[0]:.3f}->{ref[-1]:.3f}, bug: {bug[0]:.3f}->{bug[-1]:.3f}")
    assert ref[-1] < ref[0]*0.4, "Reference learns."
    assert bug[-1] < bug[0]*0.95, "Fix train_buggy to learn similarly."

    print("✅ Exercise 8 passed (after fix).")

if __name__ == "__main__":
    _run_tests()
