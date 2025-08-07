# 11_namespace_typo.py
import torch as T
import torch.nn as nn

def build_linear_buggy(inp,out):
    layer=ln.Linear(inp,out,bias=False)  # BUG alias typo
    with T.no_grad():
        layer.weight.copy_(T.eye(out,inp))
    return layer

def _run_tests():
    ok=True
    try:
        m=build_linear_buggy(4,3)
        y=m(T.randn(2,4))
    except Exception as e:
        print("Exception:",e)
        ok=False
    assert ok, "Fix namespace alias so it runs."
    print("✅ Exercise 11 passed (after fix).")

if __name__=="__main__":
    _run_tests()
