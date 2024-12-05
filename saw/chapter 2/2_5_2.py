import torch

def f(a):
    b = a * 2
    # b.norm() is equivalent to l2 norm of b
    # l2 norm of b is sqrt(b[0]^2 + b[1]^2 + b[2]^2 + b[3]^2)
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
print(d)
d.backward()
print(a.grad == d / a)
