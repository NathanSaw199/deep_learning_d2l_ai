# 2.5 Automatic Diferentiation

import torch

#  Can also create x = torch.arange(4.0, requires_grad=True)
x = torch.arange(4.0)

print(f"fist x {x}")

print(x.requires_grad_(True))
print(f"first x.grad {x.grad}")  # The gradient is None by default


y = 2 * torch.dot(x, x)



print(f"first y = 2 * torch.dot(x, x){y}")

# y = 2xTx with respect to x should be 4x
print(y.backward())
print(x.grad)

print(f" x.grad == 4 * x{x.grad == 4 * x}")  # The gradient of 2x^2 with respect to x is 4x


print(f"This is x.grad.zero: {x.grad.zero_()}")

y = x.sum()

print(f"This is y.backward: {y.backward()}")

print(f"x.grad: {x.grad}")


# --------------------------------------------

print(f"x.grad.zero: {x.grad.zero_()}")

y = x * x

print(f"This is y: {y}")

print(f"This is y.backward: {y.backward(gradient=torch.ones(len(y)))}")

print(f"last x.grad: {x.grad}")

# # --------------------------------------------

# 2.5.3 Detaching Computation

print(x.grad.zero_())

y = x * x

print(y)

u = y.detach()

print(u)

z = u * x

print(z)

print(z.sum().backward())

print(x.grad == u)

print(x.grad)

x.grad.zero_()
print(y.sum().backward())

print(x.grad == 2 * x)

# --------------------------------------------




