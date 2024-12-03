import torch


x = torch.arange(4.0)
print(x)
print(x.requires_grad_(True))
print("x grad", x.grad)

y = 2 * torch.dot(x, x)
print(y)
#y = 2xTx with respect to x should be 4x
# there are 4 elements in x vector so the gradient should be 4x
print("y backword: ",y.backward())
print("xgrad :", x.grad)

print(x.grad == 4*x)
print(x.grad.zero_())
# y = x.sum() is equivalent to y = x[0] + x[1] + x[2] + x[3]
y = x.sum()
print(y)
# y
print("y backword: ",y.backward())
print("xgrad :", x.grad)
print(x.grad.zero_())
print("X value : ", x)
y = x*x
# z = torch.arange(4.0)
# print(z)
print(y.backward(torch.ones(len(y))))
print("xgrad :", x.grad)
# print(torch.ones(len(z)))
x.grad.zero_()
print("X : ",x.grad)
y = x*x
print("Y : ",y)
u = y.detach()
print("U :", u)
#z = x* x* x
# u = [0., 1., 4., 9.]
# x = [0., 1., 2., 3.]
z = u*x
print("Z :", z)
print(z.sum().backward())
print(x.grad)
print(x.grad == u)
print(x.grad.zero_())
print("xgrad before baclward: ",x.grad)
print(y)
print("y sum: ", y.sum().backward())
# y = x*x so dy/dx = 2x
print("xgrad after backward", x.grad)
print("X :",x)
print("x.grad: ",x.grad ==2*x)