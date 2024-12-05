
import torch
# assign number of elements to the tensor by 12
x = torch.arange(12, dtype=torch.float32)
# print(x)
#count the number of elements in the tensor

# x =x.numel()
# print(x)
#count the number of elements in the tensor
# x = x.shape
# # print(f"tensor size : {x}")

# # there is a matrix of 3 rows and 4 columns in the tensor
# X = x.reshape(3, 4)
# # print(X)
# # fill up with zeros in the tensor of 2 rows, 3 columns and 4 depth
# y = torch.zeros((2, 3, 4))
# # print(y)
# # fill up with ones in the tensor of 2 rows, 3 columns and 4 depth

# z = torch.ones((2, 3, 4))
# # print(z)
# #sample each element randomly (and independently) from a given probability distribution
# torch.randn(3, 4)
# # print(torch.randn(3, 4))
# # construct tensors by supplying the exact values for each element by supplying (possibly nested) Python list(s) containing numerical literals. 
# torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# #indexing and slicing
# # x = torch.arange(12, dtype=torch.float32)

# # X = x.reshape(3, 4)
# # print(f" before slicing : {X}")
# # print(f"after slicing : {X[0],X[2:4], X[2:3]}")

# #replace the element in the tensor at the index of 1,2 with 17
# x = torch.arange(12, dtype=torch.float32)
# X = x.reshape(3, 4)
# print(f"original matrix : {X}")
# X[2,1] = 17
# print(f"replaced element in the index of the matrix : {z}")
# X[:2,:2] = 12
# X[:2,:] = 12

# print(f"replaced elements in the matrix : {X}")
# print(torch.exp(x))

# X = torch.tensor([1.0, 2, 4, 8])
# Y = torch.tensor([2, 2, 2, 2])
# print(f"X+Y = {X+Y},\n X-Y = {X-Y},\n X*Y = {X*Y},\n X/Y = {X/Y}")

# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# # print(X)
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# Z  = torch.tensor([[1.0, 3, 5, 6], [7,7, 7, 7], [8, 8, 8, 8]])
# print(f"{torch.cat((X,Y),dim=0)},\n{torch.cat((X,Y,Z),dim=1)}")
# print(X == Y)
# print(Y.sum())

# a = torch.arange(7).reshape((7, 1))
# b = torch.arange(7).reshape((1, 7))
# # print(a,b)
# print(a+b)
# print(a-b)

#saving memory
# before = id(Y)
# Y = X
# print(id(Y) == before)
# print(Y)
# Z = torch.zeros_like(Y)
# print(f"Z : {id(Z)}")   
# Z[:] = X + Y
# print('Z:', id(Z))  
# input = torch.empty(2, 3)
# print(torch.zeros_like(input))


# saw = 1
# ko = id(saw)
# print(ko)
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# # print(X)
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# before = id(X)
# X += Y
# print(id(X) == before)

# A = X.numpy()
# B = torch.from_numpy(A)
# print(type(A), type(B))
# sawko = torch.tensor([3.7])
# print(sawko, sawko.item(), float(sawko), int(sawko))
# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# print(X)
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # print(X==Y)
# print(X<Y)
# print(X>Y)
a = torch.arange(1,6).reshape((5, 1))
b = torch.arange(1,4).reshape((1, 3))
print(f"arrange: {a},\n{b}\n")
print(a+b)
# print(a-b)