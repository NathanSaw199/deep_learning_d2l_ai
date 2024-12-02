
import torch

# 1D tensor of 12 elements
x = torch.arange(12, dtype=torch.float32)

# print(x)

# Output: 
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
# Each of the values in the output is called an element of the tensor.

# --------------------------------------------------------
# # We can inspect the total number of elements in a tensor via its <numel> method
# # can't use the x.numel with x.shape at the same time
# x = x.numel()

# print(x)

# # Output: 12

# --------------------------------------------------------

# # can't use the x.numel with x.shape at the same time

# y = x.shape

# print(y)

# # Output: torch.Size([12])

# --------------------------------------------------------

# # # matrix of 3 rows and 4 columns
# # # We can reshape the 12 elements into a matrix of 3 rows and 4 columns which is a 2D tensor of size 3x4
# X = x.reshape(3, 4)

# print(X)

# # # 3D tensor of size 2x3x4 ( 2 stands for the number of matrices, 3 stands for the number of rows, and 4 stands for the number of columns)
# # # torch.zeros creates a tensor of the specified size with all elements set to 0 
# x = torch.zeros((2, 3, 4))

# print(x)

# # 3D tensor of size 2x3x4
# # torch.ones creates a tensor of the specified size with all elements set to 1
# x = torch.ones((2, 3, 4))

# # print(x)

# --------------------------------------------------------
# # torch.randn creates a tensor of the specified size with all elements set randomly
# # 2D 3x4 matrix with random elements
# x = torch.randn(3, 4)

# print(x)

# # 3D 2x3x4 tensor with random elements
# x = torch.randn(2, 3, 4)

# print(x)

# torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# print(y)

# --------------------------------------------------------

# print(f"Before Slicing: {x}")

# # Slicing the last element of the tensor x[-1]
# # Select the second and third rows of x x[1:3]
# print(f"After Slicing: {x[-1], x[1:3]}")

z = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# Using alternating indices [0,2] to select first and last rows
print(f"\n {z[[0,2], :]}")



# --------------------------------------------------------

# # replace the element at row 1, column 2, with 17
# X[1,2] = 17

# print (f"Matrices after replacing: {x}")

# # replace the second row with a vector
# print(f"Orginal Matrices: {X}")
# X[:2, :] = 12

# print(X)

# #  replace the second row with a vector
# print(f"Orginal Matrices: {X}")
# X[:2, :2 ] = 12

# print(X)

# OPERATIONS
#--------------------------------------------------------

# # 1D tensor of 12 elements
# x = torch.arange(12, dtype=torch.float32)

# # matrix of 3 rows and 4 columns
# # 2D tensor of size 3x4
# X = x.reshape(3, 4)

# X[:2, : ] = 12
#--------------------------------------------------------

# print(torch.exp(X))

#--------------------------------------------------------

# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])

# print(f"{x+y},\n {x-y},\n {x*y},\n {x/y},\n {x**y},\n")

#--------------------------------------------------------

# # torch.cat concatenates along the specified dimension
# x = torch.arange(12, dtype=torch.float32).reshape((3, 4))

# # 3x4 matrix with random elements
# y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# print(f"{torch.cat((x, y), dim=0),}\n {torch.cat((x, y), dim=1)}")

# print(f"{x == y}")

# print(f"Totals of element in x: {x.sum()}")

# print(f"Totals of elements in y: {y.sum()}")

# --------------------------------------------------------

# # 5 at the front mean vertical
# a = torch.arange(5).reshape((5, 1))

# # 5 at the back mean horizontal
# b = torch.arange(5).reshape((1, 5))

# # print(f"{a},\n {b}")

# print(f"{a+b}")

# --------------------------------------------------------

# saving memory 2.1.5

# X = torch.arange(12, dtype=torch.float32).reshape((3, 4))

# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# before = id(Y)
# Y = Y + X
# print(id(Y) == before)

# --------------------------------------------------------
# Z = torch.zeros_like(Y)
# print('id(Z):', id(Z))
# Z[:] = X + Y
# print('id(Z):', id(Z))

# --------------------------------------------------------


# before = id(X)
# X += Y
# print(id(X) == before)


# --------------------------------------------------------

# A = X.numpy()
# B = torch.from_numpy(A)
# print(type(A), type(B))

# a = torch.tensor([3.5])
# print(a, a.item(), float(a), int(a))


# --------------------------------------------------------

# # Exercise 

# x = torch.arange(12, dtype=torch.float32).reshape((3, 4))

# # 3x4 matrix with random elements
# y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# print(x<y)

# print(x>y)

# --------------------------------------------------------

# a = torch.arange(1, 6).reshape((5, 1))
# b = torch.arange(1, 3).reshape((1, 2))
# print(a, b)
# print(a + b)


