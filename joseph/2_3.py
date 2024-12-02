import torch


# # 2.3.1. Scalars

# x = torch.tensor(3.0)
# y = torch.tensor(2.0)

# print(x + y, x * y, x / y, x ** y)

# --------------------------------------------

# # 2.3.2. Vectors

# x = torch.arange(3)

# print (x)
# print (x[2])
# print (len(x))
# print (x.shape)

# # --------------------------------------------

# # 2.3.3 Matrices   

# A = torch.arange(6).reshape((3, 2))
# print(A)

# # transpose swap the row and column indices of a matrix
# print (A.T)

# A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])


# print (A == A.T)

# 24 / 2 ( 3 stands for the number of rows, 4 stands for the number of columns) 
# 24 stands for the number of elements in the matrix
# 2 stands for the number of dimensions of the matrix
# print(torch.arange(24).reshape(2, 3, 4))


# --------------------------------------------

# 2.3.5 Basic Properties of Tensor Arithmetic

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
# print(f"{A},\n {A+B}")

# print (A * B)


# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(f"{a + X},\n {(a * X).shape}")

# 2.3.6 Reduction
# x = torch.arange(3, dtype=torch.float32)
# print(f"{x},\n {x.sum()}")

# print (A.shape, A.sum())

# # axis = 0 means that we want to sum along the rows (axis 0) for each column
# # axis = 1 means that we want to sum along the columns (axis 1) for each row
# print( A.shape, A.sum(axis=0).shape)
# print( A.shape, A.sum(axis=1).shape)

# # summing all the elements in the tensor
# print(A.sum(axis=[0,1])== A.sum())
# print(A.sum(axis=[0,1]))
# print(A.sum())

# # A.mean is a method that computes the mean of tensor A
# # A.sum() / A.numel() is the same as A.mean
# print(A.mean(), A.sum() / A.numel())
# print(A.mean)


# print(A)
# axis = 0 we add virtically and devided by the number of elements in the row
# axis = 1 we add horizontally and devided by the number of elements in the column
# print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

# --------------------------------------------
# 2.3.7 Non-Reduction Sum

# # sum horizontally
sum_A = A.sum(axis=1, keepdims=True)
# print(sum_A, sum_A.shape)

# # sum vertically
# sum_A = A.sum(axis=0, keepdims=True)
# print(sum_A, sum_A.shape)


# print(A)
# print(sum_A)

# print(A/sum_A)

# # cumsum function , axis = 0 ( row by row sum)
# # cumsum function , axis = 1 ( column by column sum)
# # cumulative sum of the elements of A along the column (axis 0) ( row by row sum).
# print(A.cumsum(axis=0))

# --------------------------------------------
# 2.3.8 Dot Products

x = torch.arange(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
# print(x,y, torch.dot(x, y))

# print(torch.sum(x * y))

# --------------------------------------------
# # 2.3.9 Matrix-Vector Products  

# # tensor([[0., 1., 2.],   <- y
# #         [3., 4., 5.]])  <- x
# # 
# # multiply y by 0, 1, 2 and add the result = 5
# # multiply x by 3, 4, 5 and add the result = 14

# print(A.shape, x.shape, torch.mv(A, x), A@x)
# print(A)
# print(x)
# print(torch.mv(A, x))
# print(A@x)



# --------------------------------------------
# 2.3.10 Matrix-Matrix Multiplication

# B = torch.ones(3, 4)
# print(A)
# print(B)
# print(f"{torch.mm(A, B)},\n {A@B}")

# --------------------------------------------
# 2.3.11 Norm

# each element in the vector is squared and then summed up
# then the square root of the sum is taken
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))
