import torch
import numpy as np
#Prove that the transpose of the transpose of a matrix is the matrix itself: 
# A = torch.tensor([[1,2,3],
#                   [4,5,6],
#                    [7,8,9]])
# print(((A.T).T) == A)
#Given two matrices show that sum and transposition commute:
# A = torch.tensor([[1,2,3],
#                   [4,5,6],
#                   [7,8,9]])
# B = torch.tensor([[0,2,3],
#                   [4,5,6],
#                   [7,8,9]])
# print(A.T+B.T == (A+B).T)
#Given any square matrix A, is A+A⊤ always symmetric? Why?
# A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
# print(A)
# print(A == A+A.T)
#We defined the tensor X of shape (2, 3, 4) in this section (2.3). What is the output of len(X)?
# x= torch.tensor([2, 3, 4])
# print(len(x))
#Run A / A.sum(axis=1) and see what happens. Can you analyze the results?
# A = torch.arange(9).reshape(3,3)
# print(A)
# sum_A = A.sum(axis=1, keepdims=True)
# print(sum_A)
# print(A/sum_A)
# [3.0, -4.0] = 9 + 16 = 25
# [2, 2] = 4 + 4 = 8
# 25 + 8 = 33
#square root of 33 is 5.74
# u = np.array([[3.0, -4.0],[2,2]])
# print(np.linalg.norm(u))

