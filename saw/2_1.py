
import torch
x = torch.arange(12, dtype=torch.float32, device='cuda')
print(x)
# x =x.numel()
print(x)
# x = x.shape
# print(f"tensor size : {x}")
X = x.reshape(3, 4)
print(X)
