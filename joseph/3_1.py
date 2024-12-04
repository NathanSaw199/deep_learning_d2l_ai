# %matplotlib inline
import math
import time
import numpy as np
import torch
from d2l import torch as d2l



n = 10000
a = torch.ones(n)
b = torch.ones(n)

print(a)
print(b)

c = torch.zeros(n)
print(len(c))

t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{time.time() - t:.5f} sec')

t = time.time()
d = a + b
print(f'{time.time() - t:.5f} sec')


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    normal_dis = p * np.exp(-1/2*(x - mu)**2/sigma**2)
    return normal_dis

# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])

d2l.plt.show()