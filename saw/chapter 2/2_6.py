import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

num_tosses = 100
#heads is the number of heads in num_tosses coin tosses > 0.5 is heads and < 0.5 is tails
# 0.5 is the probability of getting heads
# 0.5 means 50% chance of getting heads
heads = sum([random.random()>0.5 for _ in range(num_tosses)])
#nt = n-nh
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
# tensor.tensor 0.5 means 50% chance of getting heads
fair_probs = torch.tensor([0.5, 0.5])
#multinomial is a distribution that returns the number of heads and tails in num_tosses coin tosses
print(Multinomial(100, fair_probs).sample())
print(Multinomial(100, fair_probs).sample()/100)

counts = Multinomial(10000, fair_probs).sample()
# print(counts/10000)
# counts is a tensor of size 2 which contains the number of heads and tails in 10000 coin tosses

counts = Multinomial(1, fair_probs).sample((10,))
# print(counts)
cum_counts = counts.cumsum(dim=0)
# print(cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates)
estimates = estimates.numpy()
print(estimates)