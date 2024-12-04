# %matplotlib inline
import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

num_tosses = 100

# 0.5 is the probability of heads
# 1 is the number of trials
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])

# nt = (n -nh)
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])

fair_probs = torch.tensor([0.5, 0.5])

# multinomial is a distribution that takes a vector of probabilities
print(Multinomial(100, fair_probs).sample())

# multinomial is the same as the binomial distribution
print(Multinomial(100, fair_probs).sample() / 100)

# count the number of times each outcome occurs
counts = Multinomial(10000, fair_probs).sample()
print(counts / 10000)


counts = Multinomial(1, fair_probs).sample((10000,))
print(counts)

cum_counts = counts.cumsum(dim=0)
print(cum_counts)

estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates)

estimates = estimates.numpy()
print(estimates)

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend(); 