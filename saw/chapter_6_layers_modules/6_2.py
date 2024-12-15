import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))
#torch.rand(size=(2, 4)): Generates a random tensor of shape (2, 4), meaning it has 2 rows and 4 columns. This represents 2 samples with 4 features each.
X = torch.rand(size=(2, 4))
#net(X): Passes the input tensor X through the network net..shape: Prints the shape of the output tensor, which is (2, 1). This means the network outputs 1 feature for each of the 2 input samples.
print(net(X).shape)

#net[2]: Accesses the third component of the sequential model, which is the second LazyLinear layer.
# .state_dict(): Returns a Python dictionary object that maps each layer to its parameter tensor. Here, it shows the weights and biases initialized for the last linear layer, showing an array for the weights and a single value for the bias.
print(net[2].state_dict())

print(type(net[2].bias), net[2].bias.data)

net[2].weight.grad == None

[('0.weight', torch.Size([8, 4])),
 ('0.bias', torch.Size([8])),
 ('2.weight', torch.Size([1, 8])),
 ('2.bias', torch.Size([1]))]


# We need to give the shared layer a name so that we can refer to its
# parameters

#This line creates a LazyLinear layer named shared with 8 output features. The "lazy" aspect means that this layer does not initialize its weights and biases until it sees the first input, and it does not need the number of input features specified explicitly.
shared = nn.LazyLinear(8)

#the network is defined with several layers. Notably, the shared layer is used twice. This means both instances of shared in the network actually reference the same object; they share the same weights and biases.
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
#Since net[2] and net[4] are both references to the shared layer, this check confirms that their weights are identical. Since they refer to the same object, they must be the same, so this will print True for each element.
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])