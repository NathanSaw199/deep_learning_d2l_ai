import torch
from torch import nn
#functional as F: submodule that contains functions like activation functions and other utilities which can be used functionally.
from torch.nn import functional as F
#MLP class inherits from nn.Module, which is a base class for all neural network modules in PyTorch.
class MLP(nn.Module):
    #Inside the constructor (__init__), it initializes two linear layers
    def __init__(self):
        super().__init__()
        #self.hidden: a hidden layer with a LazyLinear module which will determine its input features automatically on the first pass and set the output features to 256.
        self.hidden = nn.LazyLinear(256)
        #self.output: an output layer, also a LazyLinear module, to automatically determine its input features from the previous layer and set output features to 10 (typical for a classification problem with 10 classes).

        self.output = nn.LazyLinear(10)

    def forward(self, x):
        #Data x is passed through the hidden layer, then a ReLU activation function is applied to introduce non-linearity, followed by passing the result through the output layer.
        return self.output(F.relu(self.hidden(x)))

net = MLP()
#X is a batch of 2 random input vectors, each of size 20.
X = torch.randn(size=(2, 20))
#Y stores the output of the network after passing X through it.
Y = net(X)

#The state of the network (net) is saved to a file called mlp.params.
torch.save(net.state_dict(), 'mlp.params')
#A new instance of MLP called clone is created.
clone = MLP()
#The saved state is loaded into clone, ensuring it has the same parameters as net.
clone.load_state_dict(torch.load('mlp.params'))
#clone.eval() sets the model to evaluation mode (important for some types of layers like dropout or batchnorm).
print(clone.eval())

#Y_clone stores the output of clone after passing the same input X.
Y_clone = clone(X)
#The outputs Y_clone and Y are compared element-wise to check if they are identical, which they should be since clone was loaded with the same parameters as net.
print(Y_clone == Y)
