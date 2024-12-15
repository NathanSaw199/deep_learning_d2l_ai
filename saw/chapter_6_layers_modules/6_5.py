import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# STEP 1 : create forward function within class
# STEP 2 : create class instance
# STEP 3 : pass input to class instance for forward function to be called and calculate output

#To start, we construct a custom layer that does not have any parameters of its own. This should look familiar if you recall our introduction to modules. The following CenteredLayer class simply subtracts the mean from its input. To build it, we simply need to inherit from the base layer class and implement the forward propagation function.
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    

layer = CenteredLayer()
print(layer(torch.tensor([1.0, 2, 3, 4, 5])))

#The nn.LazyLinear(128) layer takes an input tensor of size (4, 8) (where 4 is the batch size and 8 is the feature size) and outputs a tensor of size (4, 128).
#CenteredLayer() then centers this tensor by subtracting its mean..
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())

Y = net(torch.rand(4, 8))

#The output Y.mean() should be very close to zero (numerically it might not be exactly zero due to floating point precision limits), because CenteredLayer adjusts the output to be centered around zero.
Y.mean()

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        #super().__init__(): This line initializes the base class (nn.Module), which is necessary for all PyTorch modules to properly inherit methods and properties from PyTorch’s base Module class.
        super().__init__()
        #nn.Parameter to register weight and bias as trainable parameters. 
        #self.weight = nn.Parameter(...): Here, a weight matrix is created as an instance of nn.Parameter, which tells PyTorch that this tensor should be treated as a trainable parameter in the model. The weights are initialized randomly using a normal distribution (default behavior of torch.randn). The shape of this matrix is (in_units, units), where in_units is the number of input features to the layer, and units is the number of output features (neurons).
        self.weight = nn.Parameter(torch.randn(in_units, units))
        #self.bias = nn.Parameter(...): Similar to the weight matrix, a bias vector is also initialized as a trainable parameter with a shape of (units,), where each element corresponds to a bias term for one output neuron. It is also initialized with random values.
        self.bias = nn.Parameter(torch.randn(units,))
#This method defines how the input X is processed by the layer. It takes an input tensor X, which should have a shape where one of the dimensions matches in_units.
    def forward(self, X):
        #torch.matmul(X, self.weight.data) + self.bias.data: This line performs the matrix multiplication between X and the weight matrix, then adds the bias vector. The use of .data is generally not recommended because it circumvents PyTorch's gradient tracking system. It’s better to use self.weight and self.bias directly to ensure that PyTorch can compute gradients for these parameters during backpropagation.
        linear = torch.matmul(X, self.weight) + self.bias
        #F.relu(linear): The ReLU (Rectified Linear Unit) activation function is applied to the output of the linear transformation. ReLU is defined as F.relu(x) = max(0, x) and is used to introduce non-linearity into the model, allowing it to learn more complex patterns.
        return F.relu(linear)

#linear = MyLinear(5, 3): An object of MyLinear is created with 5 input units and 3 output units. This setup implies the weight matrix will be of size 5x3, and the bias vector will have 3 elements.
linear = MyLinear(5, 3)
#print(linear.weight): This prints the initial state of the weight matrix. It’s useful for verifying the initialization or debugging.
print(linear.weight)


net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))