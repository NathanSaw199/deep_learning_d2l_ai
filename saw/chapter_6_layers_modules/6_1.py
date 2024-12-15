import torch
from torch import nn
from torch.nn import functional as F
#This is a container that sequences the layers and functions listed within it. When data is passed to a Sequential container, it is processed sequentially through each layer or function in the order they are added.It automatically handles the forwarding of outputs from one layer to the input of the next.
#nn.LazyLinear(256):LazyLinear is a variation of the standard nn.Linear layer that does not require the number of input features (in_features) to be specified. Instead, LazyLinear automatically infers the size of each input sample in its first forward pass.In this case, it's configured to produce an output of 256 features. It will determine the number of input features when it receives the initial batch of data and will initialize its weights accordingly.
#nn.ReLU():This layer applies the Rectified Linear Unit (ReLU) activation function. The ReLU function is defined as f(x) = max(0,x).This activation function is used to introduce non-linearity into the network, allowing it to learn more complex patterns.ReLU is commonly used because it is computationally efficient and helps in reducing problems like the vanishing gradient during training. 
#nn.LazyLinear(10):Another LazyLinear layer, this one configured to reduce the dimensionality of the output from the previous layer (256 features) down to 10 features. This could be useful, for example, in a classification problem where you need to output probabilities for 10 different classes.
#torch.rand(2,20):This function generates a random tensor of shape (2, 20), which simulates a mini-batch of 2 samples, each with 20 features. This could represent, for instance, input data where each feature might be some measurement or value relevant to the task at hand.
#Activation: ReLU is used as an activation function in neural networks. It activates a neuron in the network only if the input is above zero, passing it through unchanged. If the input is less than or equal to zero, it outputs zero. This is akin to saying that the neuron does not activate for non-positive inputs.
net = nn.Sequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
X = torch.rand(2,20)
print(net(X).shape)

#Perhaps the easiest way to develop intuition about how a module works is to implement one ourselves. Before we do that, we briefly summarize the basic functionality that each module must provide:

# 1. Ingest input data as arguments to its forward propagation method.

#2 . Generate an output by having the forward propagation method return a value. Note that the output may have a different shape from the input. For example, the first fully connected layer in our model above ingests an input of arbitrary dimension but returns an output of dimension 256.
#3. Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation method. Typically this happens automatically.
#4.Store and provide access to those parameters necessary for executing the forward propagation computation.
#5. Initialize model parameters as needed.

class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        #self.hidden = nn.LazyLinear(256): This defines the hidden layer with nn.LazyLinear(256). This layer will automatically determine the number of input features upon its first execution and configure itself to output 256 features. The nn.LazyLinear module is particularly useful when you do not know the size of the input layer at the time of model initialization.self.out = nn.LazyLinear(10): Defines the output layer, which will transform the output of the hidden layer (256 features) down to 10 features. This is common in classification tasks where these 10 features might represent the scores for 10 different classes.
        self.hidden = nn.LazyLinear(256) # Hidden layer
        self.out = nn.LazyLinear(10) # Output layer
    

    # Define the forward propagation of the model, that is, how to return the required model output based on the input X
    #he forward method defines the computation performed at every call and must be overridden when subclassing nn.Module.
    def forward(self,X):
        #F.relu(self.hidden(X)): The input X is passed through the hidden layer and then through a ReLU activation function. The ReLU (rectified linear unit) function is applied to introduce non-linearity, enhancing the model's ability to learn complex patterns.self.out(...): The output from the ReLU function is then fed into the output layer.The method returns the final output of the network, which will have the shape (batch_size, 10) after the input X is processed, where batch_size is inferred from the input and 10 is the number of output features defined by the output layer.
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print(net(X).shape)


#s. Recall that Sequential was designed to daisy-chain other modules together. To build our own simplified MySequential, we just need to define two key methods:

# 1.A method for appending modules one by one to a list.
# 2. A forward propagation method for passing an input through the chain of modules, in the same order as they were appended.


class MySequential(nn.Module):
    #The constructor accepts a variable number of arguments (*args), each representing a neural network module.

    def __init__(self, *args):
        super().__init__()
        #for idx, module in enumerate(args): This loop iterates over the modules passed as arguments. It uses add_module to add each module to the container. add_module requires a name (here given as the string representation of the index idx) and the module object itself.
        for idx,module in enumerate(args):
            self.add_module(str(idx),module)

    #The forward method defines how the input X is processed through the sequence of modules.for module in self.children(): Iterates over each module that was added in the constructor. The self.children() function returns an iterator over immediate child modules, respecting the order they were added.
    #X = module(X): This applies each module to X, updating X with the output of the current module before passing it to the next.The final output after the last module has processed the input is returned.
    def forward(self,X):
        for module in self.children():
            X = module(X)
        return X

net = MySequential(nn.LazyLinear(256),nn.ReLU(),nn.LazyLinear(10))
print(net(X).shape)
   
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
print(net(X))


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64),nn.ReLU(),
                                 nn.LazyLinear(32),nn.ReLU())
        self.linear = nn.LazyLinear(16)
    
    def forward(self,X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(),nn.LazyLinear(20),FixedHiddenMLP())   
print(chimera(X))

