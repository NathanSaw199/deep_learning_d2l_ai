import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# linear regression model is defined using nn.Linear 
class Linear_regression(d2l.Module):
    # __init__ function initializes the model parameters and saves them in the model 
    def __init__(self,lr):
        super().__init__()
        self.save_hyperparameters()
        # nn.Linear is a class that implements a linear transformation 
        # y = xw^T + b
        # nn.linear is a subclass of nn.Module and has two parameters: input_dim and output_dim 
        # input_dim is the number of input features and output_dim is the number of output features
        self.net = nn.LazyLinear(1)
        # weight.data.normal initializes the weight matrix with random values from a normal distribution with mean 0 and standard deviation 0.01
        self.net.weight.data.normal_(0,0.01)
        # bias.data.fill_ initializes the bias vector with zeros
        self.net.bias.data.fill_(0)
    
@d2l.add_to_class(Linear_regression)
# forward function defines how the model processes inputs and returns outputs 
# forward function is called when the model is called with input data
def forward(self,X):
    # X is the input data
    # net(X) returns the output of the model
    return self.net(X)
# The MSELoss class computes the mean squared error (without the 1/2 factor . By default, MSELoss returns the average loss over examples. It is faster (and easier to use) than implementing our own.
@d2l.add_to_class(Linear_regression)
# loss function computes the loss between the predicted and actual values
def loss(self,y_hat,y):
    # nn.MSELoss() creates a loss function that computes the mean squared error between the predicted and actual values
    fn = nn.MSELoss()
    # fn(y_hat,y) returns the loss between the predicted and actual values
    return fn(y_hat,y)
# Minibatch SGD is a standard tool for optimizing neural networks and thus PyTorch supports it alongside a number of variations on this algorithm in the optim module. When we instantiate an SGD instance, we specify the parameters to optimize over, obtainable from our model via self.parameters(), and the learning rate (self.lr) required by our optimization algorithm.
@d2l.add_to_class(Linear_regression)
# configure_optimizers function specifies the optimization algorithm and the learning rate 
# configure_optimizers function is called when the model is trained
# configure_optimizers function returns the optimizer 
# optimizer is used to update the model parameters 
def configure_optimizers(self):
    # torch.optim.SGD is a class that implements the stochastic gradient descent optimization algorithm and is used to update the model parameters
    return torch.optim.SGD(self.parameters(),lr=self.lr)

#  We did not have to allocate parameters individually, define our loss function, or implement minibatch SGD.
# Now that we have all the basic pieces in place, the training loop itself is the same as the one we implemented from scratch. So we just call the fit method, which relies on the implementation of the fit_epoch method to train our model.
# The fit method takes the model, data, and trainer as input and trains the model using the data and trainer
# The fit method trains the model for the specified number of epochs
# The fit method returns the trained model
# model = Linear_regression(lr=0.03) creates a linear regression model with a learning rate of 0.03 to train the model 
model = Linear_regression(lr=0.03)
# d2l.synthetic_data(w,b) generates synthetic data for linear regression with the specified weight and bias 
# data is a dictionary that contains the input features and output labels 
#w=torch.tensor([2,-3.4] is the weight vector 2 and -3.4 are the weights of the input features and b is the bias 
data = d2l.SyntheticRegressionData(w=torch.tensor([2,-3.4]),b=4.2)
# d2l.Trainer(max_epochs=3) creates a trainer that trains the model for 3 epochs 
# epochs is the number of times the model is trained on the entire dataset 
trainer = d2l.Trainer(max_epochs=3)
# trainer.fit(model,data) trains the model using the data and trainer
trainer.fit(model,data)
d2l.plt.show()
@d2l.add_to_class(Linear_regression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')