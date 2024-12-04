import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# Define the model

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters
        self.net = nn.LazyLinear(1)  # Define a lazy linear layer with one output
        self.net.weight.data.normal_(0, 0.01)  # Initialize weights with normal distribution
        self.net.bias.data.fill_(0)  # Initialize bias to zero

@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)  # Forward pass through the network

# Define the loss function

@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()  # Mean Squared Error loss
    return fn(y_hat, y)  # Compute the loss

# Define the optimizer algorithm

@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)  # Use Stochastic Gradient Descent optimizer

# Training the model

model = LinearRegression(lr=0.03)  # Create a linear regression model with learning rate 0.03
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)  # Generate synthetic data
trainer = d2l.Trainer(max_epochs=3)  # Create a trainer with 3 epochs
trainer.fit(model, data)  # Train the model with the data

d2l.plt.show()  # Show the plot

@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)  # Get the weights and bias
w, b = model.get_w_b()  # Retrieve the weights and bias

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')  # Print the error in estimating weights
print(f'error in estimating b: {data.b - b}')  # Print the error in estimating bias
