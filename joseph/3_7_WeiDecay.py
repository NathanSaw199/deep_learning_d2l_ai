# %matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

# Weight Decay
# Weight decay is a regularization technique that prevents the weights from growing too large by adding a penalty term to the loss function.

# Over fitting is a common problem in machine learning, where a model performs well on training data but does not generalize well to unseen data (test data).
# Under fitting occurs when a model is too simple to learn the underlying structure of the data.

# Norm and weight decay is a regularization technique that penalizes the norm of the weights in the loss function.

# High-dimensional Linear Regression
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        # Generate random input data
        self.X = torch.randn(n, num_inputs)
        # Generate random noise
        noise = torch.randn(n, 1) * 0.01
        # Initialize weights and bias
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        # Generate output data with noise
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        # Get the appropriate slice of data for training or validation
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


# Defining Norm Penalty
def l2_penalty(w):
    # Calculate L2 penalty (regularization term)
    return (w ** 2).sum() / 2 

# Defining the model
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        # Calculate loss with L2 penalty
        return (super().loss(y_hat, y) +
        self.lambd * l2_penalty(self.w))
    

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):
    # Initialize model with given lambda (regularization strength)
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    # Train the model
    trainer.fit(model, data)
    # Print L2 norm of weights
    print('L2 norm of w:', float(l2_penalty(model.w)))

# Training without regularization
# without weight decay
train_scratch(0)

d2l.plt.show()

# Using weight decay
train_scratch(3)


# -----------------------------------------------------

# Define model with weight decay using built-in optimizer
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        # Configure optimizer with weight decay
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
    

# Initialize and train the model with weight decay
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

# Print L2 norm of weights
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))

d2l.plt.show()