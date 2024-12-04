# %matplotlib inline
import torch
from d2l import torch as d2l

# Define a linear regression model from scratch
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Initialize weights with normal distribution
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        # Initialize bias with zeros
        self.b = torch.zeros(1, requires_grad=True)

@d2l.add_to_class(LinearRegressionScratch)  #@save
# Define the forward pass
def forward(self, X):
    # Perform matrix-vector multiplication and add bias
    return torch.matmul(X, self.w) + self.b

# -----------------------------------------------------

@d2l.add_to_class(LinearRegressionScratch)  #@save
# Define the loss function
def loss(self, y_hat, y):
    # Calculate mean squared error
    l = (y_hat - y) ** 2 / 2
    return l.mean()

# -----------------------------------------------------

# Define stochastic gradient descent optimizer
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    # Update parameters
    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    # Clear gradients
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@d2l.add_to_class(LinearRegressionScratch)  #@save
# Configure the optimizer
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

# -----------------------------------------------------
# Now that we have all of the parts in place ( parameters, forward pass, loss function, and optimizer), we can train the model.
# epochs is the number of passes through the dataset

@d2l.add_to_class(d2l.Trainer)  #@save
# Prepare a batch of data
def prepare_batch(self, batch):
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
# Train the model for one epoch
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        # Compute loss for the batch
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1

# Create a linear regression model with 2 inputs and learning rate 0.03
model = LinearRegressionScratch(2, lr=0.03)
# Generate synthetic data for training
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# Create a trainer with 3 epochs
trainer = d2l.Trainer(max_epochs=3)
# Train the model
# Note that `model` is an instance of `LinearRegressionScratch`
trainer.fit(model, data)

# Show the plot
d2l.plt.show()

# -----------------------------------------------------

# Evaluate the model
with torch.no_grad():
    # Print error in estimating weights
    # Note that `data.w` is a tensor, whereas `model.w` is a tensor
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    # Print error in estimating bias
    # Note that `data.b` is a scalar, whereas `model.b` is a tensor
    print(f'error in estimating b: {data.b - model.b}')