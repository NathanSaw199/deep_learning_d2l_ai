import torch
from d2l import torch as d2l

#implement the entire method from scratch, including (i) the model; (ii) the loss function; (iii) a minibatch stochastic gradient descent optimizer; and (iv) the training function that stitches all of these pieces together.
#Before we can begin optimizing our model’s parameters by minibatch SGD,  have some parameters in the first place
#we initialize weights by drawing random numbers from a normal distribution with mean 0 and a standard deviation of 0.01, setting the bias b to 0.
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    #num_inputs is the number of features in the dataset
    #lr is the learning rate
    #sigma is the standard deviation of the normal distribution
    def __init__(self, num_inputs, lr, sigma=0.01):
        #super() function that returns a temporary object of a superclass that allows us to call that superclass’s methods.
        #super().__init__() is equivalent to nn.Module.__init__(self)
        super().__init__()
        #self.save_hyperparameters() is saving the hyperparameters of the model

        self.save_hyperparameters()
        #self.w is the weight of the model and is initialized by drawing random numbers from a 
    
        #torch.normal function generates a tensor of random numbers drawn from a normal (Gaussian) distribution.The normal distribution is defined by a mean (mean) and a standard deviation (std)
        # 0: This is the mean of the normal distribution. Here, it is set to 0, meaning the distribution is centered around 0.
        #sigma: This is the standard deviation of the normal distribution. It controls the spread of the distribution. sigma is for standard deviation
        # (num_inputs, 1): This tuple defines the shape of the output tensor. It will have num_inputs rows and 1 column.
        # requires_grad=True: This argument indicates that gradients should be computed for this tensor during backpropagation. This is essential for training the model using gradient descent.
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        # This function call creates a tensor filled with zeros.
        # The argument 1 specifies the shape of the tensor. In this case, it creates a tensor with a single element (a scalar).
        #This assigns the created tensor to an instance variable b of the class.
        # This variable will likely be used as a parameter in a linear regression model, representing the bias term.
        self.b = torch.zeros(1, requires_grad=True)
#Next we must define our model, relating its input and parameters to its output.
#  linear model we simply take the matrix–vector product of the input features X and the model weights W and add the offset B to each example The product Xw is a vector and b is a scalar
#The forward method computes the predicted value by multiplying the input data X with the model weight w and adding the offset b.
#The @add_to_class decorator is used to add the forward method to the LinearRegressionScratch class.
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    #The forward method computes the predicted value by multiplying the input data X with the model weight w and adding the offset b.
    return torch.matmul(X, self.w) + self.b
#since updating our model requires taking the gradient of our loss function, we ought to define the loss function first. we use the squared loss function  to transform the true value y into the predicted value’s shape y_hat. The result returned by the following method will also have the same shape as y_hat. We also return the averaged loss value among all examples in the minibatch.
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    # The squared loss is also known as the L2 norm loss
    l = (y_hat - y) ** 2 / 2
    #l.mean() is the average loss value among all examples in the minibatch
    return l.mean()

#The following code applies the update, given a set of parameters, a learning rate lr.
#Since our loss is computed as an average over the minibatch, we do not need to adjust the learning rate against the batch size.
#We define our SGD class, a subclass of HyperParameters
#We update the parameters in the step method. The zero_grad method sets all gradients to 0, which must be run before a backpropagation step.
# mini batch sgd is a method that updates the model parameters by taking a step in the direction of the negative gradient of the loss function with respect to the parameters for each minibatch of training data. # The step size is determined by the learning rate lr. #mini batch means that we update the model parameters based on a random subset of the training data, rather than the entire training data set.

class SGD(d2l.HyperParameters):  #@save
    # __init__ has three parameters: params, lr. params is the model parameters, lr is the learning rate, 
    def __init__(self, params, lr):
        #save_hyperparameters() saves the hyperparameters of the model
        # save_hyperparameters() means that the hyperparameters of the model are saved
        #Usage in SGD:
        # When implementing SGD, you often need to specify hyperparameters like the learning rate and the number of iterations. By saving these hyperparameters, you can ensure that they are consistently used throughout the training process and can be referenced or modified if needed.
        # Implementation:
        # The save_hyperparameters method should be implemented to store the relevant hyperparameters. In your code, it seems to be a placeholder that raises NotImplemented.
        self.save_hyperparameters()
    def step(self):
        for param in self.params:
            #param -= self.lr * param.grad: This line updates the model parameters by taking a step in the direction of the negative gradient of the loss function with respect to the parameters. The step size is determined by the learning rate. This is the key step in the optimization algorithm that minimizes the loss function by adjusting the model parameters in the direction of the negative gradient.
            param -= self.lr * param.grad
    #zero_grad method sets all gradients to 0. This is necessary before a backpropagation step to avoid accumulating gradients from previous steps.
    def zero_grad(self):
        #This method sets all the gradients of the model parameters to zero. This is necessary before computing the gradients for the next batch to avoid accumulating gradients from previous batches.
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
#We next define the configure_optimizers method, which returns an instance of the SGD class. This method is used by the Trainer class to configure the optimizer for the model.
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    #The configure_optimizers method returns an instance of the SGD class, which is used by the Trainer class to configure the optimizer for the model.
    return SGD([self.w, self.b], self.lr)
#we have all of the parts in place (parameters, loss function, model, and optimizer), we are ready to implement the main training loop.In each epoch, we iterate through the entire training dataset, passing once through every example (assuming that the number of examples is divisible by the batch size). In each iteration, we grab a minibatch of training examples, and compute its loss through the model’s training_step method. Then we compute the gradients with respect to each parameter. Finally, we will call the optimization algorithm to update the model parameters
@d2l.add_to_class(d2l.Trainer)  #@save
#The prepare_batch method is used to prepare the batch before passing it to the model. This method is called in the fit_epoch method of the Trainer class.
def prepare_batch(self, batch):
    return batch
@d2l.add_to_class(d2l.Trainer)  #@save
#The fit_epoch method is used to train the model for one epoch. This method is called in the fit method of the Trainer class.
def fit_epoch(self):
    #model.train() is used to set the model to training mode. This is necessary when training the model, as it enables features like dropout and batch normalization.
    self.model.train()
    #train_dataloader is an iterator that returns batches of training data. The for loop iterates over the batches of training data.
    for batch in self.train_dataloader:
        # self.prepare_batch(batch): This method is called with batch as its argument. The prepare_batch method is defined to simply return the batch it receives as input. 
        #self.model.training_step(...): After preparing the batch, the code calls the training_step method on self.model with the prepared batch as its argument. This method is likely responsible for performing a single step of training, such as forward propagation, loss calculation, and backpropagation.
        #The result of self.model.training_step(...) is assigned to the variable loss. This implies that the training_step method returns the loss value for the given batch, which is a common practice in training machine learning models to monitor and minimize the error.
        # loss is the loss value for the given batch
        loss = self.model.training_step(self.prepare_batch(batch))
        #self.model.optim.zero_grad(): This method call sets all the gradients of the model parameters to zero. This is necessary before computing the gradients for the next batch to avoid accumulating gradients from previous batches.
        self.optim.zero_grad()
        # torch.no_grad(): This context manager is used to disable gradient computation. This is useful when you don't need to compute gradients, such as during inference or when updating the model parameters.
        with torch.no_grad():
            #loss.backward(): This method call computes the gradients of the loss value with respect to the model parameters. This is a key step in training machine learning models using backpropagation.
            loss.backward()
            #self.gradient_clip_val >0: This condition checks if the gradient_clip_val attribute of the Trainer class is greater than zero. If it is, the code inside the if block is executed
            if self.gradient_clip_val > 0:  # To be discussed later
                #self.clip_gradients(self.gradient_clip_val, self.model): This method call clips the gradients of the model parameters to a specified value. This is useful for preventing the gradients from becoming too large, which can lead to numerical instability during training.
                self.clip_gradients(self.gradient_clip_val, self.model)
                #self.optim.step(): This method call updates the model parameters using the optimizer. This is the key step in the training loop that minimizes the loss function by adjusting the model parameters in the direction of the negative gradient.
            self.optim.step()
            # self.train_batch_idx += 1: This line increments the train_batch_idx attribute of the Trainer class by 1. This attribute is used to keep track of the number of training batches processed during training.
        self.train_batch_idx += 1
        #val_dataloader is not None: This condition checks if the val_dataloader attribute of the Trainer class is not None. If it is not None, the code inside the if block is executed.
    if self.val_dataloader is None:
        return
    #model.eval() is used to set the model to evaluation mode. This is necessary when evaluating the model, as it disables features like dropout and batch normalization.
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
    #to train the model, but first we need some training data. Here we use the SyntheticRegressionData class and pass in some ground truth parameters. Then we train our model with the learning rate lr=0.03 and set max_epochs=3. Note that in general, both the number of epochs and the learning rate are hyperparameters. In general, setting hyperparameters is tricky and we will usually want to use a three-way split, one set for training, a second for hyperparameter selection, and the third reserved for the final evaluation. 
#model = LinearRegressionScratch(2, lr=0.03) creates an instance of the LinearRegressionScratch class with two input features and a learning rate of 0.03. The model is initialized with random weights and a bias term.
model = LinearRegressionScratch(2, lr=0.03)
# data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2) creates an instance of the SyntheticRegressionData class with the specified ground truth parameters. This class generates synthetic regression data for training and validation.
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
# trainer = Trainer(max_epochs=3) creates an instance of the Trainer class with the specified maximum number of epochs. This class is used to train the model with the training data.
trainer = d2l.Trainer(max_epochs=3)
# trainer.fit(model, data) trains the model with the training data using the Trainer class. This method call executes the training loop for the specified number of epochs.
trainer.fit(model, data)
# d2l.plt.show()
with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')