import torch
from torch import nn
from d2l import torch as d2l

#STEP 1: create classifier class that inherits from d2l.Module

#STEP 2 : initialize the model parameters  we use nn.Parameter to automatically register a class attribute as a parameter to be tracked by autograd

#STEP 3 : define Model relu activation function

#STEP 4 : define forward method

#STEP 5 : train the model


#Inherits: MLPscratch inherits from Classifier, implying it uses all functionalities of the Classifier including the validation_step.

class MLPScratch(d2l.Classifier):
    #num_inputs: Number of input features.
    #num_outputs: Number of output features (e.g., classes in classification).
    #num_hiddens: Number of neurons in the hidden layer.
    #lr: Learning rate for the optimizer.
    #sigma: Standard deviation used in the initialization of weights.
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
    #torch.randn(num_inputs, num_hiddens): This function generates a tensor with random numbers drawn from a standard normal distribution (mean = 0, standard deviation = 1). The dimensions of the tensor are determined by num_inputs (the number of input features to the network) and num_hiddens (the number of neurons in the hidden layer). * sigma: The randomly generated weights are then scaled by sigma, a predefined small standard deviation value, which helps control the initial variance of the weights. The scaling factor sigma is crucial for ensuring that the weights start small, reducing the likelihood of saturating the neurons' activation functions early in training.
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
    #torch.zeros(num_hiddens): This initializes the biases for the first layer to a tensor of zeros. The length of this tensor matches the number of neurons in the hidden layer (num_hiddens). Initializing biases to zero is a common practice, as the non-zero weights are usually sufficient to begin learning effectively.
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        #torch.randn(num_hiddens, num_outputs): Similar to self.W1, this generates a tensor of random values from a normal distribution, with dimensions defined by num_hiddens (the number of neurons in the hidden layer) and num_outputs (the number of output features or classes).
        #* sigma: The random values are again scaled by sigma to control their initial variance. This scaling is important for consistency in the initialization scale across different layers of the network.
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        #torch.zeros(num_outputs): This line initializes the biases for the output layer to zeros. The length of the tensor is num_outputs, corresponding to the number of output classes or features. Starting with zero biases is typically effective and does not hinder the initial stages of learning.
        self.b2 = nn.Parameter(torch.zeros(num_outputs))


        # implement the ReLU activation 
#x: This is the input tensor for which you want to apply the ReLU activation function.
def relu(X):
    #torch.zeros_like(x): This creates a new tensor a that has the same shape as x, but filled entirely with zeros. This tensor serves as the threshold for the ReLU function, which will compare each element in x against zero.
    a = torch.zeros_like(X)
    #torch.max(x, a): This function takes two tensors x and a, and returns a new tensor. For each element, it returns the maximum value between the corresponding elements in x and a. Essentially, for each element in x, it compares it against 0 (since a is a zero tensor), and returns the maximum of x or 0. 
    return torch.max(X, a)

#Since we are disregarding spatial structure, we reshape each two-dimensional image into a flat vector of length num_inputs
#to add the forward method to the MLPscratch class. 
@d2l.add_to_class(MLPScratch)
#method definition, where self refers to the instance of MLPscratch, and X is the input tensor to the network.
def forward(self, X):
    # reshapes the input tensor X to ensure it has the correct number of features (self.num_inputs). The -1 in the reshape function allows PyTorch to automatically calculate the appropriate number of rows based on the batch size, ensuring that the tensor is 2D with each row representing an input example.
    X = X.reshape((-1, self.num_inputs))
    #torch.matmul(X, self.W1): Matrix multiplication between the input X and the weights of the first layer self.W1 and Addition of the bias self.b1.ReLU activation function sets all negative values to zero
    H = relu(torch.matmul(X, self.W1) + self.b1)
    #return torch.matmul(H, self.W2) + self.b2: This line computes the final output of the network. It multiplies the hidden layer outputs H by the second layer weights self.W2 and adds the bias self.b2. The result is the final output of the network.
    return torch.matmul(H, self.W2) + self.b2
if __name__ == "__main__":
    model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model,data)
    d2l.plt.show()

#concise implementation
# #class MLP(d2l.Classifier):
#     def __init__(self, num_outputs, num_hiddens, lr):
#         super().__init__()
#         self.save_hyperparameters()
#         self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
#                                  nn.ReLU(), nn.LazyLinear(num_outputs))
#model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
#trainer.fit(model, data)