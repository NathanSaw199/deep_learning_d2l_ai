import torch
from torch import nn
from d2l import torch as d2l
#Dropout is a regularization technique designed to prevent overfitting in neural networks by randomly setting some neuron activations to zero during training.


#STEP 1 : define dropout layer 

    #dropout_layer, takes two arguments: a tensor X representing data (input activations to a layer) and a dropout rate specifying the probability of each element being dropped (set to zero).ensures the dropout rate is between 0 and 1. returning a tensor of zeros when dropout is 1 and returning X unchanged when dropout is 0.uses a mask to drop elements and scales the remaining elements by 1/(1 - dropout) to maintain the expected sum of the inputs, which is a standard practice in dropout regularization.

#STEP 2 : define DropoutMLPScratch class that inherits from d2l.Classifier and implements the forward method
    #DropoutMLPScratch class inherits from d2l.Classifier and defines a multi-layer perceptron (MLP) with dropout.: Initializes two hidden layers and an output layer, with the ability to specify the number of units in each and the dropout rates after the first and second hidden layers. Implements the forward pass, applying linear transformations followed by ReLU activations. Dropout is applied to the activations of the first and second hidden layers conditionally if the model is in training mode. This ensures dropout is not used during evaluation, which is standard practice.

#STEP 3 : train the model
    #define hyperparameters such as the number of outputs, hidden units, dropout rates, and learning rate.

#dropout_layer, takes two arguments: a tensor X representing data (input activations to a layer) and a dropout rate specifying the probability of each element being dropped (set to zero).
def dropout_layer(X, dropout):
    #asserts that the dropout rate must be between 0 and 1.
    assert 0 <= dropout <= 1
    #If dropout is 1, it returns a tensor of zeros with the same shape as X, effectively dropping all inputs.If dropout is 0, the function will do nothing to X, as the mask will consist entirely of 1s.
    if dropout == 1: return torch.zeros_like(X)
    #torch.rand(X.shape) generates a tensor of random numbers uniformly distributed between 0 and 1, having the same shape as X.(torch.rand(X.shape) > dropout).float() creates a binary mask where elements are 1 if the random number is greater than the dropout rate (meaning the neuron survives) and 0 otherwise. 
    mask = (torch.rand(X.shape) > dropout).float()
    # The input X is multiplied element-wise by the mask, and then normalized by dividing by (1.0 - dropout) to maintain the expected value of the inputs.
    return mask * X / (1.0 - dropout)
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
#DropoutMLPScratch inherits from d2l.Classifier
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        #super().__init__() calls the constructor of the base class d2l.Classifier.
        super().__init__()
        self.save_hyperparameters()
        #nn.LazyLinear(...): Creates linear layers (fully connected layers) with deferred weight initialization. The actual number of input features to these layers will be inferred the first time the forward pass is called with an input. 
        #num_hiddens_1 and num_hiddens_2 define the number of neurons in the first and second hidden layers, respectively.
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        #num_outputs defines the number of neurons in the output layer, which typically corresponds to the number of classes in a classification task.
        self.lin3 = nn.LazyLinear(num_outputs)
        #An instantiation of the ReLU activation function, which introduces non-linearity to the model and is defined by the formula ReLU(x) = max(0, x).
        self.relu = nn.ReLU()

    def forward(self, X):
        #The input tensor X is reshaped to a 2D tensor where the first dimension is batch size and the second dimension is the flattened feature vector: X.reshape((X.shape[0], -1)).self.lin1: Applies the first linear transformation.self.relu(...): Applies the ReLU activation function to introduce non-linearity.
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            #dropout_layer(H1, self.dropout_1): Applies dropout to the output of the first hidden layer (H1) during training to prevent overfitting. The dropout rate dropout_1 determines the probability of each neuron being dropped.
            H1 = dropout_layer(H1, self.dropout_1)
            #self.lin2 and another self.relu(...): Processes the output of the first layer through the second linear transformation and activation function.dropout_layer(H2, self.dropout_2): Similar to the first dropout layer, applies dropout to the second hidden layer's output during training.
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
            #self.lin3(H2): The final linear transformation is applied to the output of the second hidden layer to produce the output of the network. This output is typically fed into a softmax function for classification tasks (not shown here but typically handled in the loss function during training).
        return self.lin3(H2)
    
if __name__ == "__main__":
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
            'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    model = DropoutMLPScratch(**hparams)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    d2l.plt.show()