import torch
from torch import nn
from d2l import torch as d2l
#Rather than directly manipulating the number of parameters, weight decay, operates by restricting the values that the parameters can take.More commonly called l2 regularization outside of deep learning circles when optimized by minibatch stochastic gradient descent, weight decay might be the most widely used technique for regularizing parametric machine learning models.s. The technique is motivated by the basic intuition that among all functions f, f= 0 assigning the value 0  to all inputs) is in some sense the simplest, and that we can measure the complexity of a function by the distance of its parameters from zero. The most common method for ensuring a small weight vector is to add its norm as a penalty term to the problem of minimizing the loss. Thus we replace our original objective, minimizing the prediction loss on the training labels, with new objective, minimizing the sum of the prediction loss and the penalty term. 
#our label is given by an underlying linear function of our inputs, corrupted by Gaussian noise with zero mean and standard deviation 0.01. For illustrative purposes, we can make the effects of overfitting pronounced, by increasing the dimensionality of our problem to d = 200 and working with a small training set containing only n = 20 examples.
# class data is the base class for data and has two methods: save_hyperparameters and get_dataloader which are used to save the hyperparameters and get the dataloader respectively 
class Data(d2l.DataModule):
    # __init__ function initializes the class with the specified hyperparameters which are the number of training examples, number of validation examples, number of input features, and batch size 
    #batch size is the number of examples in a batch from the dataset 
    # we used batch_size because we want to train the model on a batch of examples rather than the entire dataset
    def __init__(self,num_train,num_val,num_inputs,batch_size):
        self.save_hyperparameters()
        # n is the number of examples in the dataset and is the sum of the number of training and validation examples 
        # n is the total number of examples in the dataset
        #num_train is the number of training examples and num_val is the number of validation examples 
        #n = num_train +num_val because the dataset is the sum of the training and validation datasets
        n = num_train +num_val
        #X is a tensor of random values with dimensions n and num_inputs because we want to generate random input features for the dataset and n is the number of examples in the dataset and num_inputs is the number of input features 
        self.X =torch.randn(n,num_inputs)
        #noise is a tensor of random values with dimensions n and 1 because we want to generate random noise for the dataset and n is the number of examples in the dataset
        #we need noise to make the effects of overfitting pronounced 
        noise = torch.randn(n,1)*0.01
        #w is a tensor of ones with dimensions num_inputs and 1 because we want to generate a weight matrix for the input features and 1 is the number of output features 
        # output feature is one because we are performing linear regression and the output is a scalar value 
        w,b = torch.ones((num_inputs,1))*0.01,0.05
        #self.y is the output of the model and is computed by multiplying the input features with the weight matrix and adding the bias and noise
        self.y = torch.matmul(self.X,w)+b+noise
#get_dataloader function returns the dataloader for the dataset 
#get_dataloader function takes two parameters: train and i 
#get_dataloader is used as a helper function to get the dataloader for the dataset because we want to train the model on a batch of examples rather than the entire dataset 
    def get_dataloader(self, train):
        #i is a slice object that is used to get a slice of the dataset 
        #slice object is used to get a slice of the dataset 
        #slice(self.num_train,None) returns a slice of the dataset from the number of training examples to the end of the dataset 
        i = slice(0,self.num_train) if train else slice(self.num_train,None)
        #get_tensorloader function returns the tensorloader for the dataset
        #[self.X,self.y],train,i are the parameters of the get_tensorloader function because we want to get the tensorloader for the input features and output labels of the dataset, train is a boolean value that specifies whether the model is training or not, and i is the slice object that specifies the slice of the dataset
        return self.get_tensorloader([self.X,self.y],train,i)
# #mplementing weight decay from scratch. Since minibatch stochastic gradient descent is our optimizer, we just need to add the squared 
#  penalty l2 to the original loss function.
def l2_penalty(w):
    #this is the l2 penalty function that computes the l2 norm of the weight vector and returns the squared l2 norm of the weight vector divided by 2
    return(w**2).sum()/2
#The weightDecayScratch class is a subclass of the LinearRegressionScratch class and has three methods: __init__, loss, and l2_penalty
class weightDecayScratch(d2l.LinearRegressionScratch):
    #__init__ function initializes the class with the specified hyperparameters which are the number of input features, lambda, learning rate, and sigma
    def __init__(self,num_inputs,lambd,lr,sigma=0.01):
        #super() function is used to call the __init__ function of the parent class which is the LinearRegressionScratch class and initializes the class with the specified hyperparameters which are the number of input features, learning rate, and sigma
        super().__init__(num_inputs,lr,sigma)
        #lambd is the regularization parameter and is used to control the amount of regularization applied to the model
        self.save_hyperparameters()
        #self.lambd is the regularization parameter and is used to control the amount of regularization applied to the model
        #loss function computes the loss between the predicted and actual values and the l2_penalty function computes the l2 norm of the weight vector 
    def loss(self,y_hat,y):
        # this is the loss function that computes the loss between the predicted and actual values and returns the sum of the loss and the l2 penalty term
        return(super().loss(y_hat,y)+self.lambd*l2_penalty(self.w))
#The following code fits our model on the training set with 20 examples and evaluates it on the validation set with 100 examples.
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)
#train_scratch function trains the model from scratch with the specified regularization parameter
def train_scratch(lambd):
    #model is an instance of the weightDecayScratch class with the specified regularization parameter and learning rate lambda is the regularization parameter and is used to control the amount of regularization applied to the model
    model = weightDecayScratch(num_inputs=200, lambd=lambd, lr=0.001)
    #model.board.yscale is the scale of the y-axis of the loss plot and is set to log because we want to plot the loss on a logarithmic scale 
    model.board.yscale = 'log'
    trainer.fit(model, data)
    print('l2 norm of w:',float(l2_penalty(model.w)))
#train_scratch function trains the model from scratch with the specified regularization parameter 0 means no regularization and 3 means high regularization. no regularization means that the model is not regularized and high regularization means that the model is regularized and high regularization is used to prevent overfitting
train_scratch(0)
train_scratch(3)