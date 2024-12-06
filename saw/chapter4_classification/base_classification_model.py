import torch
from d2l import torch as d2l
#We define the Classifier class below. In the validation_step we report both the loss value and the classification accuracy on a validation batch. 
#We draw an update for every num_val_batches batches. This has the benefit of generating the averaged loss and accuracy on the whole validation data. 
#classifier is a subclass of Module, which is a subclass of nn.Module. This means that we can use all the methods and attributes of nn.Module in classifier. and we can use the plot method to draw the loss and accuracy values. 
class Classifier(d2l.Module):
    # The method validation_step takes self and batch as arguments. self refers to the instance of the Classifier class, and batch is expected to be a tuple or list containing input data and labels.
    def validation_step(self,batch):
        # uses Python’s argument unpacking feature (*) to pass the elements of batch (except the last one, which is typically the labels) as separate arguments to the model’s forward method. The model then returns the predicted outputs y_hat
        y_hat = self(*batch[:-1])
        #calculates the loss by passing the predictions (y_hat) and the actual labels (batch[-1]) to the loss method of the class. The computed loss is then plotted using the plot method with the label 'loss'. The train=False parameter indicates that this is not training data (it’s validation data).
        self.plot('loss',self.loss(y_hat,batch[-1]),train=False)
        #computes the accuracy by calling the accuracy method with the predictions and the actual labels. The result is plotted using the plot method with the label 'acc'.
        self.plot('acc',self.accuracy(y_hat,batch[-1]),train=False)

#use a stochastic gradient descent optimizer, operating on minibatches of data. The optimizer is initialized with the model’s parameters and the learning rate lr.
#This decorator is used to dynamically add the configure_optimizers method to the d2l.Module class.It essentially extends the class functionality without modifying the original class code directly
@d2l.add_to_class(d2l.Module)
#is defined to configure and return the optimizer used for training a neural network model.This refers to the instance of the class to which this method will be added, enabling access to class attributes and other methods.
def configure_optimizers(self):
    #self.parameters(): This function call retrieves all trainable parameters of the model, which is necessary for the optimizer to know which parameters it will be updating.
    # This sets the learning rate of the optimizer to the value of the lr attribute of the class instance. The learning rate controls how much to change the model in response to the estimated error each time the model weights are updated.
    return torch.optim.SGD(self.parameters(),lr=self.lr)
#Given the predicted probability distribution y_hat, we typically choose the class with the highest predicted probability whenever we must output a hard prediction.
#Accuracy is computed as follows. First, if y_hat is a matrix, we assume that the second dimension stores prediction scores for each class. We use argmax to obtain the predicted class by the index for the largest entry in each row
# Then we compare the predicted class with the ground truth y elementwise. Since the equality operator == is sensitive to data types, we convert y_hat’s data type to match that of y. The result is a tensor containing entries of 0 (false) and 1 (true). Taking the sum yields the number of correct predictions.

#This decorator is used to add the accuracy method to the Classifier class defined in the d2l (Deep Learning) library
@d2l.add_to_class(Classifier)  #@save
#defining accuracy as taking three parameters:
# self: Refers to the instance of the class.
# Y_hat: The predicted outputs of the model for a batch of data, usually as logits or probabilities for each class.
# Y: The actual labels corresponding to the input data.
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    # reshapes the Y_hat tensor such that it has a shape of (-1, number_of_classes). -1 is used as a dimension size in reshape, PyTorch calculates what this dimension must be so that the total number of elements in the reshaped tensor remains the same as the original tensor. It does this by dividing the total number of elements in the tensor by the product of the sizes of all other specified dimensions.
    #If Y_hat is of shape (10, 20, 30) and you reshape it using Y_hat.reshape((-1, 30).Calculate the total number of elements in the tensor, which is 10*20*30 = 6000. Then divide this by the size of the second dimension in the reshaped tensor, which is 30. The result is 200. So, the reshaped tensor will have a shape of (200, 30).The last dimension size is explicitly 30.
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    #The argmax function is applied along axis=1 (columns), which returns the indices of the maximum values along this axis. These indices represent the class predictions. The predictions are then converted (type(Y.dtype)) to the same data type as the actual labels, ensuring compatibility for subsequent operations.
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    #compares the predicted classes (preds) with the actual labels (Y). Y is reshaped to a flat vector to match the shape of preds. The comparison results in a tensor of 0s and 1s, where 1 indicates a correct prediction. The result is then cast to torch.float32 for floating-point operations.
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare