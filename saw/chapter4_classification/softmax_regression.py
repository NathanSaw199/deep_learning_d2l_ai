
# STEP 1 : take input tensor X 
# STEP 2 : normalize the input tensor X with torch.exp(X)
# STEP 3 : sum the normalized tensor X along the axis 1 (rows) and keep the dimension
# STEP 4 :divide the normalized tensor X by the sum of the normalized tensor X along the axis 1 (rows) and return the result. First row of X_exp divided by first element of partition and Second row of X_exp divided by second element of partition: 
#STEP 5: create classifier class that encapsulates the loss function and accuracy function as methods. This class will be structured to take an input tensor, perform the softmax calculation, and return the softmax-transformed tensor.
#STEP 6:  creating a Softmax class that encapsulates the softmax function as a method. This class will be structured to take an input tensor, perform the softmax calculation, and return the softmax-transformed tensor. 
#STEP 7: Adding this forward method to the SoftmaxRegressionScratch class effectively completes the model by defining how inputs are processed to produce outputs
#STEP 8 : cross_entropy is designed to calculate the cross-entropy loss, which is commonly used in classification tasks to measure the difference between two probability distributions: the true distribution (actual class labels) and the predicted distribution (model output probabilities).
#STEP 9: The loss method is added to the SoftmaxRegressionScratch class using the @d2l.add_to_class decorator, allowing the model to compute its loss using the cross-entropy metric.
#STEP 10: prediction to classify some images


#the mapping from scalars to probabilities.recall the operation of the sum operator along specific dimensions in a tensor,  Given a matrix X we can sum over all elements (by default) or only over elements in the same axis. The axis variable lets us compute row and column sums:
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

#use a stochastic gradient descent optimizer, operating on minibatches of data. The optiizer is initialized with the model’s parameters and the learning rate lr.m
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
#X.sum(0,keepdim=True) sums over rows and returns a row vector. X.sum(1,keepdim=True) sums over columns and returns a column vector.
# print(X.sum(0,keepdim=True),X.sum(1,keepdim=True))
#Computing the softmax requires three steps: (i) exponentiation of each term; (ii) a sum over each row to compute the normalization constant for each example; (iii) division of each row by its normalization constant, ensuring that the result sums to 1:
#The (logarithm of the) denominator is called the (log) partition function. It was introduced in statistical physics to sum over all possible states in a thermodynamic ensemble. The implementation is straightforward:
def softmax(X):
    #Exponentiation: X_exp = torch.exp(X) computes the exponential of each element in the input tensor X. This operation is crucial for transforming raw logits into values that are suitable for normalization.
    X_exp = torch.exp(X)
    print("Xexp", X_exp)
    # calculates the sum of the exponentiated values along each row (axis 1). This sum acts as the normalization constant for each row and ensures the probabilities add up to 1. Keeping the dimension (keepdim=True) facilitates broadcasting in the division step.
    partition = X_exp.sum(1,keepdim=True)
    print("partition", partition)
    #return X_exp / partition divides each exponentiated logit by the sum of exponentiated logits in its corresponding row. This normalization ensures that each row of the output from the softmax function sums to 1, representing a probability distribution.
    return X_exp/partition
# creates a 2x5 tensor X with random values between 0 and 1. Each element is drawn from a uniform distribution.
X = torch.rand((2, 5))
# X = torch.tensor([[1.0,2.0,4.0],[4.0,5.0,6.0]])
# print(X.sum(0,keepdim=True),X.sum(1,keepdim=True))

#Using the previously defined softmax function, this line computes the softmax probabilities for the tensor X. The softmax function will exponentiate each element, sum them across rows, and then divide each element by the sum of its row to get probabilities.
X_prob = softmax(X)
print(X_prob,X_prob.sum(1))

#Since the raw data here consists of 28x28 pixel images, we flatten each image, treating them as vectors of length 784
#In softmax regression, the number of outputs from our network should be equal to the number of classes
#Since our dataset has 10 classes, our network has an output dimension of 10.Consequently, our weights constitute a 784x 10 matrix plus 1x 10 row vector for the biases
#defines a new class SoftmaxRegressionScratch that inherits from Classifier. The inheritance implies that SoftmaxRegressionScratch can use the methods and properties of Classifier, such as the validation_step method and any other utilities defined in the d2l.Module through Classifier
class SoftmaxRegresstionScratch(Classifier):
    def __init__(self,num_inputs,num_outputs,lr,sigma=0.01):
        #super().__init__(): This calls the constructor of the base class (Classifier), which sets up any necessary initializations inherited from d2l.Module.
        super().__init__()
        #this method would save or log the hyperparameters like num_inputs, num_outputs, lr, and sigma.
        self.save_hyperparameters()
        # Weights of the model, initialized from a normal distribution with mean 0 and standard deviation sigma. The size of self.W is (num_inputs, num_outputs), which allows the model to map input features to output classes.
        self.W= torch.normal(0,sigma,size = (num_inputs,num_outputs),requires_grad=True)
        #Bias terms, initialized to zeros. The size of self.b is (num_outputs,), allowing a bias to be added to each output class prediction.
        self.b = torch.zeros(num_outputs,requires_grad=True)
    
    def parameters(self):
        #It returns a list containing the weights and biases of the model, which are used by the optimizer during the training to update these parameters based on the computed gradients.
        return [self.W,self.b]

 #defines how the network maps each input to an output. Note that we flatten each 28x28  pixel image in the batch into a vector using reshape before passing the data through our model.
 #defines a forward method for the SoftmaxRegressionScratch class, and it utilizes the @d2l.add_to_class decorator to dynamically add this method to the class definition.
@d2l.add_to_class(SoftmaxRegresstionScratch)
#the forward method, which is a standard part of neural network implementations in PyTorch. The forward method specifies how the input data X passes through the network.
def forward(self,X):
    #The input tensor X is reshaped to ensure it matches the dimensions expected by the weights matrix self.W. The -1 in the reshape method allows PyTorch to automatically calculate the necessary dimension size so that the total number of elements remains consistent with the original tensor. This reshaping is crucial for preparing the batch data for matrix multiplication with the weights.
    X = X.reshape(-1,self.W.shape[0])
    #orch.matmul(X, self.W): Multiplies the input tensor X with the weights matrix self.W to compute the weighted sum of inputs.+ self.b: Adds the bias vector self.b to each row of the result from the matrix multiplication. This step shifts the decision boundary for each class
    return softmax(torch.matmul(X,self.W)+self.b)
#Implement the cross-entropy loss function. cross-entropy takes the negative log-likelihood of the predicted probability assigned to the true label.use indexing n particular, the one-hot encoding in y allows us to select the matching terms in y_hat
#To see this in action we create sample data y_hat with 2 examples of predicted probabilities over 3 classes and their corresponding labels y.The correct labels are 0 and 2.  (i.e., the first and third class). Using y as the indices of the probabilities in y_hat, we can pick out terms efficiently.
#y is a tensor containing the indices [0, 2]. These are typically labels or class indices in a classification task
if __name__ == '__main__':

    y = torch.tensor([0,2])
    #y_hat is a 2x3 tensor where each row is likely a probability distribution across three classes for two examples. For instance, the first row [0.1, 0.3, 0.6] might represent the predicted probabilities for three classes for the first example.
    y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
    # y_hat[[0,1],y] O is for the first vector of y_hat tensor and it is 0.1 because y = torch.tensor([0,2]) it is the index 0 of y_hat[0] and  y_hat[[0,1],y] 1 is for the first vector of y_hat tensor and it is 0.5 because y = torch.tensor([0,2]) it is the index 2 of y_hat[1]
    print(y_hat[[0,1],y])
    #Now we can implement the cross-entropy loss function by averaging over the logarithms of the selected probabilities.
    #y_hat: A tensor of predicted probabilities, where each row corresponds to a single example and each column corresponds to a class. The values in y_hat should be probabilities output by, for example, a softmax function, and they should sum to 1 across each row.
    #y: A tensor of true class labels, where each element is an index in the range [0, C-1] (for C classes) that indicates the true class for each example.
    def cross_entropy(y_hat,y):
        #list(range(len(y_hat))): This generates a list of indices corresponding to each row in y_hat. For example, if y_hat contains 3 examples (rows), this would generate [0, 1, 2].
        #y_hat[list(range(len(y_hat))), y]: This uses advanced indexing to select one probability per example. Specifically, it selects the predicted probability corresponding to the true class for each example. For instance, if y is [0, 2, 1], this operation would select the probability predicted for class 0 from the first row, class 2 from the second row, and class 1 from the third row of y_hat.
        #torch.log(...): Calculates the natural logarithm of each selected probability.the negative sign is applied because the logarithm of values between 0 and 1 is negative, and we want the loss to be a positive quantity.
        #.mean(): Computes the mean of the losses across all examples. This averaging step is typical in batch training, where the loss must summarize the performance of multiple data points.
        return -torch.log(y_hat[list(range(len(y_hat))),y]).mean()
    #y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
    #y = torch.tensor([0,2])
    #list(range(len(y_hat))) produces [0, 1] since y_hat has 2 rows (i.e., two examples).
    #y is [0, 2], specifying the class indices for each example
    #Using advanced indexing: y_hat[[0, 1], [0, 2]] extracts
    #From the first row (y_hat[0]), the element at index 0 → 0.1
    #From the second row (y_hat[1]), the element at index 2 → 0.5
    #the result of this operation is a tensor containing [0.1, 0.5].
    #log(0.1) ≈ -2.3026 and log(0.5) ≈ -0.6931
    #This results in [-2.3026, -0.6931].
    #(-2.3026 + -0.6931) / 2 = -1.49785
    #the negation of the log results' mean, it actually returns 1.49785, which is rounded/displayed as 1.4979
    print(cross_entropy(y_hat, y))
    # loss method to the SoftmaxRegressionScratch class using the @d2l.add_to_class decorator allows you to directly integrate the cross_entropy function within the class as its standard loss calculation method.
    #y_hat: The predicted probabilities from the model, which are expected to be the output of a softmax function—i.e., a tensor where each row represents the predicted probabilities of each class for a single example.
    #y: The actual labels, typically a tensor of integer indices that correspond to the true class labels for each example.
    @d2l.add_to_class(SoftmaxRegresstionScratch)
    def loss(self,y_hat,y):
        #The method simply calls the previously defined cross_entropy function with y_hat and y as arguments and returns the result. This encapsulation allows the softmax regression model to compute its loss using the cross-entropy metric, which quantifies how well the predicted probability distribution matches the actual distribution (defined by the labels).
        return cross_entropy(y_hat,y)
    #training the model requires the following steps:
    #d2l.FashionMNIST: This function is a utility provided by the d2l library to load the Fashion MNIST dataset, which contains images of fashion items, such as trousers, shirts, and bags.batch_size=256: This specifies that the data should be loaded and processed in batches of 256 items each. 
    data = d2l.FashionMNIST(batch_size=256)
    #num_inputs=784: Since Fashion MNIST images are 28x28 pixels, they're typically flattened into 784-dimensional vectors before being fed into a fully connected softmax model.num_outputs=10: There are 10 different classes in Fashion MNIST, one for each type of clothing item.r=0.1: This sets the learning rate for the optimizer, which controls how much the model's weights are adjusted with respect to the loss gradient on each step.
    model = SoftmaxRegresstionScratch(num_inputs=784, num_outputs=10, lr=0.1)
    #Specifies that the training should continue for 10 epochs, where an epoch represents one complete pass through the entire dataset.
    #d2l.Trainer: This is a utility class from the d2l library designed to simplify the training process. It abstracts away many of the repetitive tasks involved in setting up training loops.
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
# d2l.plt.show()
#now that training is complete, our model is ready to classify some images.
# data.val_dataloader(): This function is part of the d2l.FashionMNIST class setup and returns a DataLoader for the validation dataset. In PyTorch, a DataLoader is responsible for managing data loading for a dataset, including options for batching, shuffling, and parallel processing using multiple worker threads.
#iter(data.val_dataloader()): The iter() function is used to create an iterator from the DataLoader. DataLoader itself is an iterable that supports automatic batching, sampling, shuffling, and multiprocess data loading.  iter() is necessary to retrieve items in a sequential manner.The next() function fetches the next item from the iterator. In the context of a DataLoader, this means fetching the next batch. Each call to next() retrieves a new batch from the DataLoader until the dataset is exhausted. 
#X: This variable will contain the features of the validation dataset, typically a batch of images. Each image in the Fashion MNIST dataset is 28x28 pixels, and depending on how the dataset is structured for input into the model, X might be flattened into vectors of size 784 (28x28)
#y: This variable holds the labels corresponding to the images in X. For the Fashion MNIST dataset, these labels are integers between 0 and 9, where each integer represents a clothing category (like trousers, dresses, shirts, etc.).
    X, y = next(iter(data.val_dataloader()))
    #model(X): This calls the forward method of your SoftmaxRegresstionScratch model (or another model that has been defined and is being referred to as model) with X as the input. The output of model(X) is a tensor where each row corresponds to a different example in the batch and each column represents the probability that the example belongs to one of the classes (there are 10 classes in Fashion MNIST).
    #.argmax(axis=1): This function is applied to the result of model(X). It finds the index of the maximum value in each row of the tensor returned by model(X). Since the output tensor from the model is expected to be in the form of class probabilities (due to the softmax operation in your model), argmax is used to pick the index of the highest probability, which corresponds to the model's class prediction for each image.
    preds = model(X).argmax(axis=1)
    print(preds.shape)
    #preds: This is a tensor containing the predicted class indices for each example in the batch. These predictions are the result of the argmax operation applied to the softmax outputs of your model, which means they are typically integer values representing class labels.
    # .type(y.dtype): Ensures that the data type of the preds tensor matches the data type of the y tensor.The y tensor holds the actual labels
    # != y:This performs an element-wise comparison between the converted preds tensor and the y tensor. The result is a Boolean tensor (wrong) where each element is True if the corresponding prediction does not match the true label (indicating an incorrect prediction), and False if it matches (indicating a correct prediction).
    #wrong: This Boolean tensor is used to identify misclassified examples. Each True value in this tensor indicates a misclassification, and each False value indicates a correct classification.
    wrong = preds.type(y.dtype) != y
    #X[wrong]: This uses the Boolean tensor wrong, which contains True for each example where the prediction was incorrect and False for each correct prediction, to index into X. The result is a new tensor X that includes only the images from the original batch of data that were misclassified.y[wrong]: Similarly, this expression filters the labels tensor y to include only the labels corresponding to the misclassified images.
    #preds[wrong]: This filters the predictions tensor preds in the same way, yielding a tensor of predicted labels that only includes the incorrect predictions.
    X, y, preds = X[wrong], y[wrong], preds[wrong]
    #data.text_labels(y) and data.text_labels(preds):These function calls convert the numeric class labels in y and preds into their corresponding textual descriptions. For example, if y and preds contain labels for fashion items, the numerical label 0 might be converted to "T-shirt/top", 1 to "Trouser", and so on.
    #he zip function takes two iterables (in this case, the lists of textual labels for y and preds) and returns an iterator that aggregates elements from each of the iterables into pairs. Each pair consists of corresponding elements from the two lists, where the first element of the pair comes from data.text_labels(y) and the second from data.text_labels(preds).
    #[a + '\n' + b for a, b in zip(...)]:

    # This is a list comprehension that iterates over each pair of actual and predicted labels (denoted as a and b, respectively).
    # For each pair, it concatenates the actual label (a), a newline character ('\n'), and the predicted label (b). This results in a string format where the actual label is above the predicted label, making it easier to visually compare the two when displayed.
    labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
    data.visualize([X,y],labels=labels)
d2l.plt.show()

