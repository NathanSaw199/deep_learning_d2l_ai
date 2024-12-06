# MNIST served as the point of reference for comparing machine learning algorithms. like benchmarking
import time
import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
#load the dataset
#class fansionMNIST is a subclass of d2l.DataModule which is a subclass of torch.utils.data.Dataset to load the dataset and make it available for training and validation
#Fashion-MNIST consists of images from 10 categories, each represented by 6000 images in the training dataset and by 1000 in the test dataset. A test dataset is used for evaluating model performance (it must not be used for training). Consequently the training set and the test set contain 60,000 and 10,000 images, respectively.
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    # __init__ method is used to initialize the class that is used to create an object of the class and it has the self parameter which is a reference to the current instance of the class and batch_size and resize are the parameters of the class. batch_size is the number of samples in each batch and resize is the size of the image. batch_size = 64 because we are using 64 samples in each batch and 64 samples because it is a good number for training the model and resize = (28, 28) because the size of the image is 28x28
    def __init__(self, batch_size=64, resize=(28, 28)):
        #super() function is used to call the parent class constructor which is d2l.DataModule and it initializes the parent class
        super().__init__()
        # save_hyperparameters() method is used to save the hyperparameters of the class which are batch_size and resize
        self.save_hyperparameters()
        # trans is a variable that stores the transformations that are to be applied to the dataset images. transforms.Compose() is used to compose the transformations and transforms.Resize() is used to resize the image to the specified size and transforms.ToTensor() is used to convert the image to a tensor. we need to convert images to tensors because the input to the neural network should be in the form of tensors
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        # self.train is a variable that stores the training dataset. torchvision.datasets.FashionMNIST() is used to load the FashionMNIST dataset. root is the path where the dataset is stored, train=True is used to load the training dataset, transform is the transformations that are to be applied to the dataset images, and download=True is used to download the dataset if it is not already downloaded
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        # self.val is a variable that stores the validation dataset. torchvision.datasets.FashionMNIST() is used to load the FashionMNIST dataset. root is the path where the dataset is stored, train=False is used to load the validation dataset, transform is the transformations that are to be applied to the dataset images, and download=True is used to download the dataset if it is not already downloaded
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
#data is a variable that stores the FashionMNIST dataset. FashionMNIST() is used to load the FashionMNIST dataset. resize=(32, 32) is used to resize the image to the specified size.The images are grayscale and upscaled to 32x 32 pixels in resolution above. This is similar to the original MNIST dataset which consisted of (binary) black and white images. 
data = FashionMNIST(resize=(32, 32))
#len() function is used to get the length of the dataset. data.train is used to get the training dataset and data.val is used to get the validation dataset
print(len(data.train),len(data.val))
# that most modern image data has three channels (red, green, blue) and that hyperspectral images can have in excess of 100 channels (the HyMap sensor has 126 channels). By convention we store an image as a c x h x w tensor, where c is the number of color channels, h is the height and w is the width of the image.
#data.train[0][0].shape is used to get the shape of the first image in the training dataset. data.train[0] is used to get the first image in the training dataset and data.train[0][0] is used to get the image tensor of the first image in the training dataset because the first element of data.train[0] is the image tensor and the second element of data.train[0] is the label of the image
# data.train[0][1] is the label of the first image in the training dataset
#data.train[0][0].shape means that the shape of the first image in the training dataset is (1, 32, 32) which means that the image has 1 channel, 32 height, and 32 width 
print(data.train[0][1])
print(data.train[0][0])
data.train[0][0].shape
#The categories of Fashion-MNIST have human-understandable names. The following convenience method converts between numeric labels and their names.

@d2l.add_to_class(FashionMNIST)
#text_labels() method is used to get the names of the categories of Fashion-MNIST. labels is a list that stores the names of the categories of Fashion-MNIST. return labels[i] for i in indices is used to return the names of the categories of Fashion-MNIST for the given indices 
#indices means the indices of the categories of Fashion-MNIST and it is a list of integers that stores the indices of the categories of Fashion-MNIST
def text_labels(self,indices):
    #labels is a list that stores the names of the categories of Fashion-MNIST
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    #[labels[int(i)] for i in indices] means that for each index i in the list indices, the name of the category of Fashion-MNIST is returned and the name of the category is stored in the list 

    return [labels[int(i)] for i in indices]

#To make our life easier when reading from the training and test sets, we use the built-in data iterator rather than creating one from scratch. Recall that at each iteration, a data iterator reads a minibatch of data with size batch_size. We also randomly shuffle the examples for the training data iterator.
@d2l.add_to_class(FashionMNIST)  #@save
#get_dataloader() is a method that returns the data loader for the training or validation dataset.
#train is a boolean variable that is used to check whether the data loader is for the training dataset or the validation dataset. If train is True, the data loader is for the training dataset and if train is False, the data loader is for the validation dataset
def get_dataloader(self, train):
    data = self.train if train else self.val
    #torch.utils.data.DataLoader() is used to create a data loader. data is the dataset from which the data loader is created, self.batch_size is the number of samples in each batch, shuffle=train is used to shuffle the data if train is True, and num_workers=self.num_workers is the number of subprocesses to use for data loading
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
#To see how this works, let’s load a minibatch of images by invoking the train_dataloader method. It contains 64 images.
#data.train_dataloader() is used to get the data loader for the training dataset. next() function is used to get the next element from the data loader and it returns the next element from the data loader. X, y = next(iter(data.train_dataloader())) is used to get the next element from the data loader and it returns the next element from the data loader. X is the image tensor of the minibatch of images and y is the label tensor of the minibatch of images
#next(iter(data.train_dataloader())) is used to get the next element from the data loader and it returns the next element from the data loader because iter(data.train_dataloader()) is an iterator and next() function is used to get the next element from the iterator
X,y = next(iter(data.train_dataloader()))
print(X.shape,X.dtype,y.shape,y.dtype)
#time module is used to measure the time taken to load the dataset
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
#show_images can be used to visualize the images and the associated labels.
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
@d2l.add_to_class(FashionMNIST)
#visualize() method is used to visualize the images and the associated labels. batch is a tuple that stores the image tensor and the label tensor of the minibatch of images. nrows is the number of rows in the grid of images, ncols is the number of columns in the grid of images, and labels is a list that stores the names of the categories of Fashion-MNIST
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    #X, y = batch is used to get the image tensor and the label tensor of the minibatch of images
    X, y = batch
    #if not labels: is used to check whether the list labels is empty or not. If the list labels is empty, the names of the categories of Fashion-MNIST are stored in the list labels
    if not labels:
        #text_labels(y) is used to get the names of the categories of Fashion-MNIST for the given labels y from the batch and the names of the categories of Fashion-MNIST are stored in the list labels
        labels = self.text_labels(y)
        #show_images() is used to visualize the images and the associated labels. X.squeeze(1) is used to remove the channel dimension from the image tensor and it returns the image tensor without the channel dimension. nrows is the number of rows in the grid of images, ncols is the number of columns in the grid of images, titles=labels is used to set the titles of the images as the names of the categories of Fashion-MNIST
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
d2l.plt.show()