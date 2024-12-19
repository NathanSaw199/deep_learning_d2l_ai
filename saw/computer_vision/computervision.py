#pytorch computer vision 
#conputer vision library in pytorch torch vision

#torchvision.datasets: contains a number of popular datasets for computer vision tasks.
#torchvision.models: contains popular pre-trained models for computer vision.
#torchvision.transforms: contains common image transformations that can be chained together using Compose.
#torchvision.utils.data.DataLoader: a utility for loading and iterating over data in batches.
#torchvision.utils.data.Dataset: an abstract class representing a dataset.

import requests
from pathlib import Path
import torch
from torch import nn 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_functions import accuracy_fn
#getting a dataset 
#The FashionMNIST dataset contains images of clothing items from torchvision.datasets

#1. setup training data
train_data = datasets.FashionMNIST(
    root = 'data', #where to download the data
    train = True, #specifies training or test dataset
    download = True, #download the data or not
    transform=torchvision.transforms.ToTensor(), #convert the image to tensor
    target_transform=None #convert the target to tensor

)
test_data = datasets.FashionMNIST(
    root = 'data', #where to download the data
    train = False, #specifies training or test dataset
    download = True, #download the data or not
    transform=torchvision.transforms.ToTensor(), #convert the image to tensor
    target_transform=None #convert the target to tensor

)

print(len(train_data),len(test_data))


#the first training example 
image,label = train_data[0]
# print(image,label)
# print(image.shape,label)


class_names = train_data.classes
# print(class_names)

class_to_idx = train_data.class_to_idx
# print(class_to_idx)


image,label = train_data[0]
# print(image.shape)

#Creates a new figure using Matplotlib with a specified size of 9 inches by 9 inches.fig is the figure object that will hold the subplots.
# fig = plt.figure(figsize=(9,9))
# #Defines the number of rows and columns in the grid of subplots. This creates a 4x4 grid (16 subplots in total).
# rows,cols = 4,4
# #A loop that iterates from 1 to 16 (inclusive). Each iteration corresponds to creating one subplot.The range(1, rows*cols + 1) ensures that the subplot index matches Matplotlib’s 1-based numbering.
# for i in range(1,rows*cols+1):
#     #Randomly selects an index from the train_data dataset.torch.randint(0, len(train_data), size=[1]): Generates a random integer between 0 and the length of train_data (exclusive)..item(): Converts the single-element tensor to a Python integer. size=[1] ensures that a scalar tensor is returned.

#     random_idx = torch.randint(0,len(train_data),size =[1]).item()
#     #Retrieves an image and its corresponding label from the dataset at the randomly selected index.image: The input data (usually a tensor representing pixel values of the image).label: The target class label for the image.

#     image,label = train_data[random_idx]
#     #Adds a subplot to the figure.rows and cols specify the grid dimensions.i specifies the position of the subplot in the grid.
#     fig.add_subplot(rows,cols,i)    
#     #image.squeeze(): Removes any singleton dimensions from the tensor (e.g., converts a shape of [1, 28, 28] to [28, 28]).cmap='gray': Sets the colormap to grayscale, suitable for single-channel images like MNIST digits.

#     plt.imshow(image.squeeze(),cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False)
# # plt.show()


#2. prepare dataloader
# data is in tehe form of pytoch datasets. dataloar turns dataset into a python iterable. we want to turn data into batches( mini batches) of data because we can't pass the entire dataset into the model at once. we break it down to 32 images at a time (batch size = 32). parameter update its gradient per epoch.

#set up the batch size hyperparameter. hyperparameter is a parameter whose value is set before the learning process begins. In this case, the batch size is set to 32. wwe can set hyperparameters to different values and compare the model's performance.
Batch_Size = 32
#turn datasets into iterable (batches)
train_dataloader = DataLoader(dataset=train_data,batch_size=Batch_Size,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=Batch_Size,shuffle=False)


# print(train_dataloader,test_dataloader)
#batches of 32 images and labels and 1875 batches in total (60,000 images divided into batches of 32 images each). batchsize
# print(len(train_dataloader),len(test_dataloader))



#check out what is inside the training data loader
train_features_batch,train_labels_batch = next(iter(train_dataloader))

# print(train_features_batch.shape,train_labels_batch.shape)
#show a sample 
torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch),size=[1]).item()
image,label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.imshow(image.squeeze(),cmap='gray')
plt.title(class_names[label])
plt.axis(False)
print(f"images size {image.shape},label : {label},label size {label.shape}")
# plt.show()


#3. create a model baseline
#best practice to create baseline model is a simple model that can be improved upon.


##create a flatten layer 
flatten_model = nn.Flatten()
#get a single sample 
x = train_features_batch[0]

#flattlen the sample
output = flatten_model(x) 
print(x.shape) #original shape
print(output.shape) #flattened shape


class FashionMNISTModelV0(nn.Module):
    def __init__(self,input_shape :int,hidden_units : int,output_shape: int):
        super().__init__()
        self.Layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.Layer_stack(x)
torch.manual_seed(42)
#set up model with input paramters

model_0 = FashionMNISTModelV0(input_shape=28*28,hidden_units=10,output_shape=len(class_names)).to('cpu')

# print(model_0)
dummy_x = torch.rand([1,1,28,28])
# print(model_0(dummy_x))

#4. set up loss function and optimizer and evaluation metric

# loss function  for multi class data is cross entropy loss
# optimzer is stochastic gradient descent
# evaluation metric is accuracy


#download the helper function from learn pytoch repo 

if Path("helper_functions.py").is_file():
    print("Helper functions file found")
else:
    print("Downloading helper functions file")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb") as file:
        file.write(request.content)

# from helper_functions import calculate_accuracy
#import accuracy metric 
# set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)


#3.2 create a function to time the experiments
def print_train_time(start : float,end : float, device : torch.device = None):
    total_time = end - start
    print(f"Training time: on{device} : {total_time:.2f} seconds")
    return total_time

start_time = timer()
# write codes here
end_time = timer()

print(print_train_time(start_time,end_time,device='cpu'))